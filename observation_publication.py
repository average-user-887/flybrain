"""In-memory ownership boundary for durable terminal-observation publication.

The caller holds its runner lock around every operation.  This coordinator is
deliberately single-threaded: it performs no I/O and invokes no callbacks.
Writer I/O happens after :meth:`claim_oldest` and before :meth:`acknowledge`.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from observation_envelopes import validate_observation_envelope


KEY_FIELDS = (
    "daemon_run_id", "run_id", "instance_id", "segment_id", "presentation_id",
)


class ObservationPublicationError(ValueError):
    """A publication operation would violate queue ownership or acknowledgement."""


class ObservationQueueFull(ObservationPublicationError):
    """A new immutable result cannot be retained without exceeding capacity."""


class ObservationPayloadConflict(ObservationPublicationError):
    """The same complete key was presented with different canonical bytes."""


@dataclass(frozen=True)
class _FrozenObservation:
    key: Tuple[str, ...]
    key_bytes: bytes
    owner: Tuple[str, ...]
    identity_bytes: bytes
    payload_bytes: bytes
    payload_sha256: str


@dataclass
class _PendingMetadata:
    frozen: _FrozenObservation
    phase: str = "pending"
    attempts: int = 0
    token: Optional[str] = None
    failure_bytes: Optional[bytes] = None


def _canonical_bytes(value: Any, label: str) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ObservationPublicationError(f"{label} must be strict finite JSON: {exc}") from exc
    return text.encode("utf-8")


def _detached(value: bytes) -> Any:
    return json.loads(value.decode("utf-8"))


class ObservationPublicationQueue:
    """Bounded pending queue with one active writer claim and durable receipts."""

    def __init__(self, capacity: int = 32, *, terminal_capacity: int = 64):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if (isinstance(terminal_capacity, bool) or not isinstance(terminal_capacity, int)
                or terminal_capacity < 1):
            raise ValueError("terminal_capacity must be a positive integer")
        self.capacity = capacity
        self.terminal_capacity = terminal_capacity
        self._pending: "OrderedDict[Tuple[str, ...], _PendingMetadata]" = OrderedDict()
        self._active_token: Optional[str] = None
        # Payload history is intentionally bounded to the latest terminal per
        # recently used owner.  Key/digest metadata alone provides idempotency.
        self._durable_digests: Dict[Tuple[str, ...], str] = {}
        self._last_terminal: "OrderedDict[Tuple[str, ...], Tuple[_FrozenObservation, bytes]]" = OrderedDict()

    @staticmethod
    def _freeze(envelope: Dict[str, Any]) -> _FrozenObservation:
        try:
            validated = validate_observation_envelope(envelope)
        except (TypeError, ValueError) as exc:
            raise ObservationPublicationError(f"invalid observation envelope: {exc}") from exc
        records = validated.get("records")
        if not isinstance(records, dict) or not records or not all(
                isinstance(record, dict) and record.get("final") is True
                for record in records.values()):
            raise ObservationPublicationError("enqueue requires a frozen terminal envelope")
        if validated.get("completeness") not in ("complete", "incomplete"):
            raise ObservationPublicationError("terminal completeness is required")
        if not isinstance(validated.get("end_reason"), str) or not validated["end_reason"]:
            raise ObservationPublicationError("terminal end_reason is required")

        identity = validated.get("identity")
        values = {
            "daemon_run_id": identity.get("daemon_run_id") if isinstance(identity, dict) else None,
            "run_id": identity.get("run_id") if isinstance(identity, dict) else None,
            "instance_id": identity.get("instance_id") if isinstance(identity, dict) else None,
            "segment_id": validated.get("segment_id"),
            "presentation_id": validated.get("presentation_id"),
        }
        for field in KEY_FIELDS:
            if not isinstance(values[field], str) or not values[field]:
                raise ObservationPublicationError(f"key field {field} must be a nonempty string")
        owner_values = tuple(identity.get(field) for field in (
            "assay", "backend", "instance_id", "brain_id"))
        if any(not isinstance(value, str) or not value for value in owner_values):
            raise ObservationPublicationError("owner assay/backend/instance_id/brain_id must be nonempty")
        key = tuple(values[field] for field in KEY_FIELDS)
        payload_bytes = _canonical_bytes(validated, "observation")
        return _FrozenObservation(
            key=key,
            key_bytes=_canonical_bytes(dict(zip(KEY_FIELDS, key)), "observation key"),
            owner=owner_values,
            identity_bytes=_canonical_bytes(identity, "observation identity"),
            payload_bytes=payload_bytes,
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        )

    @staticmethod
    def _key_dict(frozen: _FrozenObservation) -> Dict[str, str]:
        return _detached(frozen.key_bytes)

    @staticmethod
    def _public_frozen(frozen: _FrozenObservation) -> Dict[str, Any]:
        return {
            "observation_key": ObservationPublicationQueue._key_dict(frozen),
            "payload_sha256": frozen.payload_sha256,
            "identity": _detached(frozen.identity_bytes),
            "observation": _detached(frozen.payload_bytes),
        }

    def enqueue(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        frozen = self._freeze(envelope)
        existing = self._pending.get(frozen.key)
        if existing is not None:
            if existing.frozen.payload_bytes != frozen.payload_bytes:
                raise ObservationPayloadConflict("same observation key has a different payload")
            return {"status": "pending", **self._public_frozen(existing.frozen)}
        durable_digest = self._durable_digests.get(frozen.key)
        if durable_digest is not None:
            if durable_digest != frozen.payload_sha256:
                raise ObservationPayloadConflict("durable observation key has a different payload")
            return {"status": "durable", **self._public_frozen(frozen)}
        if len(self._pending) >= self.capacity:
            raise ObservationQueueFull(
                f"pending observation capacity {self.capacity} reached; no entry was dropped")
        self._pending[frozen.key] = _PendingMetadata(frozen=frozen)
        return {"status": "enqueued", **self._public_frozen(frozen)}

    def claim_oldest(self) -> Optional[Dict[str, Any]]:
        if self._active_token is not None:
            raise ObservationPublicationError("an observation publication claim is already active")
        if not self._pending:
            return None
        entry = next(iter(self._pending.values()))
        if entry.phase != "pending":
            return None
        token = secrets.token_urlsafe(24)
        entry.phase = "claimed"
        entry.attempts += 1
        entry.token = token
        self._active_token = token
        return {
            "attempt_token": token,
            "attempt": entry.attempts,
            **self._public_frozen(entry.frozen),
        }

    def _claimed(self, token: str) -> _PendingMetadata:
        if not isinstance(token, str) or not token or token != self._active_token:
            raise ObservationPublicationError("stale or unknown attempt token")
        if not self._pending:
            raise ObservationPublicationError("no pending observation owns the attempt token")
        entry = next(iter(self._pending.values()))
        if entry.phase != "claimed" or entry.token != token:
            raise ObservationPublicationError("attempt token does not own the queue head")
        return entry

    def mark_failed(self, token: str, failure: Any) -> Dict[str, Any]:
        entry = self._claimed(token)
        failure_bytes = _canonical_bytes(failure, "failure metadata")
        entry.phase = "failed"
        entry.failure_bytes = failure_bytes
        entry.token = None
        self._active_token = None
        return {
            "phase": entry.phase, "attempt": entry.attempts,
            "failure": _detached(failure_bytes),
            "observation_key": self._key_dict(entry.frozen),
            "payload_sha256": entry.frozen.payload_sha256,
        }

    def retry_failed(self) -> Dict[str, Any]:
        if self._active_token is not None:
            raise ObservationPublicationError("cannot retry while a claim is active")
        if not self._pending:
            raise ObservationPublicationError("no pending observation to retry")
        entry = next(iter(self._pending.values()))
        if entry.phase != "failed":
            raise ObservationPublicationError("oldest pending observation has not failed")
        entry.phase = "pending"
        return {
            "phase": entry.phase, "attempts": entry.attempts,
            "observation_key": self._key_dict(entry.frozen),
            "payload_sha256": entry.frozen.payload_sha256,
        }

    @staticmethod
    def _validate_receipt(receipt: Dict[str, Any], frozen: _FrozenObservation) -> bytes:
        receipt_bytes = _canonical_bytes(receipt, "durable receipt")
        value = _detached(receipt_bytes)
        if not isinstance(value, dict) or value.get("durable") is not True:
            raise ObservationPublicationError("receipt durable must be literally true")
        if value.get("observation_key") != ObservationPublicationQueue._key_dict(frozen):
            raise ObservationPublicationError("receipt observation key does not match the claim")
        if value.get("payload_sha256") != frozen.payload_sha256:
            raise ObservationPublicationError("receipt payload digest does not match the claim")
        file_name = value.get("file")
        if (not isinstance(file_name, str) or not file_name or file_name in (".", "..")
                or "/" in file_name or "\\" in file_name):
            raise ObservationPublicationError("receipt file must be a nonempty ledger basename")
        line, offset = value.get("line"), value.get("offset")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise ObservationPublicationError("receipt line must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ObservationPublicationError("receipt offset must be a nonnegative integer")
        if not isinstance(value.get("idempotent"), bool):
            raise ObservationPublicationError("receipt idempotent must be a bool")
        return receipt_bytes

    def acknowledge(self, token: str, receipt: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._claimed(token)
        receipt_bytes = self._validate_receipt(receipt, entry.frozen)
        frozen = entry.frozen
        del self._pending[frozen.key]
        self._active_token = None
        self._durable_digests[frozen.key] = frozen.payload_sha256
        self._last_terminal[frozen.owner] = (frozen, receipt_bytes)
        self._last_terminal.move_to_end(frozen.owner)
        while len(self._last_terminal) > self.terminal_capacity:
            self._last_terminal.popitem(last=False)
        return {
            "status": "durable",
            **self._public_frozen(frozen),
            "receipt": _detached(receipt_bytes),
        }

    def pending_status(self, limit: int = 10) -> Dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        limit = min(limit, self.capacity)
        entries = []
        for entry in list(self._pending.values())[:limit]:
            entries.append({
                "observation_key": self._key_dict(entry.frozen),
                "payload_sha256": entry.frozen.payload_sha256,
                "identity": _detached(entry.frozen.identity_bytes),
                "phase": entry.phase,
                "attempts": entry.attempts,
                "failure": _detached(entry.failure_bytes) if entry.failure_bytes else None,
            })
        return {
            "pending_count": len(self._pending),
            "capacity": self.capacity,
            "active_claim": self._active_token is not None,
            "entries": entries,
        }

    def last_terminal(self, *, assay: str, backend: str, instance_id: str,
                      brain_id: str) -> Optional[Dict[str, Any]]:
        owner = (assay, backend, instance_id, brain_id)
        found = self._last_terminal.get(owner)
        if found is None:
            return None
        frozen, receipt_bytes = found
        return {
            **self._public_frozen(frozen),
            "receipt": _detached(receipt_bytes),
        }
