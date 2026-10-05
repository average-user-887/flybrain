from pathlib import Path
p = Path('neurofly_daemon.py'); s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


rep("""        self.paused = False
        self.last_error = None
        self.run_id = uuid.uuid4().hex""", """        self.paused = False
        # Fail-safe halt: an exception inside a step stops the simulation (nothing
        # advances while ``last_error`` is set) instead of stepping a broken state.
        # The halt is reported (status "error", packet ``halted``, the dashboard
        # pill) and lifted only by a command that rebuilds the controller and world
        # (a successful assay or backend switch); see _halt_on_error/_clear_error.
        self.last_error = None
        self.error_detail: Optional[Dict[str, Any]] = None
        self.cleared_errors: deque = deque(maxlen=16)   # lifted halts, newest last
        self.run_id = uuid.uuid4().hex""")
rep("""        except Exception as step_err:
            print(f"[Daemon] Exception in arena.step: {step_err}", file=sys.stderr)
            self.last_error = str(step_err)
            self.active_brain.log("simulation_error", error=self.last_error, step=self.total_steps,
                                  run_id=self.run_id, segment_id=self.segment_id)
            self._publish_due = True""", """        except Exception as step_err:
            print(f"[Daemon] Exception in arena.step: {step_err}", file=sys.stderr)
            self._halt_on_error(step_err)
            self.active_brain.log("simulation_error", error=self.last_error, step=self.total_steps,
                                  run_id=self.run_id, segment_id=self.segment_id,
                                  error_type=self.error_detail["type"])
            self._publish_due = True""")
rep("""    def _trial_end_reason(self, step_result""", """    HALT_RECOVERY = ("The simulation is halted by this error and does not advance. Select an assay "
                     "(selecting the same one retries it) or switch the controller backend to rebuild "
                     "the controller and resume. Pausing, resuming or changing speed does not lift it.")

    def _halt_on_error(self, exc: BaseException):
        \"\"\"Record the step failure that halts the simulation. Caller holds the lock.\"\"\"
        registry = self.registry if self.graph_mode else None
        active = registry.active if registry is not None else None
        self.last_error = str(exc) or type(exc).__name__
        self.error_detail = {
            "message": self.last_error, "type": type(exc).__name__, "step": self.total_steps,
            "sim_time_s": round(self.total_steps * self.dt, 5), "paradigm": self.active_paradigm_id,
            "backend": self.backend, "instance_id": active.instance_id if active is not None else None,
            "at": round(time.time(), 3), "recover": self.HALT_RECOVERY}

    def _clear_error(self, cleared_by: str) -> Optional[Dict[str, Any]]:
        \"\"\"Lift a halt after a successful rebuild; the cleared error is kept and logged.\"\"\"
        if self.last_error is None:
            return None
        detail = dict(self.error_detail or {"message": self.last_error})
        detail.update(cleared_by=cleared_by, cleared_at=round(time.time(), 3), cleared_step=self.total_steps,
                      resumed_paradigm=self.active_paradigm_id, resumed_backend=self.backend)
        detail.pop("recover", None)
        self.cleared_errors.append(detail)
        self.last_error = None
        self.error_detail = None
        self.active_brain.log("simulation_error_cleared", error=detail["message"], cleared_by=cleared_by,
                              step=self.total_steps, run_id=self.run_id, segment_id=self.segment_id)
        print(f"[Daemon] Halt lifted by {cleared_by}: {detail['message']}", flush=True)
        self._publish_due = True
        return detail

    def _trial_end_reason(self, step_result""")
rep("""            "paused": self.paused,
            "error": self.last_error,
            "sim_time_s\"""", """            "paused": self.paused,
            "error": self.last_error,
            "halted": self.last_error is not None,
            "error_detail": self.error_detail,
            "sim_time_s\"""")
rep("""                             "identity": self.identity()}
        self._publish_due = True
        return result""", """                             "identity": self.identity(),
                             # Applied, but the simulation still does not advance.
                             "halted_by_error": self.last_error}
        self._publish_due = True
        return result""")
rep("""            try:
                self._init_arena(target)
            except (ValueError, OSError, RuntimeError) as exc:
                # RuntimeError covers BackendError, GraphUnavailable and checkpoint errors.
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            return {"status": "ok", "active_paradigm": self.active_paradigm_id, "identity": self.identity()}""",
"""            try:
                self._init_arena(target)
            except (ValueError, OSError, RuntimeError) as exc:
                # RuntimeError covers BackendError, GraphUnavailable and checkpoint errors.
                # A failed switch rebuilt nothing, so a standing halt stays.
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            # The target's instance, world and arena are now active: lift a halt.
            cleared = self._clear_error("switch_paradigm")
            return {"status": "ok", "active_paradigm": self.active_paradigm_id, "identity": self.identity(),
                    "cleared_error": cleared}""")
rep("""            try:
                self._switch_backend(target)
            except Exception as exc:
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            return {"status": "ok", "backend": self.backend, "identity": self.identity()}""",
"""            previous = self.backend
            try:
                self._switch_backend(target)
            except Exception as exc:
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            # Only a real change rebuilds the controller; re-selecting it is a no-op.
            cleared = self._clear_error("switch_backend") if self.backend != previous else None
            return {"status": "ok", "backend": self.backend, "identity": self.identity(),
                    "cleared_error": cleared}""")
rep("""            "status": "error" if self.runner.last_error else "online",
            "error": self.runner.last_error,""", """            "status": "error" if self.runner.last_error else "online",
            "error": self.runner.last_error,
            # A halted run is not paused: nothing advances until a rebuild lifts it.
            "halted": self.runner.last_error is not None,
            "error_detail": getattr(self.runner, "error_detail", None),
            "cleared_errors": list(getattr(self.runner, "cleared_errors", ())),""")
p.write_text(s)
print("patched")
