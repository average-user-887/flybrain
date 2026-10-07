"""abort(defer_cleanup=...) : both values revoke at once, never block, and keep the partial files."""
import json
import threading
import pytest
from neurofly import recording as r
from tests.test_recording_streaming import _writer


@pytest.mark.parametrize('defer', [True, False])
def test_abort_never_blocks_cleans_in_background_and_keeps_partial(tmp_path, defer):
    _, _, writer = _writer(tmp_path)
    assert writer._partial.exists()
    done = threading.Event()
    # A finalizer holding the lock must not delay abort(); cleanup waits for it.
    assert writer._finalization_lock.acquire(timeout=1)
    try:
        t = threading.Thread(target=lambda: (writer.abort('boom', defer_cleanup=defer), done.set()))
        t.start()
        assert done.wait(2), 'abort waited for cleanup'
        t.join(2)
        assert writer.closed and writer.aborted == 'boom' and not writer.completed
        assert r.recording_writer_invalid_reason(writer.path) is not None
        assert not writer._raw.closed, 'cleanup ran while a finalizer held the lock'
    finally:
        writer._finalization_lock.release()
    first_thread = writer._cleanup_thread
    assert first_thread is not None
    first_thread.join(5)
    assert not first_thread.is_alive() and writer._raw.closed
    assert json.loads((writer._projection.spool.path / 'status.json').read_text())['state'] == 'failed'
    # The partial recording and its rows are retained, nothing was published.
    assert writer._partial.exists() and not writer.path.exists() and not writer._done_path.exists()
    assert len(list(writer._projection.sources)) == 1
    # A second abort (either value) neither restarts cleanup nor changes the reason.
    writer.abort('again', defer_cleanup=not defer)
    assert writer._cleanup_thread is first_thread and writer.aborted == 'boom'
