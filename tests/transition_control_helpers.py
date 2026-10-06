"""Legacy test adapter: use real durable records and final queued command replies."""
import time
from learning_recorder import LearningRecorder, RecorderThread


def transition_command(runner, cmd, *, lock_held=False):
    # Legacy unit fixtures call from a locked block. Persistence/writer completion
    # must happen outside that lock, as it does in the real command transport.
    if lock_held:
        runner.lock.release()
    records = None
    previous = runner.learning_records
    try:
        drain = previous
        if drain is None:
            records = LearningRecorder(runner.output_dir / 'transition-test-records',
                                       session={'daemon_run_id': runner.run_id})
            drain = RecorderThread(runner, records, summary_interval=999)
            runner.attach_learning_records(drain)
        reply = runner.dispatch_command(cmd)
        if reply['status'] != 'queued':
            return reply
        tx = runner._pending_assay_control
        if not drain.liveness()['started']:
            deadline = time.monotonic() + 10
            while not tx['entry']['done'].is_set() and time.monotonic() < deadline:
                drain.poll_once()
                tx['entry']['done'].wait(0.01)
        assert tx['entry']['done'].wait(10), 'transition did not finish'
        if tx.get('writer'):
            tx['writer'].join(10)
            assert not tx['writer'].is_alive()
        if tx.get('failure_writer'):
            tx['failure_writer'].join(10)
            assert not tx['failure_writer'].is_alive()
        return tx['entry']['result']
    finally:
        if records is not None:
            records.close()
        runner.learning_records = previous
        if lock_held:
            runner.lock.acquire()
