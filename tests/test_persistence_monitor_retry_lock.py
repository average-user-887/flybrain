"""PersistenceMonitor.retry_due must read the channel table under the writers' lock."""
import threading

import neurofly_daemon as d


class _GuardedChannels(dict):
    """Fails any read made while the monitor lock is not held."""

    def __init__(self, monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitor = monitor
        self.reads = 0

    def get(self, key, default=None):
        assert self._monitor._lock.locked(), "channels read without the monitor lock"
        self.reads += 1
        return super().get(key, default)


def test_retry_due_reads_channels_under_lock():
    monitor = d.PersistenceMonitor()
    monitor.failed("checkpoint", OSError(5, "io"), now=1000.0)
    monitor.channels = _GuardedChannels(monitor, monitor.channels)
    assert monitor.retry_due("checkpoint", now=1000.0 + d.PersistenceMonitor.BACKOFF_START_S - 1) is False
    assert monitor.retry_due("checkpoint", now=1000.0 + d.PersistenceMonitor.BACKOFF_START_S) is True
    assert monitor.retry_due("never_failed", now=0.0) is True
    assert monitor.channels.reads == 3
    assert not monitor._lock.locked()


def test_retry_due_waits_for_a_writer_holding_the_lock():
    monitor = d.PersistenceMonitor()
    monitor.failed("checkpoint", OSError(5, "io"), now=1000.0)
    started, finished, result = threading.Event(), threading.Event(), []

    def reader():
        started.set()
        result.append(monitor.retry_due("checkpoint", now=1000.0))
        finished.set()

    with monitor._lock:
        thread = threading.Thread(target=reader)
        thread.start()
        assert started.wait(5)
        assert not finished.wait(0.2), "retry_due did not wait for the lock"
    thread.join(5)
    assert result == [False]


def test_backoff_doubles_to_cap_and_deadline_is_kept():
    monitor = d.PersistenceMonitor()
    seen = []
    for i in range(7):
        seen.append(monitor.failed("checkpoint", OSError(5, "io"), now=100.0 * i)["backoff_s"])
    assert seen == [30.0, 60.0, 120.0, 240.0, 480.0, 600.0, 600.0]
    deadline = monitor.channels["checkpoint"]["next_retry_at"]
    assert monitor.retry_due("checkpoint", now=deadline - 0.001) is False
    assert monitor.channels["checkpoint"]["next_retry_at"] == deadline   # a poll never moves it
    assert monitor.retry_due("checkpoint", now=deadline) is True
