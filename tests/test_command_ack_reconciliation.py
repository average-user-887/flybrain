"""CARD63: a backend request whose HTTP reply is lost can still be matched to its
exact outcome.

Root's 8c3390d browser run: a paused fixed->plastic switch answered nothing within
the dashboard's request timeout, so the dashboard had no daemon command id and
showed "outcome unknown" forever while the stream already showed plastic.  The
dashboard now sends its own ``client_command_id``; the daemon echoes it on the
reply and the final acknowledgement, reports it on the pending-control progress,
and answers GET /api/command_ack for it.  An id it has no record of is
``unknown``, never success.  Synthetic test graph; no behavioural claim.
"""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from learning_recorder import LearningRecorder, RecorderThread
import neurofly_daemon
from neurofly_daemon import NeuroflyHTTPHandler
from tests.test_wp5_live_loop import graph_runner


def _dispatch(runner, cmd, *, while_pending=None):
    """Real transport order: dispatch, observe the pending transaction, drain durable writes."""
    records = LearningRecorder(runner.output_dir / 'ack-test-records', session={'daemon_run_id': runner.run_id})
    drain = RecorderThread(runner, records, summary_interval=999)
    previous = runner.learning_records
    runner.attach_learning_records(drain)
    try:
        reply = runner.dispatch_command(cmd)
        if reply['status'] != 'queued':
            return reply, reply
        tx = runner._pending_assay_control
        if while_pending is not None:
            while_pending(reply, tx)
        deadline = time.monotonic() + 10
        while not tx['entry']['done'].is_set() and time.monotonic() < deadline:
            drain.poll_once()
            tx['entry']['done'].wait(0.01)
        assert tx['entry']['done'].is_set()
        for key in ('writer', 'failure_writer'):
            if tx.get(key):
                tx[key].join(10)
        return reply, tx['entry']['result']
    finally:
        records.close()
        runner.learning_records = previous


def test_switch_reply_and_final_ack_carry_the_request_id_and_are_findable(tmp_path):
    runner = graph_runner(tmp_path, paradigm='open-arena')
    seen = {}

    def pending(reply, tx):
        assert reply['client_command_id'] == 'nf-req-1'
        assert 'client_command_id' not in tx['entry']['cmd']           # transport metadata, not the command
        look = runner.command_ack_lookup(client_command_id='nf-req-1')
        assert look['state'] == 'pending' and look['daemon_run_id'] == runner.run_id
        progress = runner.observation_lifecycle_status()['pending_control']
        assert progress['client_command_id'] == 'nf-req-1' and progress['name'] == 't-maze'
        assert progress['elapsed_s'] >= 0 and progress['persistence_phase']
        assert look['pending_control']['command_id'] == progress['command_id'] == reply['command_id']
        seen['command_id'] = reply['command_id']
    reply, final = _dispatch(runner, {'action': 'switch_paradigm', 'paradigm': 't-maze',
                                      'client_command_id': 'nf-req-1'}, while_pending=pending)
    assert final['status'] == 'ok' and final['ack']['applied'] is True
    assert final['client_command_id'] == 'nf-req-1' and final['command_id'] == seen['command_id']
    assert any(a.get('client_command_id') == 'nf-req-1' for a in runner.command_acks)
    by_client = runner.command_ack_lookup(client_command_id='nf-req-1')
    by_daemon = runner.command_ack_lookup(command_id=seen['command_id'])
    assert by_client['state'] == by_daemon['state'] == 'acknowledged'
    assert by_client['ack']['ack']['identity'] == final['ack']['identity']


def test_refusal_is_acknowledged_under_the_request_id(tmp_path):
    runner = graph_runner(tmp_path, paradigm='open-arena')
    reply, final = _dispatch(runner, {'action': 'switch_backend', 'backend': 'no-such-backend',
                                      'client_command_id': 'nf-refused'})
    assert final['status'] == 'error'
    look = runner.command_ack_lookup(client_command_id='nf-refused')
    assert look['state'] == 'acknowledged' and look['ack']['status'] == 'error'
    assert runner.backend == 'connectome-fixed'


def test_invalid_request_id_is_refused_before_anything_is_applied(tmp_path):
    runner = graph_runner(tmp_path, paradigm='open-arena')
    before = runner.identity()
    result = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze',
                                      'client_command_id': 'bad id with spaces'})
    assert result['status'] == 'error' and result['applied'] is False
    assert runner.identity() == before and runner._pending_assay_control is None


def test_unknown_or_other_process_ids_are_unknown_never_success(tmp_path):
    first = graph_runner(tmp_path / 'a', paradigm='open-arena')
    _dispatch(first, {'action': 'switch_paradigm', 'paradigm': 't-maze', 'client_command_id': 'nf-before-restart'})
    assert first.command_ack_lookup(client_command_id='nf-before-restart')['state'] == 'acknowledged'
    restarted = graph_runner(tmp_path / 'a', paradigm='open-arena')        # same store, new process
    look = restarted.command_ack_lookup(client_command_id='nf-before-restart')
    assert look['state'] == 'unknown' and look['daemon_run_id'] == restarted.run_id != first.run_id
    assert 'ack' not in look
    assert restarted.command_ack_lookup(client_command_id='nf-never-sent')['state'] == 'unknown'
    assert restarted.command_ack_lookup()['state'] == 'unknown'


def test_ack_index_is_bounded(tmp_path):
    COMMAND_ACK_INDEX_SIZE = getattr(neurofly_daemon, 'COMMAND_ACK_INDEX_SIZE', 256)
    runner = graph_runner(tmp_path, paradigm='open-arena')
    for i in range(COMMAND_ACK_INDEX_SIZE + 40):
        runner._index_command_ack({'status': 'ok', 'command_id': f'd-{i}', 'client_command_id': f'c-{i}'})
    assert len(runner.command_ack_index) <= COMMAND_ACK_INDEX_SIZE
    assert runner.command_ack_lookup(client_command_id=f'c-{COMMAND_ACK_INDEX_SIZE + 39}')['state'] == 'acknowledged'
    assert runner.command_ack_lookup(client_command_id='c-0')['state'] == 'unknown'


def test_command_ack_endpoint_is_read_only_and_needs_no_simulation_lock(tmp_path):
    runner = graph_runner(tmp_path, paradigm='open-arena')
    _dispatch(runner, {'action': 'switch_paradigm', 'paradigm': 't-maze', 'client_command_id': 'nf-http'})
    handler = type('H', (NeuroflyHTTPHandler,), {'runner': runner})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}/api/command_ack'
        with runner.lock:          # a long rebuild holding the simulation lock does not block the lookup
            got = json.loads(urllib.request.urlopen(base + '?client_command_id=nf-http', timeout=5).read())
            missing = json.loads(urllib.request.urlopen(base + '?client_command_id=nf-none', timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()
    assert got['state'] == 'acknowledged' and got['ack']['client_command_id'] == 'nf-http'
    assert missing['state'] == 'unknown'


def test_lookup_returns_the_complete_final_ack_of_a_durable_control(tmp_path):
    """CARD74: the indexed copy was taken before observation_key, payload_sha256 and
    latency were added, so the lookup omitted the terminal evidence."""
    runner = graph_runner(tmp_path, paradigm='open-arena')
    early = []

    def pending(reply, tx):
        # Every intermediate phase until the final ack: nothing is acknowledged early.
        assert not tx['entry']['done'].is_set()
        early.append(runner.command_ack_lookup(client_command_id='nf-durable')['state'])
        assert 'nf-durable' not in runner.command_ack_index
        assert reply['command_id'] not in runner.command_ack_index
        assert not any(a.get('client_command_id') == 'nf-durable' for a in runner.command_acks)

    reply, final = _dispatch(runner, {'action': 'switch_paradigm', 'paradigm': 't-maze',
                                      'client_command_id': 'nf-durable'}, while_pending=pending)
    assert early and set(early) <= {'pending', 'queued', 'applying'}
    assert final['status'] == 'ok' and final['ack']['applied'] is True
    assert final['observation_key'] and final['payload_sha256']
    assert 'latency_ms' in final['ack']
    stream = [a for a in runner.command_acks if a.get('client_command_id') == 'nf-durable']
    assert len(stream) == 1
    looked = runner.command_ack_lookup(client_command_id='nf-durable')
    assert looked['state'] == 'acknowledged'
    canon = lambda d: json.loads(json.dumps(d, sort_keys=True))
    assert canon(looked['ack']) == canon(final) == canon(stream[0])
    assert looked['ack']['observation_key'] == final['observation_key']
    assert looked['ack']['payload_sha256'] == final['payload_sha256']
    assert canon(runner.command_ack_lookup(command_id=final['command_id'])['ack']) == canon(final)


def test_no_acknowledgement_is_indexed_at_any_step_before_the_save_completes(tmp_path):
    runner = graph_runner(tmp_path, paradigm='open-arena')
    records = LearningRecorder(runner.output_dir / 'stepwise-records', session={'daemon_run_id': runner.run_id})
    drain = RecorderThread(runner, records, summary_interval=999)
    previous = runner.learning_records
    runner.attach_learning_records(drain)
    states = []
    try:
        reply = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze',
                                         'client_command_id': 'nf-stepwise'})
        assert reply['status'] == 'queued'
        tx = runner._pending_assay_control
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            done = tx['entry']['done'].is_set()
            state = runner.command_ack_lookup(client_command_id='nf-stepwise')['state']
            states.append((tx.get('phase'), state, done))
            if not done:
                assert state != 'acknowledged', states
            else:
                break
            drain.poll_once()
            tx['entry']['done'].wait(0.005)
        for key in ('writer', 'failure_writer'):
            if tx.get(key):
                tx[key].join(10)
    finally:
        records.close()
        runner.learning_records = previous
    assert states[-1][2] is True and states[-1][1] == 'acknowledged'
    assert any(not done for _, _, done in states)
