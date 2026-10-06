"""Raw canonical envelopes retain real CLI/API origin across presentations."""
import pytest
from neurofly_daemon import ContinuousExperimentRunner
from tests.test_assay_control_transactions import attached
from tests.test_lifecycle_control_transactions import complete


@pytest.mark.parametrize('policy', [{'continuous': True}, {'continuous': False, 'trial_seconds': 3.0}])
def test_runtime_policy_origin_and_command_identity_survive_new_segments(tmp_path, policy):
    runner, records, drain = attached(tmp_path)
    assert runner._live_observation()['override'] is None
    runner.step_once()
    result, tx = complete(runner, drain, dict(action='set_observation_policy', **policy))
    assert result['ack']['applied']
    config = runner._live_observation()
    assert config['override']['source'] == 'api'
    assert config['override']['manifest_run_id'] == runner.manifest.run_id
    assert f":policy-command:{tx['entry']['id']}" in config['config_id']
    assert config['override']['set_at_sim_s'] == runner.dt
    runner.step_once()
    for command in ({'action': 'reset_trial'}, {'action': 'set_learning', 'enabled': False}):
        complete(runner, drain, command)
        current = runner._live_observation()
        assert current['override']['source'] == 'api'
        assert current['override']['set_at_sim_s'] == runner.dt
        assert f":policy-command:{tx['entry']['id']}" in current['config_id']
    records.close()
    restarted = ContinuousExperimentRunner(initial_paradigm='open-arena', trial_length_s=None,
                                           output_dir=runner.output_dir, checkpoint_interval=3600)
    restored = restarted._live_observation()
    assert restored['override'] is None
    assert ':policy-command:' not in restored['config_id']
    assert not restarted.continuous and restarted.trial_length_s is None


def test_real_initial_cli_origin_is_preserved_until_runtime_change(tmp_path):
    runner, records, drain = attached(tmp_path, continuous=True)
    assert runner._live_observation()['override']['source'] == 'cli:--continuous'
    complete(runner, drain, {'action': 'set_learning', 'enabled': False})
    assert runner._live_observation()['override']['source'] == 'cli:--continuous'
    result, tx = complete(runner, drain, {'action': 'set_observation_policy', 'continuous': True})
    assert result['ack']['applied']
    assert runner._live_observation()['override']['source'] == 'api'
    records.close()
