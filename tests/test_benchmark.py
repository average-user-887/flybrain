import json

from scripts.benchmark import main


def test_quick_benchmark_writes_receipt(tmp_path):
    out = tmp_path / 'bench.json'
    assert main(['--quick', '--stages', 'brain-synthetic,daemon-modular', '--out', str(out)]) == 0
    receipt = json.loads(out.read_text())
    assert receipt['schema'] == 'neurofly.host-benchmark.v1'
    for stage in ('brain-synthetic', 'daemon-modular'):
        assert receipt['stages'][stage]['sim_s_per_wall_s'] > 0
