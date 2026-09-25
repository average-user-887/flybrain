"""Local run queue for embodied runs (neurofly_body/run_queue.py)."""
import json

import pytest

from neurofly_body import cli, run_queue


def _fake_runner(calls, fail=()):
    def runner(argv, log_path):
        output = argv[argv.index("--output") + 1]
        calls.append(argv)
        log_path.write_text("ran\n")
        if any(name in output for name in fail):
            return 3
        from pathlib import Path
        Path(output).mkdir(parents=True)
        (Path(output) / "summary.json").write_text(json.dumps({"trajectory_sha256": "ab", "real_time_factor": 0.4}))
        return 0
    return runner


def test_jobs_run_in_order_one_at_a_time(tmp_path):
    q = tmp_path / "q"
    run_queue.add(q, "first", ["--duration", "1"])
    run_queue.add(q, "second", ["--duration", "2"])
    calls = []
    state = run_queue.run(q, runner=_fake_runner(calls), log=lambda _: None)
    assert [c[1] for c in calls] == ["1", "2"]
    assert state == {"pending": [], "running": [], "done": ["first", "second"], "failed": []}
    done = json.loads(next((q / "done").glob("*first.json")).read_text())
    assert done["exit_status"] == 0 and done["trajectory_sha256"] == "ab"
    assert done["output"].endswith("runs/first")


def test_failure_is_recorded_and_the_queue_continues(tmp_path):
    q = tmp_path / "q"
    run_queue.add(q, "bad", ["--duration", "1"])
    run_queue.add(q, "good", ["--duration", "1"])
    state = run_queue.run(q, runner=_fake_runner([], fail=("bad",)), log=lambda _: None)
    assert state["failed"] == ["bad"] and state["done"] == ["good"]
    assert json.loads(next((q / "failed").glob("*.json")).read_text())["exit_status"] == 3


def test_runner_exception_is_a_failed_job(tmp_path):
    q = tmp_path / "q"
    run_queue.add(q, "boom", [])

    def runner(argv, log_path):
        raise OSError("disk full")

    state = run_queue.run(q, runner=runner, log=lambda _: None)
    job = json.loads(next((q / "failed").glob("*.json")).read_text())
    assert state["failed"] == ["boom"] and job["error"] == "OSError: disk full"


def test_interrupted_job_is_failed_never_rerun(tmp_path):
    q = run_queue.init(tmp_path / "q")
    (q / "running" / "0001-half.json").write_text(json.dumps({"name": "half", "argv": []}))
    calls = []
    state = run_queue.run(q, runner=_fake_runner(calls), log=lambda _: None)
    assert calls == [] and state["failed"] == ["half"]
    assert "interrupted" in json.loads((q / "failed" / "0001-half.json").read_text())["error"]


def test_add_rejects_duplicates_bad_names_and_output(tmp_path):
    q = tmp_path / "q"
    run_queue.add(q, "a", [])
    with pytest.raises(FileExistsError):
        run_queue.add(q, "a", [])
    with pytest.raises(ValueError):
        run_queue.add(q, "../escape", [])
    with pytest.raises(ValueError, match="--output"):
        run_queue.add(q, "b", ["--output", "x"])


def test_cli_add_validates_run_arguments(tmp_path, capsys):
    q = tmp_path / "q"
    assert cli.main(["queue", "add", str(q), "ok", "--", "--duration", "1", "--seed", "2"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["queue", "add", str(q), "typo", "--", "--durration", "1"])
    assert cli.main(["queue", "status", str(q)]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["pending"] == ["ok"]


pytest.importorskip("flygym")


def test_real_subprocess_run(tmp_path):
    q = tmp_path / "q"
    run_queue.add(q, "modular", ["--controller", "modular", "--duration", "0.02", "--seed", "1"])
    state = run_queue.run(q, log=lambda _: None)
    assert state["done"] == ["modular"], (q / "logs" / "modular.log").read_text()
    assert (q / "runs" / "modular" / "telemetry.jsonl").is_file()
