import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lane_dataset import benchmark


def captured(worker, ticks=80, **changes):
    manifest = dict(worker=worker, port=2400+100*worker, measured_throughput=dict(ticks=ticks))
    manifest.update(changes)
    return dict(capture=dict(manifest=manifest))


def test_completion_requires_distinct_workers_and_each_full_capture():
    assert benchmark.captures_complete([captured(0), captured(1)], 2, 80, "benchmark")
    assert not benchmark.captures_complete([captured(0), captured(0)], 2, 80, "benchmark")
    assert not benchmark.captures_complete([captured(0, 79), captured(1, 81)], 2, 80, "benchmark")
    assert not benchmark.captures_complete([captured(0), captured(1, port=2400)], 2, 80, "benchmark")
    assert not benchmark.captures_complete([captured(0, error="capture failed")], 1, 80, "develop")
    assert not benchmark.captures_complete([dict(capture={})], 1, 80, "benchmark")


def test_generation_failure_records_are_not_successful_throughput():
    report = dict(capture=dict(worker=0, episodes=[dict(status="accepted")]))
    assert benchmark.captures_complete([report], 1, 80, "generate")
    report["capture"]["episodes"][0]["status"] = "retry_limit_reached"
    assert not benchmark.captures_complete([report], 1, 80, "generate")


class Process:
    def __init__(self, returncode=None, stubborn=False):
        self.returncode = returncode
        self.stubborn = stubborn
        self.terminated = self.killed = self.waited = False
    def poll(self):
        return self.returncode
    def terminate(self):
        self.terminated = True
    def wait(self, timeout):
        self.waited = True
        if self.stubborn and not self.killed:
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = -9 if self.killed else -15
        return self.returncode
    def kill(self):
        self.killed = True


def test_shutdown_reaps_children_and_kills_stubborn_worker():
    normal, stubborn, finished = Process(), Process(stubborn=True), Process(0)
    benchmark.stop_workers([normal, stubborn, finished])
    assert normal.terminated and normal.waited
    assert stubborn.terminated and stubborn.waited and stubborn.killed
    assert not finished.terminated


def test_first_failed_worker_cancels_sibling_and_speed_is_forwarded(tmp_path, monkeypatch):
    failed, sibling = Process(2), Process()
    children = iter([failed, sibling])
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return next(children)
    fake_carla = SimpleNamespace(Client=lambda *args: SimpleNamespace(set_timeout=lambda value: None, get_server_version=lambda: "0.9.16"))
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "gpu_sample", lambda: pytest.fail("failed workers must stop before another sample"))
    report = benchmark.run(tmp_path, 2, 80, "benchmark", tmp_path / "benchmark", speed=2.5)
    assert not report["complete_pipeline"] and sibling.terminated and sibling.waited
    assert report["exit_codes"] == [2, -15]
    for worker, command in enumerate(commands):
        assert command[command.index("--speed")+1] == "2.5"
        assert command[command.index("--port")+1] == str(2400+100*worker)
        assert command[command.index("--worker")+1] == str(worker)


def test_gpu_probe_has_a_bounded_timeout(monkeypatch):
    seen = {}
    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="0, GPU-fixture, 100, 40", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.gpu_sample()["csv"].startswith("0, GPU-fixture")
    assert seen["timeout"] == 10


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows launcher assertion")
def test_launcher_setting_checks_exact_values_and_rejects_duplicates():
    launcher = Path(__file__).resolve().parents[1] / "launch.ps1"
    script = r'''
$ast = [System.Management.Automation.Language.Parser]::ParseFile($args[0], [ref]$null, [ref]$null)
$definition = $ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-ExactCommandSetting'}, $true)
Invoke-Expression $definition.Extent.Text
$checks = @(
    (Test-ExactCommandSetting '-graphicsadapter=1 -quality-level=Epic' '-graphicsadapter=1'),
    (Test-ExactCommandSetting '-graphicsadapter=10' '-graphicsadapter=1'),
    (Test-ExactCommandSetting '-carla-rpc-port=24000' '-carla-rpc-port=2400'),
    (Test-ExactCommandSetting '-ExecCmds=r.DynamicRes.OperationMode=0,r.ScreenPercentage=100' 'r.ScreenPercentage=100'),
    (Test-ExactCommandSetting '-ExecCmds=r.DynamicRes.OperationMode=0,r.ScreenPercentage=1000' 'r.ScreenPercentage=100'),
    (Test-ExactCommandSetting '-quality-level=Epic -quality-level=Low' '-quality-level=Epic')
)
$checks | ConvertTo-Json -Compress
'''
    # Parse just the pure check function; this test never launches CARLA.
    command = "& { " + script + " } '" + str(launcher).replace("'", "''") + "'"
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == [True, False, False, True, False, False]
