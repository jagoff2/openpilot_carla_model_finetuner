import json
import subprocess
import sys

import pytest

from lane_dataset.admission import build_labels, qualify_candidate
from lane_dataset.postprocess import shard_postprocess_lock
from lane_dataset.review_video import export_clip


def shard_fixture(root,status="development"):
    shard = root/"episode"
    shard.mkdir()
    (shard/"manifest.json").write_text(json.dumps(dict(status=status)))
    return shard


@pytest.mark.parametrize("writer", [build_labels,qualify_candidate,export_clip])
def test_concurrent_postprocessors_fail_before_touching_manifest(tmp_path,writer):
    shard = shard_fixture(tmp_path)
    before = (shard/"manifest.json").read_bytes()
    with shard_postprocess_lock(shard):
        with pytest.raises(RuntimeError,match="postprocessing already active"):
            writer(shard)
    assert (shard/"manifest.json").read_bytes() == before
    assert list(shard.iterdir()) == [shard/"manifest.json"]


def test_postprocess_lock_excludes_other_process_and_releases_after_crash(tmp_path):
    shard = shard_fixture(tmp_path)
    probe = "from lane_dataset.postprocess import shard_postprocess_lock; import sys; "
    probe += "\ntry:\n with shard_postprocess_lock(sys.argv[1]): pass\nexcept RuntimeError: sys.exit(17)\n"
    with shard_postprocess_lock(shard):
        blocked = subprocess.run([sys.executable,"-c",probe,str(shard)],capture_output=True,text=True)
        assert blocked.returncode == 17, blocked.stderr
    crash = "from lane_dataset.postprocess import shard_postprocess_lock; import sys,os; "
    crash += "\nwith shard_postprocess_lock(sys.argv[1]): os._exit(0)\n"
    finished = subprocess.run([sys.executable,"-c",crash,str(shard)],capture_output=True,text=True)
    assert finished.returncode == 0, finished.stderr
    with shard_postprocess_lock(shard):
        pass


@pytest.mark.parametrize("writer", [build_labels,qualify_candidate,export_clip])
def test_accepted_shards_remain_immutable_without_creating_lock_artifacts(tmp_path,writer):
    shard = shard_fixture(tmp_path,status="accepted")
    before = (shard/"manifest.json").read_bytes()
    with pytest.raises(FileExistsError,match="accepted shards are immutable"):
        writer(shard)
    assert (shard/"manifest.json").read_bytes() == before
    assert not (tmp_path/".postprocess-locks").exists()
