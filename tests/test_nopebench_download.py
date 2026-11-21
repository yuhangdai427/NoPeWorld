"""nopebench-download on a tiny synthetic archive set: selection by task, split and case, sha256 check, extraction, markers, cleanup."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from nopebench import cli, download


def make_tar(path, files):
    with tarfile.open(path, "w") as tar:
        for name, text in files.items():
            raw = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Depth of the id-level path under each top folder, as in the real index ("task1/replays/<generator>/<route id>", "task1/rows/<file>").
ID_DEPTH = {"replays": 4}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake dataset repository: manifest/archives.json, LAYOUT.md and archives/*.tar; downloads are copies."""
    repo = tmp_path / "repo"
    (repo / "archives").mkdir(parents=True)
    (repo / "manifest").mkdir()
    (repo / "LAYOUT.md").write_text("layout\n")
    (repo / "manifest" / "task2_routes.txt").write_text("A__n001\nB__n002\n")
    contents = {"scenes.tar": {"scenes/A/geometry_gpu.json": "{}", "scenes/B/geometry_gpu.json": "{}"},
                "task1.tar": {"task1/rows/hy_worldplay.jsonl.gz": "x", "task1/replays/hy_worldplay/A__n001/render24/clean.mp4": "v",
                              "task1/replays/hy_worldplay/A__n001__lead/render24/clean.mp4": "v"},
                "task2.tar": {"task2/routes/A__n001/route.json": "{\"route_id\": \"A__n001\"}",
                              "task2/routes/B__n002/route.json": "{\"route_id\": \"B__n002\"}",
                              "task2/open_routes/A__c01_wide/route.json": "{\"route_id\": \"A__c01_wide\"}"},
                "task3.tar": {"task3/static_first_person/wbench_case1/case.json": "{}",
                              "task3/static_first_person/wbench_case10/case.json": "{}"},
                "val.tar": {"val/rows/hy_worldplay/rows.jsonl.gz": "x", "val/episodes/C__fp000/meta.json": "{}"},
                "train-rows.tar": {"train/rows/hy_worldplay/stream.json": "{}"},
                "train-episodes-00000-of-00002.tar": {"train/episodes/A__fp001/meta.json": "{}"},
                "train-episodes-00001-of-00002.tar": {"train/episodes/B__fp002/meta.json": "{}",
                                                      "train/episodes/B__fp002__nw0/meta.json": "{}"},
                "train-crops-00000-of-00002.tar": {"train/crops/A__fp001__c004/crop.json": "{}"},
                "train-crops-00001-of-00002.tar": {"train/crops/B__fp002__c010/crop.json": "{}"}}
    archives = []
    for name, files in contents.items():
        digest = make_tar(repo / "archives" / name, files)
        contains = sorted({"/".join(f.split("/")[:ID_DEPTH.get(f.split("/")[1], 3)]) for f in files})
        archives.append({"file": f"archives/{name}", "bytes": (repo / "archives" / name).stat().st_size, "sha256": digest,
                         "members": len(files), "extracted_bytes": sum(len(t) for t in files.values()), "contains": contains})
    (repo / "manifest" / "archives.json").write_text(json.dumps({"archives": archives}))
    (repo / "text_prompts").mkdir()
    (repo / "text_prompts" / "task2.json").write_text("{}")
    fetched = []

    def fake_snapshot(repo_id, repo_type=None, local_dir=None, allow_patterns=None):
        for f in repo.rglob("*"):
            rel = f.relative_to(repo).as_posix()
            if f.is_file() and not rel.startswith("archives/") and any(Path(rel).match(p) for p in allow_patterns):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / rel).write_bytes(f.read_bytes())
        return local_dir

    def fake_fetch(repo_id, filename, local_dir, repo_type="dataset"):
        fetched.append(Path(filename).name)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / filename).read_bytes())
        return target

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(download, "fetch", fake_fetch)
    return {"repo": repo, "fetched": fetched, "data": tmp_path / "data"}


def test_index_contains_real_paths(hub):
    index = json.loads((hub["repo"] / "manifest/archives.json").read_text())
    by = {Path(a["file"]).name: a["contains"] for a in index["archives"]}
    assert "task1/replays/hy_worldplay/A__n001__lead" in by["task1.tar"] and "task2/open_routes/A__c01_wide" in by["task2.tar"]


def test_task2_takes_scenes_and_task2(hub):
    done = download.download_data(["task2"], local_dir=hub["data"])
    assert sorted(done) == ["scenes.tar", "task2.tar"]
    assert (hub["data"] / "task2/routes/B__n002/route.json").exists() and (hub["data"] / "scenes/A/geometry_gpu.json").exists()
    assert not list((hub["data"] / "archives").glob("*.tar"))
    assert (hub["data"] / "manifest/task2_routes.txt").exists() and (hub["data"] / "text_prompts/task2.json").exists()


def test_open_routes_come_from_task2(hub):
    assert sorted(download.download_data(["task2_open"], local_dir=hub["data"])) == ["scenes.tar", "task2.tar"]
    assert (hub["data"] / "task2/open_routes/A__c01_wide/route.json").exists()


def test_task1_and_task3(hub):
    assert sorted(download.download_data(["task1"], local_dir=hub["data"])) == ["task1.tar"]
    assert sorted(download.download_data(["task3"], local_dir=hub["data"])) == ["task3.tar"]


def test_train_and_val(hub):
    assert sorted(download.download_data(["train"], local_dir=hub["data"])) == [
        "scenes.tar", "train-crops-00000-of-00002.tar", "train-crops-00001-of-00002.tar", "train-episodes-00000-of-00002.tar",
        "train-episodes-00001-of-00002.tar", "train-rows.tar"]
    assert sorted(download.download_data(["val"], local_dir=hub["data"])) == ["scenes.tar", "val.tar"]
    assert download.download_data(["val"], local_dir=hub["data"])["scenes.tar"] == "already extracted"


def test_cases_pick_their_archives(hub):
    index = json.loads((hub["repo"] / "manifest/archives.json").read_text())
    names = lambda parts, cases: sorted(Path(a["file"]).name for a in download.select_archives(index, parts, cases))
    assert names(["train"], ["B__fp002"]) == ["scenes.tar", "train-crops-00001-of-00002.tar", "train-episodes-00001-of-00002.tar",
                                              "train-rows.tar"]
    assert names(["task2"], ["B__n002"]) == ["scenes.tar", "task2.tar"]
    assert names(["task1"], ["A__n001"]) == ["task1.tar"]
    assert names(None, ["wbench_case1"]) == ["scenes.tar", "task3.tar"]
    assert names(["task3"], ["wbench_case1"]) == ["task3.tar"]
    with pytest.raises(ValueError):
        download.select_archives(index, ["task3"], ["wbench_case2"])
    with pytest.raises(ValueError):
        download.download_data(["task2"], cases=["Nowhere__n000"], local_dir=hub["data"])


def test_rerun_skips_and_keep_archives(hub):
    download.download_data(["task3"], local_dir=hub["data"], keep_archives=True)
    assert (hub["data"] / "archives/task3.tar").exists()
    assert download.download_data(["task3"], local_dir=hub["data"]) == {"task3.tar": "already extracted"}
    assert hub["fetched"].count("task3.tar") == 1


def test_bad_checksum_is_refused(hub):
    index = json.loads((hub["repo"] / "manifest/archives.json").read_text())
    for a in index["archives"]:
        if a["file"] == "archives/task3.tar":
            a["sha256"] = "0" * 64
    (hub["repo"] / "manifest/archives.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="sha256"):
        download.download_data(["task3"], local_dir=hub["data"])
    assert not (hub["data"] / "task3").exists()


def test_unsafe_member_is_refused(tmp_path):
    bad = tmp_path / "bad.tar"
    make_tar(bad, {"../outside.txt": "x"})
    with pytest.raises(ValueError, match="outside"):
        download.safe_extract(bad, tmp_path / "data")


def test_cli(hub, monkeypatch):
    calls = []
    monkeypatch.setattr(download, "download_weights", lambda generators=None: calls.append(generators) or "weights")
    cli.download_main(["--what", "all", "--split", "train", "--cases", "A__fp001", "--generators", "hy_worldplay", "--data-root", str(hub["data"])])
    assert (hub["data"] / "train/episodes/A__fp001/meta.json").exists() and not (hub["data"] / "train/episodes/B__fp002").exists()
    assert (hub["data"] / "train/rows/hy_worldplay/stream.json").exists()
    assert calls == [["hy_worldplay"]]
