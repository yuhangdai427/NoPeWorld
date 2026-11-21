"""Downloads NoPeBench (tar archives) and the NoPeWorld heads from the Hugging Face Hub.

The dataset repository holds plain files (README.md, LAYOUT.md, manifest/, text_prompts/), the archives `archives/*.tar` and their
index `manifest/archives.json`:

    {"archives": [{"file": "archives/task2.tar", "bytes": 7319982080, "sha256": "...", "members": 6017, "extracted_bytes": 7314791898,
                   "contains": ["task2/routes/<route_id>", "task2/open_routes/<route_id>", ...]}, ...]}

`contains` lists the id-level folders (or files) an archive holds. Each archive is checked against its sha256, extracted into the data
root and deleted unless kept; a marker in `<data root>/.archives/` lets a rerun skip it.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from nopeworld.paths import root

DATASET_REPO = "PeterDAI/NoPeBench"
MODEL_REPO = "PeterDAI/NoPeWorld"
ARCHIVE_INDEX = "manifest/archives.json"
LOOSE_FILES = ["README.md", "LAYOUT.md", "manifest/*", "text_prompts/*"]
MARKERS = ".archives"
# The archives each part needs (shell patterns on archive file names). Task 1 replays and labels are all in task1.tar; the Task 2
# evaluator reads the scene maps; the open routes are in task2.tar.
PARTS = {"task1": ["task1.tar"],
         "task2": ["scenes.tar", "task2.tar"],
         "task2_open": ["scenes.tar", "task2.tar"],
         "task3": ["task3.tar"],
         "train": ["scenes.tar", "train-rows.tar", "train-episodes-*.tar", "train-crops-*.tar"],
         "val": ["scenes.tar", "val.tar"]}
# With --cases: the archives every case of a part needs besides its own folders.
SHARED = {"task2": ["scenes.tar"], "task2_open": ["scenes.tar"], "train": ["scenes.tar", "train-rows.tar"], "val": ["scenes.tar"]}


def archive_name(entry):
    return PurePosixPath(entry["file"]).name


def held_ids(entry):
    """The ids an archive holds: the last component of each `contains` path."""
    return {PurePosixPath(p).name for p in entry.get("contains") or []}


def holds(entry, case):
    """True when the archive holds the id itself or a folder derived from it (`<id>__lead`, `<id>__nw0`, `<id>__c008`)."""
    return any(i == case or i.startswith(case + "__") for i in held_ids(entry))


def file_sha256(path, block=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def select_archives(index, parts=None, cases=None):
    """The archive entries for these parts (all archives without parts), narrowed to those holding `cases` when given."""
    archives = index["archives"]
    if parts:
        patterns = [p for part in parts for p in PARTS[part]]
        archives = [a for a in archives if any(fnmatch.fnmatch(archive_name(a), p) for p in patterns)]
    if cases:
        missing = [c for c in cases if not any(holds(a, c) for a in archives)]
        if missing:
            raise ValueError(f"no selected archive holds {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        shared = {n for part in parts for n in SHARED.get(part, [])} if parts else {"scenes.tar"}
        archives = [a for a in archives if any(holds(a, c) for c in cases) or archive_name(a) in shared]
    return archives


def safe_extract(tar_path, destination):
    """Extracts a tar archive into `destination`, refusing members that would land outside it or are links."""
    destination = Path(destination).resolve()
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"{tar_path}: member {member.name!r} lands outside the data root")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"{tar_path}: member {member.name!r} is a link or a device")
        tar.extractall(destination, members=members)


def no_brotli():
    """huggingface_hub 1.x downloads over httpx, whose brotli decoder fails with brotli / brotlicffi 1.2 ("decoder process called
    with data when 'can_accept_more_data()' is False"). The Hub's clients here ask for gzip or deflate only."""
    try:
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import default_client_factory
    except ImportError:
        return

    def factory():
        client = default_client_factory()
        client.headers["Accept-Encoding"] = "gzip, deflate"
        return client
    set_client_factory(factory)


def fetch(repo_id, filename, local_dir, repo_type="dataset"):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir=str(local_dir)))


def download_data(parts=None, cases=None, local_dir=None, keep_archives=False, repo_id=DATASET_REPO):
    """Downloads, checks and extracts the archives these parts / cases need into the data root; returns what was done per archive."""
    from huggingface_hub import snapshot_download
    no_brotli()
    data = Path(local_dir or root("data"))
    data.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(data), allow_patterns=LOOSE_FILES)
    index = json.loads((data / ARCHIVE_INDEX).read_text())
    markers = data / MARKERS
    markers.mkdir(exist_ok=True)
    done = {}
    for entry in select_archives(index, parts, cases):
        name = archive_name(entry)
        marker = markers / f"{name}.extracted"
        if marker.exists() and marker.read_text().strip() == entry["sha256"]:
            done[name] = "already extracted"
            continue
        tar_path = fetch(repo_id, entry["file"], data)
        digest = file_sha256(tar_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{name}: sha256 {digest} is not the manifest's {entry['sha256']}; delete {tar_path} and download again")
        safe_extract(tar_path, data)
        marker.write_text(entry["sha256"] + "\n")
        if not keep_archives:
            os.remove(tar_path)
        done[name] = "extracted"
    cache = data / ".cache"
    if not keep_archives and cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
    return done


def download_weights(generators=None, local_dir=None, repo_id=MODEL_REPO):
    """The heads (all, or those of some world models) into the weights root."""
    from huggingface_hub import snapshot_download
    no_brotli()
    patterns = None if not generators else ["README.md", "manifest.json"] + [f"heads/{g}/*" for g in generators]
    return snapshot_download(repo_id, local_dir=str(local_dir or root("weights")), allow_patterns=patterns)
