"""`utils/unpack_jrdb.py`: where JRDB's archives land, that a rerun resumes rather than rewrites, and what it refuses."""

import sys
import zipfile
from pathlib import Path
from typing import Dict

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from unpack_jrdb import ARCHIVES, unpack  # noqa: E402


def _write_archives(source: Path) -> Dict[str, bytes]:
    """One small archive per name in `ARCHIVES`, laid out like JRDB's own: a top-level folder holding everything.

    Returns every file member's archive path and bytes, which is what the destination has to hold afterwards."""
    source.mkdir(parents=True, exist_ok=True)
    files = {}
    for index, name in enumerate(ARCHIVES):
        folder = name[len("train_"):-len(".zip")]
        members = {f"{folder}/seq_0/frames.json": f'{{"archive": {index}}}'.encode(),
                   f"{folder}/seq_0.bag": bytes(range(256)) * (index + 3)}
        with zipfile.ZipFile(source / name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{folder}/", b"")
            archive.writestr(f"{folder}/seq_0/", b"")
            for path, data in members.items():
                archive.writestr(path, data)
        files.update(members)
    return files


def test_every_member_lands_under_dest_at_its_archive_path(tmp_path):
    files = _write_archives(tmp_path / "downloads")
    dest = tmp_path / "JRDB-data" / "train_dataset"

    written, skipped = unpack(tmp_path / "downloads", dest)

    assert (written, skipped) == (len(files), 0)
    for path, data in files.items():
        assert (dest / path).read_bytes() == data
    assert not list(dest.rglob("*.part")), "a finished run leaves no partial file behind"


def test_a_rerun_skips_every_finished_file(tmp_path):
    files = _write_archives(tmp_path / "downloads")
    dest = tmp_path / "train_dataset"
    unpack(tmp_path / "downloads", dest)

    assert unpack(tmp_path / "downloads", dest) == (0, len(files))


def test_a_rerun_rewrites_a_file_whose_size_is_wrong_and_replaces_a_stale_part(tmp_path):
    files = _write_archives(tmp_path / "downloads")
    dest = tmp_path / "train_dataset"
    unpack(tmp_path / "downloads", dest)

    truncated = dest / "rosbags" / "seq_0.bag"
    truncated.write_bytes(truncated.read_bytes()[:10])
    interrupted = dest / "labels" / "seq_0.bag"
    interrupted.unlink()
    (dest / "labels" / "seq_0.bag.part").write_bytes(b"half a file")

    assert unpack(tmp_path / "downloads", dest) == (2, len(files) - 2)
    assert truncated.read_bytes() == files["rosbags/seq_0.bag"]
    assert interrupted.read_bytes() == files["labels/seq_0.bag"]
    assert not list(dest.rglob("*.part"))


def test_a_member_that_would_land_outside_dest_is_refused_before_anything_is_written(tmp_path):
    _write_archives(tmp_path / "downloads")
    with zipfile.ZipFile(tmp_path / "downloads" / ARCHIVES[0], "a") as archive:
        archive.writestr("../escaped.txt", b"outside")
    dest = tmp_path / "out" / "train_dataset"

    with pytest.raises(ValueError, match="escaped.txt"):
        unpack(tmp_path / "downloads", dest)

    assert not (tmp_path / "out" / "escaped.txt").exists()
    assert not dest.exists() or not [p for p in dest.rglob("*") if p.is_file()]


def test_a_missing_archive_is_named_before_anything_is_written(tmp_path):
    _write_archives(tmp_path / "downloads")
    (tmp_path / "downloads" / "train_labels.zip").unlink()
    dest = tmp_path / "train_dataset"

    with pytest.raises(FileNotFoundError, match="train_labels.zip"):
        unpack(tmp_path / "downloads", dest)

    assert not dest.exists()
