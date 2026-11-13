"""
Unpack JRDB's 2019 train archives into the directory the project reads JRDB from.

JRDB needs a free registration and is never downloaded automatically, so the five archives below are fetched by
hand from the 2019 section of the download page; which ones, and why only those, is `memory/data-model.md`,
"What to download". They land in DR-SPAAM's layout, `<dest>/{rosbags,timestamps,labels,detections,calibration}/`,
so DR-SPAAM's released JRDB code can be pointed at the parent directory as it is.

Run from the repository root:

    .venv/Scripts/python.exe utils/unpack_jrdb.py                          # archives in ~/Downloads
    .venv/Scripts/python.exe utils/unpack_jrdb.py --source D:/jrdb_zips

A rerun is safe and cheap. Every file is written to `<name>.part` and renamed only once the archive's CRC has been
checked and the size matches, so a file under its final name is complete, and one already there at its archive size
is skipped: an interrupted run resumes where it stopped instead of rewriting 45 GB of rosbags. The archives are only
read, never moved or deleted.
"""

import argparse
import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

from takheliped.utils.file_utils import JRDB_DATA_PATH

ARCHIVES = ("train_rosbags.zip", "train_timestamps.zip", "train_labels.zip", "train_detections.zip", "train_calibration.zip")
DEFAULT_SOURCE = Path.home() / "Downloads"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "library" / "takheliped" / JRDB_DATA_PATH / "train_dataset"

GB = 1024 ** 3
ANNOUNCE_BYTES = 100 * 1024 ** 2   # files at least this big get a line each, so a long run shows it is moving
COPY_CHUNK = 16 * 1024 ** 2


def unpack_archive(archive: Path, dest: Path) -> Tuple[int, int]:
    """Extract `archive` under `dest`; return how many files were written and how many were already complete.

    Every member is checked to land inside `dest` before the first one is written, so a hostile or malformed archive
    writes nothing at all."""
    root = dest.resolve()
    written = skipped = 0
    with zipfile.ZipFile(archive) as opened:
        members = opened.infolist()
        for info in members:
            if not (root / info.filename).resolve().is_relative_to(root):
                raise ValueError(f"{archive.name}: member {info.filename!r} would land outside {root}")
        for info in members:
            target = root / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with opened.open(info) as source, part.open("wb") as sink:   # reading to the end is what checks the CRC
                shutil.copyfileobj(source, sink, COPY_CHUNK)
            if part.stat().st_size != info.file_size:
                raise OSError(f"{info.filename}: wrote {part.stat().st_size} bytes, the archive says {info.file_size}")
            part.replace(target)
            written += 1
            if info.file_size >= ANNOUNCE_BYTES:
                print(f"  {info.filename}: {info.file_size / GB:.2f} GB", flush=True)
    return written, skipped


def unpack(source: Path, dest: Path) -> Tuple[int, int]:
    """Extract every archive in `ARCHIVES` from `source` under `dest`; return the files written and skipped in total.

    All five must be present before anything is written, so a missing download is reported up front rather than
    after the rosbags have taken their time."""
    missing = [name for name in ARCHIVES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"not in {source}: {', '.join(missing)}")
    written = skipped = 0
    for name in ARCHIVES:
        start = time.monotonic()
        archive_written, archive_skipped = unpack_archive(source / name, dest)
        print(f"{name}: {archive_written} written, {archive_skipped} already complete, "
              f"{time.monotonic() - start:.0f} s", flush=True)
        written += archive_written
        skipped += archive_skipped
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="directory holding the five archives (default: %(default)s)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where they are unpacked (default: %(default)s)")
    args = parser.parse_args()
    written, skipped = unpack(args.source, args.dest)
    files = [path for path in args.dest.rglob("*") if path.is_file()]
    print(f"done: {written} written, {skipped} already complete; {args.dest} holds {len(files)} files, "
          f"{sum(path.stat().st_size for path in files) / GB:.1f} GB")


if __name__ == "__main__":
    main()
