"""
This file is totally based on research papaer about DROW.
Here the link to the paper can be found:
https://arxiv.org/abs/1603.02636
The utility functions present here can be found in paper's GitHub:
https://github.com/VisualComputingInstitute/DROW/blob/master/v2/utils/__init__.py
"""

from json import loads
from pathlib import Path
from typing import Union, List, Dict, Mapping, Optional, Sequence, Tuple

from numpy import genfromtxt, fromregex, where, array, array_equal, vectorize, float32, uint32, arange, clip
from numpy.typing import NDArray

from ..utils.file_utils import DROW_DATA_PATH, DROW_TEST_SET, DROW_TRAIN_SET, DROW_VALIDATION_SET
from ..utils.drow_utils import laser_measures
from ..utils.generic_utils import Logging
from .drow_partitions import DROW_PARTITIONS

_DATASET_PATH = Path(__file__).parent.parent / DROW_DATA_PATH

#: Which directory each split name lives in, so a caller can ask for a split by the name the trainer uses for
#: FROG's rather than by a path. These three directories are the `official` partition; the others are tables of
#: recordings drawn from all three (`drow_partitions.py`).
DROW_SPLITS = {"train": DROW_TRAIN_SET, "val": DROW_VALIDATION_SET, "test": DROW_TEST_SET}


def partition_files(datapath: Union[Path, str], split: str, mode: str = "official",
                    partitions: Mapping[str, Mapping[str, Sequence[str]]] = DROW_PARTITIONS) -> List[str]:
    """The recordings (`<dir>/<stem>`, no extension) `split` holds under partition `mode`: the split's own directory
    for `official`, otherwise the table's recordings, in its order, from whichever of the three directories holds each."""
    if split not in DROW_SPLITS:
        raise ValueError(f"unknown DROW split {split!r}; expected one of {', '.join(DROW_SPLITS)}")
    if mode == "official":
        return [f"{f.parent}/{f.stem}" for f in (Path(datapath) / DROW_SPLITS[split]).glob("*.csv")]
    if mode not in partitions:
        raise ValueError(f"unknown DROW partition {mode!r}; expected official or one of {', '.join(partitions)}")
    located = {f.stem: f"{f.parent}/{f.stem}" for d in DROW_SPLITS.values() for f in (Path(datapath) / d).glob("*.csv")}
    missing = [stem for stem in partitions[mode][split] if stem not in located]
    if missing:
        raise FileNotFoundError(f"DROW partition {mode!r} names recordings not on disk: {', '.join(missing)}")
    return [located[stem] for stem in partitions[mode][split]]


class DROW_Dataset(Logging):
    TIME_FRAME = 5
    _LOAD_JSON_VECTOR = vectorize(loads, otypes=[object])

    def __init__(self, datapath: Union[Path, str] = _DATASET_PATH, dataset: Union[Path, str] = DROW_TEST_SET, laser_scans: int = laser_measures, time_frame_size: int = TIME_FRAME, verbose: bool = True, split: Optional[str] = None,
                 mode: str = "official"):
        """`split` names one of `DROW_SPLITS` and overrides `dataset`, so the same call shape serves this loader and
        FROG's; without it the test set loads, which is what every caller predating the argument asked for. `mode`
        is the partition the split is taken from (`partition_files`), and needs `split`."""
        Logging.__init__(self, verbose)
        self.time_frame = time_frame_size
        if split is not None:
            self.filenames = partition_files(datapath, split, mode)
            dataset = DROW_SPLITS[split] if mode == "official" else f"{mode} {split}"
        elif mode != "official":
            raise ValueError("a DROW partition other than official needs a split")
        else:
            self.filenames = [f"{f.parent}/{f.stem}" for f in (Path(datapath) / Path(dataset)).glob("*.csv")]

        self.scan_id: NDArray[uint32]
        self.scan_time: NDArray[float32]
        self.scans: NDArray[float32]

        scan_data = array([self._load_scan(f"{f}.csv", laser_scans) for f in self.filenames], dtype=object)
        self.scan_id, self.scan_time, self.scans = scan_data.transpose()
        self._print(f"Scans from {dataset}/*.csv loaded!")

        self.det_id: NDArray[uint32]
        self.det_wc: NDArray[float32]
        self.det_wa: NDArray[float32]
        self.det_wp: NDArray[float32]

        wc_id, self.det_wc = array([self._load_det(f"{f}.wc") for f in self.filenames], dtype=object).transpose()
        wa_id, self.det_wa = array([self._load_det(f"{f}.wa") for f in self.filenames], dtype=object).transpose()
        wp_id, self.det_wp = array([self._load_det(f"{f}.wp") for f in self.filenames], dtype=object).transpose()
        assert all(array_equal(a, b) and array_equal(a, c) for a, b, c in zip(wc_id, wa_id, wp_id)), "Dataset corrupt!"
        self.det_id = wc_id
        self._print(f"Detections from {dataset}/*.[wc|wa|wp] loaded!")

        self.odoms: NDArray
        self.idet2iscan: List[Dict[int, int]]

        self.odoms = array([self._load_odom(f"{f}.odom2") for f in self.filenames], dtype=object)
        self.idet2iscan = [{i: where(sid == d)[0][0] for i, d in enumerate(did)} for sid, did in zip(self.scan_id, self.det_id)]
        self._print(f"Detections from {dataset}/*.odom2 loaded!")

    @staticmethod
    def _load_scan(fname: Union[Path, str], laser_scans: int) -> NDArray:
        return genfromtxt(fname, delimiter=",", dtype=[("id", uint32), ("time", float32), ("scan", float32, laser_scans)], unpack=True)

    @classmethod
    def _load_det(cls, fname: Union[Path, str]) -> Tuple[NDArray, NDArray]:
        data = fromregex(fname, r"(\d+),([^\n]+)", dtype=[("id", uint32), ("json", object)])
        return data["id"], cls._LOAD_JSON_VECTOR(data["json"])

    @staticmethod
    def _load_odom(fname: Union[Path, str]) -> NDArray:
        return genfromtxt(fname, delimiter=",", dtype=[("eq", uint32), ("t", float32), ("xya", float32, 3)])

    def get_scan(self, sequence_id: int, scan_id: int, time_window: int, dtime: int = 1) -> Tuple[NDArray, NDArray]:
        """
        Return a (scans, odoms) window of `time_window` frames ending at
        scan_id, spaced `dtime` raw scans apart (dtime=1: consecutive frames,
        the original behaviour — e.g. a 0.5s window at DROW's ~10Hz rate).

        dtime > 1 spreads the same number of frames over more real time —
        e.g. dtime=40 on FROG's 40Hz data spans ~4s instead of ~125ms. This
        mirrors the official DROW-v2 reference loader's own `dtime` stride
        parameter (this port originally dropped it in favour of a fixed
        contiguous window).

        Indices before the start of the sequence are clamped to 0, which
        repeats the earliest available scan/odom — same effect the old
        pad-with-first-scan logic had, generalised to a strided window via
        fancy indexing.
        """
        idx = clip(scan_id - dtime * arange(time_window - 1, -1, -1), 0, scan_id)
        return self.scans[sequence_id][idx], self.odoms[sequence_id][idx]
