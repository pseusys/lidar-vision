"""The candidate cache is written and read as a stream, and its bytes have not changed.

DROW is what made this a correctness question rather than a tidiness one. The cache holds one entry per frame
of a whole split -- not per annotated frame -- so DROW's train split is 341,147 of them at about 5 KB each,
roughly 1.7 GB. `pickle.dumps` builds the entire byte string beside the object graph, so writing it needed
about 3.4 GB on top of the 614 MB of raw scans and torch's own allocations, and on 2026-11-05 it raised
`MemoryError` after 56 minutes of decoding. `pickle.loads(path.read_bytes())` had the same defect on the way
back in, where it would have hit the same wall on the next run even if the write had squeezed through.

Two things are pinned here. That the round trip preserves every field, dtype included, since the memory reads
`float16` features and `bool` masks and a silent widening would cost accuracy rather than raise. And that the
**bytes are unchanged**, because the alternative -- a format only the new code can read -- would silently
invalidate every FROG cache on disk, each of which costs an hour to rebuild.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train_takheliped as tak  # noqa: E402


def one_sequence(frames=3, top_p=4, features=2, sectors=5, people=2):
    """A `SequenceCandidates` with every field distinct and every dtype the real one uses."""
    return tak.SequenceCandidates(
        xy=np.arange(frames * top_p * 2, dtype=np.float32).reshape(frames, top_p, 2),
        score=np.linspace(0, 1, frames * top_p, dtype=np.float32).reshape(frames, top_p),
        valid=np.ones((frames, top_p), bool),
        target=np.zeros((frames, top_p), bool),
        pose=np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        time=np.arange(frames, dtype=np.float64) * 0.0787,
        annotated=np.array([True, False, True][:frames]),
        gt_xy=np.arange(frames * people * 2, dtype=np.float32).reshape(frames, people, 2),
        gt_valid=np.ones((frames, people), bool),
        gt_covered=np.zeros((frames, people), bool),
        features=np.arange(frames * top_p * features, dtype=np.float16).reshape(frames, top_p, features),
        free_range=np.full((frames, sectors), 29.99, np.float16),
    )


def same(a, b):
    assert [f.name for f in tak.fields(tak.SequenceCandidates)], "the dataclass has fields"
    for f in tak.fields(tak.SequenceCandidates):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype, (f.name, x.dtype, y.dtype)
        assert np.array_equal(x, y), f.name


class TestTheRoundTripPreservesTheCache:
    def test_every_field_and_dtype_survives(self, tmp_path):
        """A widened `float16` or a `bool` turned into `int8` would change what the memory reads, silently."""
        seqs = [one_sequence(), one_sequence(frames=2, top_p=3)]
        path = tmp_path / "cache.pkl"
        tak.write_candidate_cache(path, seqs)
        back = tak.read_candidate_cache(path)
        assert len(back) == len(seqs)
        for a, b in zip(seqs, back):
            same(a, b)

    def test_the_payload_is_plain_dicts_rather_than_the_dataclass(self):
        """So a cache loads from any module, not only from this script run as `__main__`."""
        payload = tak.candidate_cache_payload([one_sequence()])
        assert isinstance(payload[0], dict)
        assert set(payload[0]) == {f.name for f in tak.fields(tak.SequenceCandidates)}


class TestTheBytesHaveNotChanged:
    """A format only the new code reads would invalidate every FROG cache on disk, at an hour each to rebuild."""

    def test_a_cache_written_the_old_way_still_loads(self, tmp_path):
        seqs = [one_sequence()]
        path = tmp_path / "old.pkl"
        path.write_bytes(pickle.dumps(tak.candidate_cache_payload(seqs)))     # exactly what the old writer did
        same(seqs[0], tak.read_candidate_cache(path)[0])

    def test_a_cache_written_now_is_byte_identical_to_the_old_writer(self, tmp_path):
        seqs = [one_sequence(), one_sequence(frames=1)]
        path = tmp_path / "new.pkl"
        tak.write_candidate_cache(path, seqs)
        assert path.read_bytes() == pickle.dumps(tak.candidate_cache_payload(seqs))


class TestNeitherEndHoldsTheWholeCacheTwice:
    """The contract that failed, asserted the only way it can be: the whole-buffer calls are never reached.

    `pickle.dumps` and `Path.read_bytes` are precisely the two calls that need a second full-size copy, so a
    writer that cannot reach them cannot need one. Peak memory itself is not assertable here -- a fixture small
    enough for a test suite has no peak -- but the call is.
    """

    def test_writing_does_not_build_the_whole_pickle_in_memory(self, tmp_path, monkeypatch):
        def refuse(*a, **k):
            raise AssertionError("pickle.dumps needs the whole cache as bytes beside the arrays; stream instead")

        monkeypatch.setattr(pickle, "dumps", refuse)
        path = tmp_path / "streamed.pkl"
        tak.write_candidate_cache(path, [one_sequence()])
        assert path.stat().st_size > 0

    def test_reading_does_not_slurp_the_whole_file(self, tmp_path, monkeypatch):
        path = tmp_path / "streamed.pkl"
        tak.write_candidate_cache(path, [one_sequence()])

        def refuse(self, *a, **k):
            raise AssertionError("read_bytes holds the whole cache as bytes beside the arrays; stream instead")

        monkeypatch.setattr(Path, "read_bytes", refuse)
        assert len(tak.read_candidate_cache(path)) == 1

    @pytest.mark.parametrize("name", ["write_candidate_cache", "read_candidate_cache"])
    def test_load_split_uses_these_helpers_rather_than_its_own_calls(self, name):
        """Otherwise the helpers are tested and the caller keeps the defect."""
        assert name in tak.load_split.__code__.co_names
class TestTheDatasetIsReleasedBeforePackDoublesTheArrays:
    """`pack` concatenates every field across recordings, so it needs a second full copy of the candidates --
    1.7 GB on DROW's train split. Whatever is still alive at that moment is added to the peak, and the raw
    dataset (614 MB of scans on DROW) plus the closure holding the network is 0.8 GB of it.

    The decode therefore lives in its own function, whose locals die when it returns. That is a structural
    property rather than an observable one, so it is asserted structurally: the names must not be `load_split`'s
    to hold.
    """

    @staticmethod
    def _names(fn):
        """Locals **and** cell variables. `ds` and `angles` are closed over by `candidates_of`, so the compiler
        puts them in `co_cellvars` rather than `co_varnames` -- checking only the latter asserts nothing."""
        code = fn.__code__
        return set(code.co_varnames) | set(code.co_cellvars)

    def test_load_split_does_not_keep_the_dataset_in_its_own_frame(self):
        held = self._names(tak.load_split)
        assert "ds" not in held, "the raw dataset stays alive through pack(); decode in a frame that is discarded"
        assert "net" not in held and "candidates_of" not in held, sorted(held)

    def test_the_decode_is_what_holds_them_instead(self):
        assert "decode_split_candidates" in tak.load_split.__code__.co_names
        held = self._names(tak.decode_split_candidates)
        for name in ("ds", "angles", "candidates_of"):
            assert name in held, (name, sorted(held))

    def test_pack_is_still_what_load_split_returns(self):
        """The extraction must not have moved the packing: a Split, not a list of recordings."""
        assert "pack" in tak.load_split.__code__.co_names
