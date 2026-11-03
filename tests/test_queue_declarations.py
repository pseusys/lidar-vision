"""What a queue module may declare as a job's outputs.

`run_queue.spawn_process` starts a kill timer the moment `finished(job)` is true: once every declared output exists it waits
`EXIT_GRACE_SECONDS` and then kills the child. A job that declares a file it writes **early** is therefore a job the runner
will cut off mid-work.

The frozen-baseline chain's `track` declared only `track_rows.json`, which the coordinate descent writes before any of the eight replays.
Seed 0 finished its replays 606 s after that file appeared and survived by six seconds; seeds 1 and 2 were killed, each
losing `bytetrack_cbmot`'s test pickle. `run_report` then dropped the method without a word -- its
`methods = {m: p for m, p in methods.items() if p["test"].exists()}` -- and scored three classical pipelines instead of four,
so `best_classical` fell from `bytetrack_cbmot` to `sort_m1` and the memory's margin read +3.56 pp instead of +1.98 pp.

The rule pinned here: **a job declares the files it writes last, not the first file it happens to produce.**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import queue_frozen_tracking_baselines
from run_queue import Job, finished

TRACK_METHODS = ("sort_m1", "sort_m3", "bytetrack", "bytetrack_cbmot")


def relocate(job: Job, root: Path) -> Job:
    """The same job with its outputs under `root`, so `finished` can be exercised without the real checkpoint tree."""
    return Job(name=job.name, cli=job.cli, cwd=job.cwd, needs=job.needs,
               outputs=tuple(root / path.name for path in job.outputs))


class TestTrackDeclaresWhatItWritesLast:
    """The replays are `track`'s product; `track_rows.json` and `track_tuning.json` are intermediates."""

    def test_every_method_s_test_pickle_is_declared(self):
        declared = {path.name for path in queue_frozen_tracking_baselines.track(0).outputs}
        assert declared >= {f"track_{method}_test.pkl" for method in TRACK_METHODS}

    def test_every_method_s_val_pickle_is_declared(self):
        declared = {path.name for path in queue_frozen_tracking_baselines.track(0).outputs}
        assert declared >= {f"track_{method}_val.pkl" for method in TRACK_METHODS}

    def test_the_tuning_file_alone_does_not_make_the_job_look_finished(self, tmp_path):
        """The regression itself: this is what started the 600 s kill timer ten minutes before the replays ended."""
        job = relocate(queue_frozen_tracking_baselines.track(0), tmp_path)
        (tmp_path / "track_rows.json").write_text("{}")
        assert not finished(job)

    def test_the_job_looks_finished_once_every_replay_is_on_disk(self, tmp_path):
        job = relocate(queue_frozen_tracking_baselines.track(0), tmp_path)
        for path in job.outputs:
            path.write_text("")
        assert finished(job)


class TestTheChainIsJudgedByItsFiles:
    def test_no_job_in_the_chain_is_judged_by_its_exit_code(self):
        """Both `track` and `arms` exit 1 on this host's ROCm teardown after finishing, so exit codes cannot judge them."""
        assert all(job.outputs for job in queue_frozen_tracking_baselines.JOBS)
