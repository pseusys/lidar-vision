"""What the DROW queue runs, pinned so a rerun reproduces these numbers rather than a near-miss of them.

The `balanced` run is the reason this file exists. It trained stage 3 with `--chunk-s 1 10`, a curriculum,
where the adopted recipe is a flat 10 s with none -- a departure nobody noticed until the checkpoints were
read back, by which time the figures had been written up as the recipe's. The flags a result was produced
under are part of the result, so they are asserted here rather than remembered.

Two of them are the owner's decisions of 2026-11-04 and would silently invert what the number means: the
range limit, because at FROG's default 10 m the run cannot represent 8% of DROW's labels at all, and the
dataset itself, because the flag defaults to `frog`.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import queue_drow  # noqa: E402


def flags(name):
    job = next(j for j in queue_drow.JOBS if j.name == name)
    return job, list(job.cli)


class TestEveryJobReadsDrowAtItsOwnSensorLimit:
    @pytest.mark.parametrize("name", ["stage2", "capacity_small", "capacity_dropout", "stage3"])
    def test_the_dataset_is_named_because_the_flag_defaults_to_frog(self, name):
        cli = flags(name)[1]
        assert cli[cli.index("--dataset") + 1] == "drow"

    @pytest.mark.parametrize("name", ["stage2", "capacity_small", "capacity_dropout"])
    def test_the_range_limit_is_drows_and_not_frogs_default(self, name):
        """At 10 m, `sanitize_ranges` flattens every beam carrying a label past it: 8% of DROW's test labels."""
        cli = flags(name)[1]
        assert cli[cli.index("--max-range-m") + 1] == "29.99"

    def test_stage_three_does_not_pass_a_range_limit_because_it_has_no_such_flag(self):
        """It is a `step2` flag, and deliberately: the limit shapes the network, so `network_shape` restores it
        from the checkpoint `--stage2` names. Passing it here is what stopped the chain on 2026-11-05."""
        assert "--max-range-m" not in flags("stage3")[1]

    def test_no_job_decimates_the_beams(self):
        """DROW is already at 0.5 deg; the decimation belongs to the FROG chain that has to meet it."""
        for job in queue_drow.JOBS:
            assert "--beam-stride" not in job.cli, job.name


class TestTheHeadlineArmIsTheAdoptedRecipe:
    def test_stage_three_takes_one_flat_chunk_length_and_no_curriculum(self):
        cli = flags("stage3")[1]
        at = cli.index("--chunk-s")
        assert cli[at + 1] == "10" and not cli[at + 2].replace(".", "").isdigit(), cli[at:at + 3]

    def test_stage_two_carries_the_recipes_own_width_rather_than_an_arms(self):
        """The claim under test is the published parameter table carried over untouched, so it is not tuned here."""
        assert "--channels" not in flags("stage2")[1]
        assert "--dropout" not in flags("stage2")[1]

    def test_the_memory_is_the_frozen_one(self):
        cli = flags("stage3")[1]
        assert {"--no-exchange", "--no-learned-match", "--no-coast"} <= set(cli)

    def test_stage_three_reads_the_averaged_stage_two_of_the_headline_arm(self):
        job, cli = flags("stage3")
        assert job.needs == ("stage2_avg",)
        assert cli[cli.index("--stage2") + 1].endswith("step2_calibration.averaged.pth")
        assert "drow_stage2_s0" in cli[cli.index("--stage2") + 1]


class TestTheCheapArmsRunBeforeTheExpensiveOne:
    """The ranking's one substantive change: these are a pre-flight check on stage 2, not a follow-up to stage 3."""

    def test_both_capacity_arms_are_listed_before_stage_three(self):
        order = [job.name for job in queue_drow.JOBS]
        assert order.index("capacity_small") < order.index("stage3")
        assert order.index("capacity_dropout") < order.index("stage3")

    def test_neither_arm_waits_on_anything(self):
        for name in ("capacity_small", "capacity_dropout"):
            assert flags(name)[0].needs == ()

    def test_the_arms_differ_from_the_headline_in_exactly_one_setting_each(self):
        base = set(flags("stage2")[1])
        for name, flag in (("capacity_small", "--channels"), ("capacity_dropout", "--dropout")):
            extra = set(flags(name)[1]) - base
            assert flag in extra, (name, sorted(extra))

    def test_each_arm_writes_to_its_own_directory(self):
        out = [job.cli[job.cli.index("--out-dir") + 1] for job in queue_drow.JOBS if "--out-dir" in job.cli]
        assert len(out) == len(set(out)), out


class TestEveryJobIsACommandTheParserAccepts:
    """The check the string assertions could not make, and the reason this class exists.

    Every test above matches flags as text, so a flag the subcommand does not have reads as present and
    correct. On 2026-11-05 that let `--max-range-m 29.99` reach `step3a`, which has no such option: the chain
    ran its gate, its stage 2 and both capacity arms, then burned three attempts in nine seconds on
    `unrecognized arguments` and stopped. Parsing is the only assertion that catches it, and it costs nothing.
    """

    def _parse(self, cli):
        """`cli` as `train_takheliped.py` itself would parse it, or `SystemExit` the way argparse raises it."""
        from train_takheliped import build_parser
        return build_parser().parse_args(list(cli))

    @pytest.mark.parametrize("name", ["stage2", "capacity_small", "capacity_dropout", "stage3"])
    def test_the_trainer_accepts_every_flag_the_job_passes(self, name):
        job, cli = flags(name)
        assert cli[0] == "train_takheliped.py", cli[0]
        self._parse(cli[1:])

    def test_an_unknown_flag_is_what_this_catches(self):
        """The guard fails when it should: the exact argv that stopped the chain."""
        with pytest.raises(SystemExit):
            self._parse(["step3a", "--stage2", "x.pth", "--max-range-m", "29.99"])

    def test_stage_three_still_reads_drows_range_limit_from_its_stage_two(self):
        """Dropping the flag changes nothing about the run: `network_shape` carries `max_range_m` over with the
        weights, which is why the flag was never owed. Pinned so nobody re-adds it to fix a phantom problem."""
        from train_takheliped import network_shape
        assert network_shape({"max_range_m": "29.99"})["max_range_m"] == 29.99


class TestEveryJobDeclaresWhatItWritesLast:
    """`run_queue` kills a job once its declared outputs exist, so an early file is a truncated run."""

    @pytest.mark.parametrize("name,written", [("stage2", "step2_results.json"),
                                              ("capacity_small", "step2_results.json"),
                                              ("capacity_dropout", "step2_results.json"),
                                              ("stage2_avg", "step2_calibration.averaged.pth"),
                                              ("stage3", "step3a_results.json")])
    def test_the_last_file_is_the_declared_one(self, name, written):
        assert [path.name for path in flags(name)[0].outputs] == [written]

    def test_the_gate_is_the_test_suite(self):
        assert "pytest" in " ".join(queue_drow.GATE.cli)
