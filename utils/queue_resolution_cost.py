#!/usr/bin/env python3
"""The cost half of the resolution answer, three averaging windows re-ranked, and DROW's first two-burst chunk.

**Why these jobs, in this order.** Halving FROG's angular resolution landed on 2026-11-06 with an accuracy answer
and no cost answer: the whole detector loses **0.88 pp** and the object memory loses nothing, but the only timing
on record is `step3a`'s own GPU figure, which reads 3.58 ms against the 0.25 deg seeds' 5.03 / 3.55 / 3.59 -- the
same number as two of the three, because the memory runs over 256 candidate slots and never sees a beam. **Where
decimation should pay is stage 2 on the CPU**, which is the device this project is judged on
(`AGENTS.md`), and nothing has measured it. Without that figure the resolution finding says a sensor at half
FROG's spacing costs a point and does not say what it buys.

**The timing runs first, and alone.** The protocol is an idle host, so the two latency jobs precede everything
that touches the GPU. They are back to back rather than interleaved -- `latency` times one checkpoint per
invocation -- which makes this a within-session comparison, the only kind this project trusts anyway: the absolute
`stage2` level has drifted between sessions for reasons still unexplained (`memory/performance-log.md` footnote
12), so the two readings are comparable to each other and not to a figure from another day.

**Then the three re-ranks, which are the cheap half of the averaging defect.** `average_checkpoints.py` ranked
every candidate window on FROG `official` val whatever the run trained on, so three runs chose their window on
the wrong data. `--rank-only` re-ranks them correctly and writes `averaging.json` **without** replacing the
averaged checkpoint, because `drow_stage2_s0`'s is named inside the DROW headline's trained stage 3 and
overwriting it would leave a model and its claimed weights disagreeing. What this answers is one question per
run: did the window move? It goes to `averaging.rank_only.json`, not over `averaging.json`, because that file is
the record of what each run's window *was* chosen on and for these three it is the evidence that found the defect.
If none moved, three recorded numbers are confirmed for two minutes of work each. If
the 0.5 deg one moved, the -0.88 pp above is loose by whatever the better window is worth, and re-averaging plus
a 5.9 h stage 3 becomes a decision rather than a guess.

**Then the one long arm, chosen by the owner on 2026-11-06: `--chunk-s 35` alone, not 35 and 65.** DROW annotates
in bursts of 20 at every 5th scan, 7.5 s long, with 24 s of nothing between them -- a burst period of 400 scans,
31.5 s. The adopted 10 s is 0.32 of a period, so every rollout sits inside **one** burst and sees about 6
annotated frames where FROG's 10 s sees 400. **35 s is the first length that spans a second burst** (1.11
periods), which is the whole point of the arm; 65 s only says whether the effect keeps growing, and is not worth
7 h until 35 s says there is an effect. The caveat the arm carries: the 24 s gap exceeds the slots' 10 s fade, so
a slot spawned in one burst is retired before the next begins, and a longer chunk hands stage 3 more supervised
frames but still no gradient crossing a gap. A null result therefore has two readings, and the fade arm is what
separates them.

**What is not here.** The multi-class head (item 84), which is a day of code before it is a run, and the DROW
stage 3 that waits for it -- this arm trains on today's binary stage 2 deliberately, because the mechanism it
measures is supervision density and does not depend on the head.

    python queue_resolution_cost.py --list
    python queue_resolution_cost.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from average_checkpoints import RANK_ONLY_NAME
from run_queue import CHECKPOINTS, REPO, Job, main

#: FROG at its native 0.25 deg: the stage 2 the frozen headline seed was trained on, and that seed's memory.
FULL_STAGE2 = CHECKPOINTS / "a71_soft_label_s0"
FULL_STAGE3 = CHECKPOINTS / "a71_freeze_s0"
#: The same chain at 0.5 deg, finished 2026-11-06.
HALF_STAGE2 = CHECKPOINTS / "frog_half_stage2_s0"
HALF_STAGE3 = CHECKPOINTS / "frog_half_stage3_s0"
#: The DROW headline, whose stage 2 this queue re-ranks and whose averaged weights the chunk arm reads.
DROW_STAGE2 = CHECKPOINTS / "drow_stage2_s0"
#: `balanced`'s stage 2, the third run ranked on the wrong val split.
BALANCED_STAGE2 = CHECKPOINTS / "balanced_stage2_s0"
#: The chunk arm's own directory: an arm off the headline, not a replacement for it.
DROW_CHUNK35 = CHECKPOINTS / "drow_stage3_chunk35_s0"

AVERAGED = "step2_calibration.averaged.pth"
MEMORY = "step3a_object_memory.best.pth"

#: DROW, which every DROW job names because the flag defaults to `frog`.
DROW = ("--dataset", "drow")
#: DROW's 0.5 deg grid met by decimating FROG. Every stage's flag, because `dataset_spec` reads it from the
#: command line rather than from the checkpoint -- including, since 2026-11-06, the helpers that re-score.
HALF = ("--beam-stride", "2")
#: The CPU protocol: 2,000 consecutive scans, batch 1, four threads, on an idle host.
CPU = ("--device", "cpu", "--threads", "4", "--frames", "2000")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")
#: `latency` names its file after the device and thread count, so each arm needs its own directory.
LATENCY_FULL = REPO / "results" / "latency_resolution_full"
LATENCY_HALF = REPO / "results" / "latency_resolution_half"
LATENCY_FILE = "latency_cpu_t4.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # The idle-host jobs first. 0.25 deg is the reference, so it is timed first and the comparison is read
    # within this session only.
    Job(name="latency_full", outputs=(LATENCY_FULL / LATENCY_FILE,),
        cli=("tracking_baselines.py", "latency", *CPU,
             "--stage2", str(FULL_STAGE2 / AVERAGED), "--memory", str(FULL_STAGE3 / MEMORY),
             "--out", str(LATENCY_FULL))),
    # `*HALF` is what makes this a 0.5 deg timing rather than a 360-beam network fed 720 beams, which is what
    # this sweep would silently have measured before 2026-11-06 (`memory/gotchas.md`).
    Job(name="latency_half", outputs=(LATENCY_HALF / LATENCY_FILE,), needs=("latency_full",),
        cli=("tracking_baselines.py", "latency", *CPU, *HALF,
             "--stage2", str(HALF_STAGE2 / AVERAGED), "--memory", str(HALF_STAGE3 / MEMORY),
             "--out", str(LATENCY_HALF))),

    # The three re-ranks. `--rank-only` writes `averaging.json` and leaves the averaged checkpoint in place,
    # which is what lets an old run be re-ranked when a trained stage 3 already depends on its weights.
    Job(name="rerank_half", outputs=(HALF_STAGE2 / RANK_ONLY_NAME,), needs=("latency_half",),
        cli=("average_checkpoints.py", str(HALF_STAGE2), *HALF, "--rank-only")),
    Job(name="rerank_drow", outputs=(DROW_STAGE2 / RANK_ONLY_NAME,), needs=("rerank_half",),
        cli=("average_checkpoints.py", str(DROW_STAGE2), *DROW, "--rank-only")),
    Job(name="rerank_balanced", outputs=(BALANCED_STAGE2 / RANK_ONLY_NAME,), needs=("rerank_drow",),
        cli=("average_checkpoints.py", str(BALANCED_STAGE2), "--frog-mode", "balanced", "--rank-only")),

    # The long arm. No `--max-range-m`: it is `step2`'s flag alone and `network_shape` restores it from the
    # stage-2 checkpoint, so passing it here stopped a chain three times in nine seconds on 2026-11-05.
    Job(name="drow_chunk35", outputs=(DROW_CHUNK35 / "step3a_results.json",), needs=("rerank_balanced",),
        cli=("train_takheliped.py", "step3a", *DROW, "--stage2", str(DROW_STAGE2 / AVERAGED), *FROZEN,
             "--chunk-s", "35", "--seed", "0", "--out-dir", str(DROW_CHUNK35), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
