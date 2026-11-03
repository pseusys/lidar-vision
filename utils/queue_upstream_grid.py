#!/usr/bin/env python3
"""`TODO.md` item 64: score Li2Former with their own suppression and their own metric, at both their radii.

**Why this is two jobs and not eight.** `evaluate.py --upstream-grid` runs inference once and scores that one
pass four ways -- our decoder and theirs, crossed with our wp-AUC and their `calPrecisionRecall`. A radius
therefore costs one CPU pass, about two hours on this host at the ~2.9 s/frame `drow_eval` measured, and the
"ours/ours" cell of each pass *is* the number that chain already reported: **48.1% at d=0.5, 40.5% at d=0.3**.
`tests/test_upstream_grid_wiring.py` pins that equality to 1e-9, which is the only reason the other three cells
can be read against it.

**What is being scored, and what is not.** Epoch 8's weights, the last ones before their regression objective's
escape basin took the vote head to zero (`train vote` 0.1529 -> 0.0732 -> 0.0000 over epochs 9-11). This is the
arm at its best, not a collapsed head. Their published 76.4% AP@0.5 / 72.5% AP@0.3 are *trained* numbers under
their own recipe, and no checkpoint of theirs was ever released -- so cell 4 answers "what does their protocol
say about these weights", never "did their paper reproduce".

**Their metric raises rather than returning zero when a cell has no true positives at all**, from
`np.where(precs != 0)[0][0]` inside their `eer()`. That is left as released, on the owner's instruction not to
fix their code; a cell dying there is itself the answer, since its AP would be 0.

    python queue_upstream_grid.py --list
    python queue_upstream_grid.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows. CPU only, so no ROCm
kernel compilation and no teardown hang: these exit normally. Both jobs **declare no outputs, because
`evaluate.py` prints**, which also means no kill timer is ever armed (item 56's defect).
"""

from run_queue import REPO, Job, main

# Epoch 8, 203,882,894 bytes, written 2026-11-02 11:36:56 -- 32.9% wp-AUC on full DROW val, 48.1% on test.
BEST = REPO / "checkpoints_drow" / "li2former_official_parallel_resumed.best.pth"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

# d=0.5, the radius their 76.4% is quoted at.
GRID_05 = Job(
    name="grid_r05",
    cli=("evaluate.py", "--dataset", "drow", "--split", "test", "--eval-r", "0.5",
         "--upstream-grid", "--no-bench", "--li2former-official", str(BEST)))

# d=0.3, their 72.5% column -- the one a mislocalised vote cannot hide in.
GRID_03 = Job(
    name="grid_r03",
    cli=("evaluate.py", "--dataset", "drow", "--split", "test", "--eval-r", "0.3",
         "--upstream-grid", "--no-bench", "--li2former-official", str(BEST)))

JOBS = [GRID_05, GRID_03]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
