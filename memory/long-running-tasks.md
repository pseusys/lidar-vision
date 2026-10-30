# Long-running tasks: launch, watch, report

*keywords:* long run, background, monitor, watchdog, heartbeat, status report, timestamp, stale, crash, lingering process, resume, relaunch, progress.txt, hourly check, what is the progress

A training run here takes 4-20 hours, and this machine interrupts them: the GPU driver bugchecks under load, a finished process can
hang instead of exiting, and a careless second job exhausts the host's memory.
This is how a long task is launched, watched and reported, so the owner never has to ask "what's the progress?".

Every incident behind a rule below is in [`gotchas.md`](gotchas.md); the cost is in the "because" clause.

## Watchers expire long before a training run does

A background command started by the agent is killed at roughly **30 minutes**, whatever deadline the script itself carries. On
2026-10-01 a 3-hour reporting sleeper and a 10-hour event watcher were both killed 31 minutes after arming, with no output.
Earlier in the same week the identical sleeper survived its full 3 hours, so **the ceiling moves** -- never assume the one that
worked last session still will.

**Consequences, and the fix:**

- A watcher that outlives its limit is silent in exactly the way a healthy watcher is silent. Silence is not evidence the chain
  is alive; `progress.txt` with a recent timestamp is.
- **Use a cron job for the cadence, not a sleeping background task.** `CronCreate` fires on wall-clock time, independently of
  the background-task limit, and survives for the session. Half-hourly is a reasonable poll for a run that takes hours. Pick an
  off-minute (`23,53 * * * *`, not `0,30`).
- **Keep the chain detached** (below), so the watcher dying never takes the run with it. The watcher's job is to wake the agent;
  the chain's job is to finish regardless.
- Cron jobs are session-only too. Re-create them at the start of each session, after reading `progress.txt` to find out what
  actually happened while nobody was watching.

## Launch the chain detached, or it dies with the session

A launcher started through the agent's own background-command tool is a **child of the session process**. When that session
ends -- a crash, a teardown, a restart -- the whole tree goes with it, **including grandchildren started detached**, because
the tree is held in a job object rather than merely parented. On 2026-10-01 a
6.8-hour stage-3 run was killed this way three hours after its last epoch, with nobody watching: the chain simply stopped
existing, and the only trace was a `progress.txt` whose last line was hours stale.

**`Start-Process` is not enough.** Tried on 2026-10-01: the chain was started detached and still died with the session, because
Windows puts the agent's whole process tree in a **job object**, and a job object takes its children's children too. The run
reached the end of its 1 s stage and stopped at 18:11, the minute the session turned over.

**Use the Task Scheduler, which is a different service rather than a different process:**

```powershell
$past = (Get-Date).AddMinutes(-5).ToString("HH:mm")     # a start time already gone, so the trigger never fires by itself
schtasks /create /tn "lidar_<item>" /tr "`"$python`" `"$queue`"" /sc once /st $past /it /f
schtasks /run   /tn "lidar_<item>"
```

**The chain itself is `utils/run_queue.py`** since 2026-10-09 (A57), not a copied PowerShell script: `$queue` is a committed
Python module that builds `Job`s and calls `main`, and the runner owns retries, the skip-if-done rule, the guard classification
and the progress lines. It takes `--list`, `--steps` and `--progress`. Every launcher incident in `TODO.md` A57's table came
from the eight hand-copied copies, so write a queue module and do not copy a launcher again.

**It happened again on 2026-10-16, and this is what it looks like from the logs.** The step-1 chain was launched with
`Start-Process` from the session rather than through the Task Scheduler -- the mistake this section exists to prevent -- and
died with the session at 23:31 after about ten hours, three arms in. **The signature is unmistakable once you know it:**
every remaining job has `attempt1`, `attempt2` and `attempt3` logs, **all zero bytes, all stamped within the same second or
two**. The children could not spawn at all, so the runner burned each job's whole retry budget in milliseconds and recorded
nothing. A job that genuinely failed writes a log with a traceback in it and takes time doing so.

So: **zero-byte attempt logs with identical timestamps mean a teardown, not a failing job.** Do not go looking for a bug in
the arm; relaunch the chain, which skips what is already finished. The retry counters live in the runner's own process, so a
relaunch starts them fresh -- the burnt attempts cost nothing but a confusing log directory.

**Never give the one-off trigger a future time.** An earlier version of this template used `/st 23:59`, and on 2026-10-02 the
A56 tasks then fired a second time at 23:59, re-running two finished chains over their own outputs (their pickles survived, two
logs were truncated). `schtasks` warns that a past `/st` "may not run"; that is the point, since `/run` starts it now.
Delete the task once its chain is done.

`-WindowStyle Hidden` matters too: without it the chain's console sits on the desktop, and a keystroke into it or a close of it ends the chain with `0xC000013A` (`gotchas.md`, twice in 2026-10). `/it` matters: the task then runs **in the logged-on user's interactive session**, which is what gives it the GPU. A task
without it can land in session 0 and find no device. Verified 2026-10-02: the trainer came up under the scheduler reporting
`CUDA / ROCm: AMD Radeon RX 9060 XT (1 device(s))`.

With `--resume` on every step, a relaunch then costs the current stage rather than the whole run.

**Each attempt keeps its own log**, `<job>.<stamp>.attempt<n>.log` under `checkpoints_three_horizon/queue_logs/`, so a relaunch
no longer overwrites the earlier stage's epoch history the way the PowerShell `Run-Step` did. **Watchers are the
exception** and stay session-bound: their whole purpose is to notify the agent, so a watcher that outlives the session has
nobody to tell. Re-arm them at the start of each session instead, and read `progress.txt` first -- it is the one record that
survives everything.

**The resumed run is a different draw**, not the same run continued (`interpreting-evaluation.md`). That is fine for a
screening arm and wrong for a reproduction; say which one it is when reporting the result.

## Before launching

1. **Run the guard, do not eyeball it:** `python memory/scripts/machine_is_free.py && <your command>`. It exits non-zero when a
   project job is running or under 2 GB is free, and it exists because an agent killed a live training run with `pytest` **twice**
   -- on 2026-09-22 at 3 hours in, and again on 2026-09-28 at 1.2 hours in, the second time while the rule above was already written.
2. **A dead run is not the only cost:** the same exhaustion fails whatever you started, so a test result from a loaded machine
   means nothing either (2026-09-28: `test_both_clis_can_print_help` failed on a paging-file error, not on a code change).
3. **Launch through a launcher that retries with `--resume`**, detached, never as a child of a tool call: a tool-call timeout once
   killed the launcher's whole process tree, and an interactive session's end kills anything it owns.
4. **Say the time.** Print the current time with the launch line, in the report and in `checkpoints_three_horizon/progress.txt`, so
   a stale report is obvious at a glance.

## While it runs

**Arm a watcher next to the task, and let events wake you, not the clock.** Waiting on the task's own output (its log, or a new
line in `progress.txt`) catches a finish, a crash or a guard within a minute. A fixed hourly poll adds nothing on a run that
advances every 2-13 minutes, and each wake-up costs a turn.

**Arm two watchers, not one, and re-arm both after every wake-up** (owner's design, 2026-09-28):

| watcher | what it is | what it wakes you for |
| --- | --- | --- |
| **event** | polls `progress.txt` for `END`, `RETRY`, `CHAIN STOPPED`, `DONE` -- by `wc -l` and `sed -n`, **never `tail -F`**, which locks the chain out of its own progress file (`gotchas.md`) | a step finished, crashed, retried, or the chain ended |
| **clock** | `sleeper.sh <seconds>`: sleeps, prints the time, exits | the 2-3 hour status report falls due |

They must be **separate background tasks**, because a single task that exits on "either an event or a deadline" lets an event
reset the reporting clock -- which is exactly what happened on 2026-09-28, when a 20 hour chain produced no scheduled report at
all. Each completion is one wake-up: on an event, act on it and re-arm the event watcher; on the clock, write the paragraph and
re-arm the clock. Re-arm whichever fired, and check the other is still alive.

**The gap this leaves:** both watchers die with the session, and there is a short window between a wake-up and the re-arm. That
is what the restart check below is for.

**Every check runs this list.** Three of the five symptoms show a *live* process, so "is it alive?" is not a check:

| symptom | what it is | what to do |
| --- | --- | --- |
| process alive, log unchanged >15 min | a finished ROCm process hanging at exit, even past `os._exit` (`gotchas.md`) | kill it, so the chain moves on; 7.5 h was lost to one, and a watchdog with a wrong regex nearly lost another. Better: let the chain do it -- wait for the step's final log line, then stop the process two minutes later (A56's `StepUntil`) |
| process gone, no results file, no guard message | crash, kill, or a driver bugcheck | relaunch with `--resume`; it continues from the last evaluation |
| a guard message in the log (`not finite at chunk`, `candidates' val AP moved`, `the training loss is`) | the run is **wrong**, not interrupted | stop, report immediately, do not retry |
| free memory under ~2 GB | something else is competing | stop the intruder before the run dies |
| a new bugcheck in the System log | the driver crashed the machine | note it with the dump path, then relaunch |

**The >15 min symptom needs a phase check before you act on it, or it kills healthy runs** (2026-10-29, twice in one
day). `train_takheliped.py step3a` has two phases that do tens of minutes of work and print **only on completion**, so
the clock alone cannot tell them from a hang:

* **Test evaluation.** 50,088 frames at stride 1 and again at stride 5. `freeze_s0` sat silent for 24 min here.
* **Candidate-cache rebuild.** `candidate_cache_path` keys on the stage-2 checkpoint's write time, so a new stage-2
  seed rebuilds train and val caches first -- `freeze_s1` spent **39 min** on it (924 s + 1422 s) before epoch 1.

**The discriminator is the results file, not the clock.** The teardown hang happens *after* `step3a_results.json` is
written, so: no results file means work remains -- leave it alone. Results file present and the process still alive
means the hang -- kill it. Confirm with two CPU samples a few seconds apart if you want a second signal, but remember
CPU burn alone is not proof of progress (a crashed-but-hanging process also burns CPU); the file is the evidence.

**Report bad news at once**, never at the next heartbeat.

## Reporting

- **Silence while healthy.** No message for a check that finds nothing.
- **One paragraph every 2-3 hours**: the current time, which step is running, how far in (epoch or stage), elapsed against
  expected, the last completed step, and when the resume file was last written. That last number is what says how much a crash
  would now cost.
- **The `Running` line of every answer** already carries the same duty (`../AGENTS.md`): one line, every job in flight, with its
  progress.
- **Every observer report carries two clocks: `now` and `next`** (owner, 2026-10-27). `now` is the wall-clock time the check
  was taken; `next` is when the report itself says the next thing is due -- the next epoch line, the arm's earliest stop, the
  chain's expected finish. Give both as times, not as durations.

  *Why two and not one.* A single check can only say "the log is N minutes old", which is meaningless without knowing the
  cadence -- and the cadence changes per arm (582 s an epoch for `dim 64`, 1,888 s during one unexplained excursion, 25 minutes
  while a candidate cache rebuilds). A **reader scrolling the session history** sees a column of `now` stamps against a column
  of `next` predictions, and a hang shows up as the moment `now` runs past a `next` that never arrived. That is a property of
  the *sequence* of reports, which no single report can carry, and it is why the pair has to be in every one of them rather
  than only in the ones that look interesting.

  It also disciplines the writer: predicting `next` forces an explicit model of what the job should do, so a missed prediction
  is visible rather than absorbed. Three times in the 2026-10-26/27 sweep an estimate was wrong -- a "50-epoch ceiling" that
  was 100, a patience of 8 read as 12, a "twenty minute" resume that reset its own patience window and ran for hours -- and
  each would have been caught a check earlier by a `next` that did not come true.

## After a session restart

**Check `progress.txt` and the running processes before anything else.** Watchers do not survive a session, and two nights were
lost this way: one to a run that was scheduled and never launched, one to a chain that had stopped hours earlier unnoticed.
A run that is mid-flight needs nothing; a stopped one needs a relaunch with `--resume`.

## What this does not cover

A **reboot** kills the launcher as well as the run, so the queue needs a human or a new session to restart it; the runs themselves
resume from their own state. A per-user logon task would close that gap and has not been set up.
