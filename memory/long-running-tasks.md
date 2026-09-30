# Long-running tasks: launch, watch, report

*keywords:* long run, background, monitor, watchdog, heartbeat, status report, timestamp, stale, crash, lingering process, resume, relaunch, progress.txt, hourly check, what is the progress

A training run here takes 4-20 hours, and this machine interrupts them: the GPU driver bugchecks under load, a finished process can
hang instead of exiting, and a careless second job exhausts the host's memory.
This is how a long task is launched, watched and reported, so the owner never has to ask "what's the progress?".

Every incident behind a rule below is in [`gotchas.md`](gotchas.md); the cost is in the "because" clause.

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
| **event** | waits on `progress.txt` for `END`, `RETRY`, `CHAIN STOPPED`, `DONE` | a step finished, crashed, retried, or the chain ended |
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
| process alive, log unchanged >15 min | a finished ROCm process hanging at exit | kill it, so the chain moves on; 7.5 h was lost to one, and a watchdog with a wrong regex nearly lost another |
| process gone, no results file, no guard message | crash, kill, or a driver bugcheck | relaunch with `--resume`; it continues from the last evaluation |
| a guard message in the log (`not finite at chunk`, `candidates' val AP moved`, `the training loss is`) | the run is **wrong**, not interrupted | stop, report immediately, do not retry |
| free memory under ~2 GB | something else is competing | stop the intruder before the run dies |
| a new bugcheck in the System log | the driver crashed the machine | note it with the dump path, then relaunch |

**Report bad news at once**, never at the next heartbeat.

## Reporting

- **Silence while healthy.** No message for a check that finds nothing.
- **One paragraph every 2-3 hours**: the current time, which step is running, how far in (epoch or stage), elapsed against
  expected, the last completed step, and when the resume file was last written. That last number is what says how much a crash
  would now cost.
- **The `Running` line of every answer** already carries the same duty (`../AGENTS.md`): one line, every job in flight, with its
  progress.

## After a session restart

**Check `progress.txt` and the running processes before anything else.** Watchers do not survive a session, and two nights were
lost this way: one to a run that was scheduled and never launched, one to a chain that had stopped hours earlier unnoticed.
A run that is mid-flight needs nothing; a stopped one needs a relaunch with `--resume`.

## What this does not cover

A **reboot** kills the launcher as well as the run, so the queue needs a human or a new session to restart it; the runs themselves
resume from their own state. A per-user logon task would close that gap and has not been set up.
