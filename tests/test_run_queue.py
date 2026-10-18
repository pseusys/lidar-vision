"""The queue runner `utils/run_queue.py`, which replaces the hand-copied PowerShell launchers (`TODO.md` A57).

Every rule pinned here is one the duplicated scripts got wrong at least once, and the reasons are in `TODO.md` A57's table:
a step that trained ahead of a failed gate, a `Wait-Finished` that matched the wrong step's marker, a retry loop that re-ran a
run whose training guard had already fired. The process spawn is a seam, so none of this needs a GPU, a checkpoint or torch.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from run_queue import (ES_CONTINUOUS, ES_SYSTEM_REQUIRED, MAX_ATTEMPTS, Job, Outcome, guard_fired, main, report, run,
                       spawn_process, stay_awake, validate)

TIMESTAMP_LENGTH = len("2026-10-09T13:30:00")


@dataclass
class Reply:
    """What the fake machine does for one attempt: whether the job's outputs appear, what its log says, its exit code."""
    outputs: bool = True
    log: str = ""
    code: int = 0


class FakeMachine:
    """Stands in for the process spawn. Records every attempt and plays the reply scripted for that job."""

    def __init__(self, replies: dict[str, Reply] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[str] = []

    def __call__(self, job: Job, log: Path) -> int:
        self.calls.append(job.name)
        reply = self.replies.get(job.name, Reply())
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(reply.log, encoding="utf-8")
        if reply.outputs:
            for output in job.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("result", encoding="utf-8")
        return reply.code


def _job(name: str, tmp_path: Path, **kwargs) -> Job:
    return Job(name=name, cli=("train_takheliped.py", "step2", "--out-dir", name), outputs=(tmp_path / f"{name}.json",), **kwargs)


def _run(jobs, tmp_path, machine, **kwargs):
    return run(jobs, spawn=machine, progress=tmp_path / "progress.txt", log_root=tmp_path / "logs", **kwargs)


class TestDependencies:
    def test_a_job_waits_for_what_it_needs_whatever_the_declaration_order(self, tmp_path):
        jobs = [_job("second", tmp_path, needs=("first",)), _job("first", tmp_path)]

        machine = FakeMachine()
        results = _run(jobs, tmp_path, machine)

        assert machine.calls == ["first", "second"]
        assert [results[name].outcome for name in ("first", "second")] == [Outcome.DONE, Outcome.DONE]

    def test_a_failed_job_blocks_what_depended_on_it_and_the_dependent_never_starts(self, tmp_path):
        jobs = [_job("trains", tmp_path, needs=("gate_run",)), _job("gate_run", tmp_path)]

        machine = FakeMachine({"gate_run": Reply(outputs=False, code=1)})
        results = _run(jobs, tmp_path, machine)

        assert machine.calls == ["gate_run"] * MAX_ATTEMPTS
        assert results["gate_run"].outcome is Outcome.FAILED
        assert results["trains"].outcome is Outcome.BLOCKED
        assert "gate_run" in results["trains"].reason

    def test_blocking_is_transitive(self, tmp_path):
        jobs = [_job("a", tmp_path), _job("b", tmp_path, needs=("a",)), _job("c", tmp_path, needs=("b",))]

        machine = FakeMachine({"a": Reply(outputs=False, code=1)})
        results = _run(jobs, tmp_path, machine)

        assert [results[name].outcome for name in "abc"] == [Outcome.FAILED, Outcome.BLOCKED, Outcome.BLOCKED]

    def test_independent_jobs_still_run_after_another_branch_failed(self, tmp_path):
        jobs = [_job("ladder", tmp_path), _job("ladder_probe", tmp_path, needs=("ladder",)), _job("tracking", tmp_path)]

        machine = FakeMachine({"ladder": Reply(outputs=False, code=1)})
        results = _run(jobs, tmp_path, machine)

        assert results["tracking"].outcome is Outcome.DONE
        assert results["ladder_probe"].outcome is Outcome.BLOCKED


class TestGate:
    def test_the_gate_runs_before_any_job(self, tmp_path):
        gate = Job(name="gate", cli=("-m", "pytest", "tests", "-q"), outputs=())

        machine = FakeMachine()
        _run([_job("trains", tmp_path)], tmp_path, machine, gate=gate)

        assert machine.calls == ["gate", "trains"]

    def test_a_job_with_no_outputs_is_judged_by_its_exit_code(self, tmp_path):
        gate = Job(name="gate", cli=("-m", "pytest", "tests", "-q"), outputs=())

        results = _run([], tmp_path, FakeMachine({"gate": Reply(code=0)}), gate=gate)

        assert results["gate"].outcome is Outcome.DONE
        assert results["gate"].attempts == 1

    def test_a_failed_gate_blocks_every_job_and_nothing_trains(self, tmp_path):
        gate = Job(name="gate", cli=("-m", "pytest", "tests", "-q"), outputs=())

        machine = FakeMachine({"gate": Reply(code=1)})
        results = _run([_job("trains", tmp_path)], tmp_path, machine, gate=gate)

        assert machine.calls == ["gate"]
        assert results["gate"].outcome is Outcome.FAILED
        assert results["trains"].outcome is Outcome.BLOCKED

    def test_the_gate_is_not_retried(self, tmp_path):
        gate = Job(name="gate", cli=("-m", "pytest", "tests", "-q"), outputs=())

        machine = FakeMachine({"gate": Reply(code=1)})
        _run([], tmp_path, machine, gate=gate)

        assert machine.calls == ["gate"]


class TestSkipping:
    def test_a_job_whose_outputs_exist_is_skipped_and_never_spawned(self, tmp_path):
        job = _job("trains", tmp_path)
        job.outputs[0].write_text("from an earlier launch", encoding="utf-8")

        machine = FakeMachine()
        results = _run([job], tmp_path, machine)

        assert machine.calls == []
        assert results["trains"].outcome is Outcome.SKIPPED

    def test_a_skipped_job_still_satisfies_what_depended_on_it(self, tmp_path):
        done, waiting = _job("done", tmp_path), _job("waiting", tmp_path, needs=("done",))
        done.outputs[0].write_text("from an earlier launch", encoding="utf-8")

        results = _run([done, waiting], tmp_path, FakeMachine())

        assert results["waiting"].outcome is Outcome.DONE

    def test_a_job_is_skipped_only_when_every_one_of_its_outputs_exists(self, tmp_path):
        job = Job(name="trains", cli=("x.py",), outputs=(tmp_path / "one.json", tmp_path / "two.json"))
        job.outputs[0].write_text("half a run", encoding="utf-8")

        machine = FakeMachine()
        results = _run([job], tmp_path, machine)

        assert machine.calls == ["trains"]
        assert results["trains"].outcome is Outcome.DONE

    def test_steps_restricts_the_run_without_blocking_the_dependents_of_what_was_left_out(self, tmp_path):
        jobs = [_job("first", tmp_path), _job("second", tmp_path, needs=("first",))]

        machine = FakeMachine()
        results = _run(jobs, tmp_path, machine, steps=("second",))

        assert machine.calls == ["second"]
        assert results["first"].outcome is Outcome.SKIPPED
        assert results["second"].outcome is Outcome.DONE


class TestRetries:
    def test_a_job_that_produces_nothing_is_attempted_three_times_then_fails(self, tmp_path):
        machine = FakeMachine({"trains": Reply(outputs=False, code=1)})

        results = _run([_job("trains", tmp_path)], tmp_path, machine)

        assert machine.calls == ["trains"] * MAX_ATTEMPTS
        assert results["trains"].attempts == MAX_ATTEMPTS
        assert results["trains"].outcome is Outcome.FAILED

    def test_a_job_that_succeeds_on_its_second_attempt_is_done(self, tmp_path):
        class Flaky(FakeMachine):
            def __call__(self, job: Job, log: Path) -> int:
                self.replies[job.name] = Reply(outputs=len(self.calls) > 0, code=0)
                return super().__call__(job, log)

        machine = Flaky({"trains": Reply(outputs=False)})
        results = _run([_job("trains", tmp_path)], tmp_path, machine)

        assert results["trains"].attempts == 2
        assert results["trains"].outcome is Outcome.DONE

    def test_a_job_whose_outputs_appeared_is_done_whatever_its_exit_code(self, tmp_path):
        machine = FakeMachine({"trains": Reply(outputs=True, code=1)})

        results = _run([_job("trains", tmp_path)], tmp_path, machine)

        assert machine.calls == ["trains"]
        assert results["trains"].outcome is Outcome.DONE

    def test_each_attempt_writes_its_own_log(self, tmp_path):
        machine = FakeMachine({"trains": Reply(outputs=False, code=1)})

        _run([_job("trains", tmp_path)], tmp_path, machine)

        assert len(list((tmp_path / "logs").glob("trains.*.log"))) == MAX_ATTEMPTS


class TestGuards:
    @pytest.mark.parametrize("line", [
        "the memory's loss is not finite at chunk 7, stopping",
        "the candidates' val AP moved by 0.00 points over three stages, stopping",
        "the training loss is not decreasing, stopping",
    ])
    def test_every_training_guard_is_recognised(self, line):
        assert guard_fired(f"epoch 1.00 ok\n{line}\n") is not None

    def test_an_ordinary_log_fires_no_guard(self):
        assert guard_fired("epoch 1.00 val AP 73.88\nwrote step2_results.json\n") is None

    def test_a_job_whose_guard_fired_is_not_retried(self, tmp_path):
        guarded = Reply(outputs=False, log="the memory's loss is not finite at chunk 7, stopping\n", code=1)

        machine = FakeMachine({"trains": guarded})
        results = _run([_job("trains", tmp_path)], tmp_path, machine)

        assert machine.calls == ["trains"]
        assert results["trains"].outcome is Outcome.FAILED
        assert "not finite at chunk" in results["trains"].reason


class TestValidation:
    def test_a_need_naming_no_job_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="nonexistent"):
            validate([_job("trains", tmp_path, needs=("nonexistent",))])

    def test_two_jobs_of_the_same_name_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="trains"):
            validate([_job("trains", tmp_path), _job("trains", tmp_path)])

    def test_a_dependency_cycle_is_rejected(self, tmp_path):
        jobs = [_job("a", tmp_path, needs=("b",)), _job("b", tmp_path, needs=("a",))]

        with pytest.raises(ValueError, match="cycle"):
            validate(jobs)

    def test_a_queue_is_validated_before_anything_is_spawned(self, tmp_path):
        machine = FakeMachine()

        with pytest.raises(ValueError):
            _run([_job("trains", tmp_path, needs=("nonexistent",))], tmp_path, machine)

        assert machine.calls == []


class TestProgressMarkers:
    def _lines(self, tmp_path: Path) -> list[str]:
        return [line[:-TIMESTAMP_LENGTH].strip() for line in (tmp_path / "progress.txt").read_text(encoding="utf-8").splitlines()]

    def test_a_clean_run_writes_the_markers_the_watchers_poll_for(self, tmp_path):
        _run([_job("trains", tmp_path)], tmp_path, FakeMachine())

        assert self._lines(tmp_path) == [
            "QUEUE chain START",
            "QUEUE trains attempt 1 START",
            "QUEUE trains attempt 1 END exit 0",
            "QUEUE chain DONE",
        ]

    def test_a_skip_a_retry_a_guard_and_a_block_each_name_their_job(self, tmp_path):
        jobs = [_job("skipped", tmp_path), _job("guarded", tmp_path), _job("retried", tmp_path), _job("waiting", tmp_path, needs=("guarded",))]
        jobs[0].outputs[0].write_text("from an earlier launch", encoding="utf-8")
        replies = {
            "guarded": Reply(outputs=False, log="the training loss is not decreasing, stopping\n", code=1),
            "retried": Reply(outputs=False, code=1),
        }

        _run(jobs, tmp_path, FakeMachine(replies))
        lines = self._lines(tmp_path)

        assert "QUEUE skipped SKIP, already done" in lines
        assert "QUEUE guarded STOPPED, a training guard fired" in lines
        assert "QUEUE retried attempt 1 RETRY" in lines
        assert "QUEUE retried STOPPED after three attempts" in lines
        assert "QUEUE waiting BLOCKED, guarded did not finish" in lines
        assert lines[-1].startswith("QUEUE chain STOPPED")

    def test_the_chain_start_line_names_the_steps_when_the_run_is_restricted(self, tmp_path):
        _run([_job("first", tmp_path), _job("second", tmp_path)], tmp_path, FakeMachine(), steps=("second",))

        assert self._lines(tmp_path)[0] == "QUEUE chain START steps second"

    def test_progress_is_appended_so_an_earlier_chain_is_still_readable(self, tmp_path):
        (tmp_path / "progress.txt").write_text("QUEUE chain DONE 2026-10-08T09:00:00\n", encoding="utf-8")

        _run([_job("trains", tmp_path)], tmp_path, FakeMachine())

        assert self._lines(tmp_path)[0] == "QUEUE chain DONE"


class TestReport:
    def test_the_table_carries_every_job_its_outcome_and_its_attempts(self, tmp_path):
        jobs = [_job("trains", tmp_path), _job("waiting", tmp_path, needs=("trains",))]

        results = _run(jobs, tmp_path, FakeMachine({"trains": Reply(outputs=False, code=1)}))
        table = report(results)

        assert "trains" in table and "FAILED" in table and str(MAX_ATTEMPTS) in table
        assert "waiting" in table and "BLOCKED" in table


class TestEntryPoint:
    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        return ["--progress", str(tmp_path / "progress.txt"), "--log-root", str(tmp_path / "logs"), *extra]

    def test_listing_the_queue_runs_nothing(self, tmp_path, capsys):
        jobs = [_job("second", tmp_path, needs=("first",)), _job("first", tmp_path)]

        machine = FakeMachine()
        code = main(jobs, argv=self._argv(tmp_path, "--list"), spawn=machine)

        assert code == 0
        assert machine.calls == []
        assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == ["first", "second"]

    def test_listing_says_which_jobs_are_already_done(self, tmp_path, capsys):
        job = _job("trains", tmp_path)
        job.outputs[0].write_text("from an earlier launch", encoding="utf-8")

        main([job], argv=self._argv(tmp_path, "--list"), spawn=FakeMachine())

        assert "done" in capsys.readouterr().out

    def test_a_clean_queue_exits_zero(self, tmp_path):
        assert main([_job("trains", tmp_path)], argv=self._argv(tmp_path), spawn=FakeMachine()) == 0

    def test_a_queue_with_a_failure_exits_nonzero(self, tmp_path):
        machine = FakeMachine({"trains": Reply(outputs=False, code=1)})

        assert main([_job("trains", tmp_path)], argv=self._argv(tmp_path), spawn=machine) == 1

    def test_the_steps_option_restricts_the_run(self, tmp_path):
        jobs = [_job("first", tmp_path), _job("second", tmp_path)]

        machine = FakeMachine()
        main(jobs, argv=self._argv(tmp_path, "--steps", "second"), spawn=machine)

        assert machine.calls == ["second"]

    def test_the_outcome_table_is_printed(self, tmp_path, capsys):
        main([_job("trains", tmp_path)], argv=self._argv(tmp_path), spawn=FakeMachine())

        assert "trains" in capsys.readouterr().out


class TestSpawnProcess:
    """The one seam that touches the operating system, so it is exercised against real processes."""

    def _job(self, tmp_path: Path, code: str, outputs: tuple[Path, ...] = ()) -> Job:
        return Job(name="probe", cli=("-c", code), outputs=outputs, cwd=tmp_path)

    def test_the_exit_code_reaches_the_caller(self, tmp_path):
        assert spawn_process(self._job(tmp_path, "raise SystemExit(3)"), tmp_path / "probe.log") == 3

    def test_everything_the_job_printed_is_in_its_log(self, tmp_path):
        log = tmp_path / "probe.log"

        spawn_process(self._job(tmp_path, "import sys; print('to stdout'); print('to stderr', file=sys.stderr)"), log)

        assert "to stdout" in log.read_text(encoding="utf-8") and "to stderr" in log.read_text(encoding="utf-8")

    def test_the_job_runs_in_the_working_directory_it_was_given(self, tmp_path):
        job = self._job(tmp_path, "from pathlib import Path; Path('made_here.txt').write_text('x')", (tmp_path / "made_here.txt",))

        assert spawn_process(job, tmp_path / "probe.log") == 0
        assert job.outputs[0].exists()

    def test_a_job_still_running_once_its_outputs_exist_is_waited_out_not_polled_forever(self, tmp_path, monkeypatch):
        monkeypatch.setattr("run_queue.POLL_SECONDS", 0.05)
        output = tmp_path / "late.json"
        code = f"from pathlib import Path; import time; Path(r'{output}').write_text('x'); time.sleep(0.3)"

        assert spawn_process(Job(name="probe", cli=("-c", code), outputs=(output,), cwd=tmp_path), tmp_path / "probe.log") == 0


class TestStayAwake:
    """The chain must not be killed by the host going to sleep (2026-10-17).

    Three deaths, three causes: the session's job object took it on 2026-10-16, the gate blocked it, and then -- with it
    properly detached under the Task Scheduler -- **the machine slept at 22:06:02** and killed the children anyway, four
    seconds after a user-mode process asked to suspend. `o1_rescore_limit` had printed its final evaluation and had not
    yet written `step3a_results.json`, so a finished 2.6-hour arm was lost between one line and the next.

    Detachment does not help with sleep: the request comes from outside the process tree. So the runner asks Windows to
    keep the system up while it has work, which is what `SetThreadExecutionState` is for -- the same API a video player
    or an installer uses. The lock is released when the process exits, including when it is killed, so it cannot leave
    the machine permanently awake.
    """

    def test_it_asks_windows_to_keep_the_system_up(self):
        asked = []

        with stay_awake(setter=lambda flags: asked.append(flags) or 1):
            pass

        assert asked, "nothing was requested"
        assert asked[0] != 0, "the first call must set a keep-awake state, not clear it"

    def test_it_releases_the_lock_on_the_way_out(self):
        asked = []

        with stay_awake(setter=lambda flags: asked.append(flags) or 1):
            pass

        # Releasing is ES_CONTINUOUS on its own -- the keep-awake bit cleared, the "continuous" semantics kept. Zero is
        # not a valid argument, so the contract is "the last call drops ES_SYSTEM_REQUIRED", not "the last call is 0".
        assert asked[-1] == ES_CONTINUOUS
        assert not asked[-1] & ES_SYSTEM_REQUIRED

    def test_it_releases_the_lock_even_when_the_queue_raises(self):
        asked = []

        try:
            with stay_awake(setter=lambda flags: asked.append(flags) or 1):
                raise RuntimeError("a job blew up")
        except RuntimeError:
            pass

        assert asked[-1] == ES_CONTINUOUS

    def test_a_platform_that_cannot_do_this_runs_anyway(self):
        """The runner is not Windows-only in principle, and a missing API must not stop a queue from running."""
        def refuse(flags):
            raise OSError("no such API")

        with stay_awake(setter=refuse):
            pass            # reaching here is the assertion

    def test_a_refusal_from_windows_is_not_fatal_either(self):
        """`SetThreadExecutionState` returns 0 on failure; a queue that cannot hold the lock should still run."""
        with stay_awake(setter=lambda flags: 0):
            pass
