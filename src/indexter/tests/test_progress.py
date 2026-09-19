import io
import re
import sys
import time

import pytest
from rich.console import Console

from indexter.progress import ConsoleProgress, NullProgress, Progress, RecordingProgress


class _TrippedStream:
    """A stream that records writes and fails if anyone asks whether it's a terminal."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        raise AssertionError("terminal detection was attempted")

    def fileno(self) -> int:
        raise AssertionError("terminal detection was attempted")


def _drive(progress: Progress) -> None:
    progress.model_classified("acquisition", "some/model")
    progress.phase_start("model")
    progress.phase_done("model")
    progress.phase_start("files")
    progress.advance()
    progress.phase_done("files")
    progress.phase_start("embed", total=3)
    progress.advance(2)
    progress.advance()
    progress.phase_done("embed")


class TestNullProgress:
    def test_satisfies_the_protocol(self):
        assert isinstance(NullProgress(), Progress)

    def test_writes_nothing_and_never_detects_a_terminal(self, monkeypatch, capfd):
        out, err = _TrippedStream(), _TrippedStream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setattr("os.isatty", lambda _fd: pytest.fail("terminal detection was attempted"))

        _drive(NullProgress())

        assert out.written == []
        assert err.written == []
        captured = capfd.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestRecordingProgress:
    def test_satisfies_the_protocol(self):
        assert isinstance(RecordingProgress(), Progress)

    def test_captures_events_in_order(self):
        recorder = RecordingProgress()

        _drive(recorder)

        assert recorder.events == [
            ("model", "acquisition", "some/model"),
            ("start", "model", None),
            ("done", "model"),
            ("start", "files", None),
            ("advance", "files", 1),
            ("done", "files"),
            ("start", "embed", 3),
            ("advance", "embed", 2),
            ("advance", "embed", 1),
            ("done", "embed"),
        ]

    def test_helpers(self):
        recorder = RecordingProgress()

        _drive(recorder)

        assert recorder.started() == ["model", "files", "embed"]
        assert recorder.advanced("embed") == 3
        assert recorder.advanced("files") == 1
        assert recorder.advanced("resolve") == 0


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_COLOUR_SGR = re.compile(r"\x1b\[[0-9;]*(?:3[0-9]|4[0-9]|9[0-7]|10[0-7]|38|48)[0-9;]*m")
SLOW = 0.02  # a threshold short enough that a test can outwait it
NEVER = 60.0  # a threshold no test will reach


def _console(*, terminal: bool = True, width: int = 100) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=terminal, width=width, color_system="standard" if terminal else None)
    return console, buffer


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition not reached in time")
        time.sleep(0.005)


def _render_active(progress: ConsoleProgress) -> str:
    console, buffer = _console()
    console.print(progress._active_line())
    return _plain(buffer.getvalue())


class TestConsoleProgressSatisfiesProtocol:
    def test_is_a_progress(self):
        console, _ = _console()
        assert isinstance(ConsoleProgress(console), Progress)


class TestLazyPainting:
    def test_phase_under_threshold_paints_nothing(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=NEVER) as progress:
            progress.phase_start("embed", total=10)
            progress.advance(10)
            progress.phase_done("embed")

        assert buffer.getvalue() == ""

    def test_phase_over_threshold_paints_active_then_completion(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("embed", total=10)
            progress.advance(3)
            _wait_for(lambda: "Embedding" in buffer.getvalue())
            progress.advance(7)
            progress.phase_done("embed")

        text = _plain(buffer.getvalue())
        assert "Embedding" in text
        assert text.rindex("\u2713 Embedded 10 nodes") > text.index("Embedding")
        assert progress._live is None

    def test_fast_phase_between_slow_ones_stays_silent(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=NEVER) as progress:
            for phase in ("files", "resolve", "embed"):
                progress.phase_start(phase)
                progress.phase_done(phase)

        assert buffer.getvalue() == ""

    def test_phase_ending_before_the_timer_fires_never_paints(self):
        console, buffer = _console()
        progress = ConsoleProgress(console, threshold=0.05)
        progress.phase_start("files")
        progress.phase_done("files")
        time.sleep(0.15)  # the cancelled timer must not paint late

        assert buffer.getvalue() == ""
        assert progress._live is None


class TestActiveLine:
    def test_determinate_phase_shows_a_bar_with_counts(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.phase_start("embed", total=4863)
            progress.advance(2976)

            text = _render_active(progress)

        assert "61%" in text
        assert "2,976/4,863" in text
        assert re.search(r"[\u2501\u2578\u257a\u2500]", text)

    def test_indeterminate_phase_shows_a_running_count_only(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.phase_start("files")
            progress.advance(143)

            text = _render_active(progress)

        assert "143 files" in text
        assert "%" not in text
        assert not re.search(r"[\u2501\u2578\u257a\u2500]", text)

    def test_a_single_file_is_singular(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.phase_start("files")
            progress.advance()

            text = _render_active(progress)

        assert "1 file" in text
        assert "1 files" not in text

    def test_elapsed_time_follows_the_clock(self):
        now = [100.0]
        with ConsoleProgress(threshold=NEVER, clock=lambda: now[0]) as progress:
            progress.phase_start("embed", total=10)
            now[0] = 108.0

            assert "0:08" in _render_active(progress)

    def test_count_never_shows_past_the_total(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.phase_start("embed", total=5)
            progress.advance(9)

            assert "5/5" in _render_active(progress)


class TestOnlyTheActivePhaseAnimates:
    def test_finished_phases_are_static_lines_and_one_live_at_a_time(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("files")
            progress.advance(219)
            _wait_for(lambda: progress._live is not None)
            progress.phase_done("files")
            assert progress._live is None

            progress.phase_start("embed", total=8)
            _wait_for(lambda: progress._live is not None)
            live_during_embed = progress._live
            progress.phase_done("embed")

        text = _plain(buffer.getvalue())
        assert live_during_embed is not None
        assert text.index("\u2713 Indexed 219 files") < text.index("\u2713 Embedded 8 nodes")
        assert text.count("\u2713 Indexed 219 files") == 1

    def test_close_discards_an_unfinished_phase_without_a_completion_line(self):
        console, buffer = _console()
        progress = ConsoleProgress(console, threshold=SLOW)
        progress.phase_start("files")
        _wait_for(lambda: progress._live is not None)

        progress.close()

        assert progress._live is None
        assert "\u2713" not in _plain(buffer.getvalue())


class TestModelLoadNarration:
    def test_acquisition_names_model_says_one_time_and_shows_elapsed(self):
        now = [0.0]
        with ConsoleProgress(threshold=NEVER, clock=lambda: now[0]) as progress:
            progress.model_classified("acquisition", "org/some-model")
            progress.phase_start("model")
            now[0] = 7.0

            text = _render_active(progress)

        assert "org/some-model" in text
        assert "one-time" in text
        assert "0:07" in text

    def test_acquisition_promises_no_total_percentage_or_estimate(self):
        now = [0.0]
        with ConsoleProgress(threshold=NEVER, clock=lambda: now[0]) as progress:
            progress.model_classified("acquisition", "org/some-model")
            progress.phase_start("model")
            now[0] = 45.0
            progress.advance(3)

            text = _render_active(progress)

        assert "%" not in text
        assert not re.search(r"\d\s?[KMGT]i?B", text)
        assert not re.search(r"remaining|eta|left", text, re.IGNORECASE)
        assert not re.search(r"[\u2501\u2578\u257a\u2500]", text)

    def test_cached_load_says_loading_and_names_the_model(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.model_classified("cached", "org/some-model")
            progress.phase_start("model")

            text = _render_active(progress)

        assert "Loading" in text
        assert "org/some-model" in text
        assert "one-time" not in text

    def test_reclassification_mid_phase_changes_the_narration(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.model_classified("cached", "org/some-model")
            progress.phase_start("model")
            progress.model_classified("acquisition", "org/some-model")

            assert "one-time" in _render_active(progress)

    def test_completion_lines_follow_the_final_classification(self):
        for kind, verb in (("cached", "Loaded"), ("acquisition", "Downloaded")):
            console, buffer = _console()
            with ConsoleProgress(console, threshold=SLOW) as progress:
                progress.model_classified(kind, "org/some-model")
                progress.phase_start("model")
                _wait_for(lambda: progress._live is not None)
                progress.phase_done("model")

            assert f"\u2713 {verb} embedding model" in _plain(buffer.getvalue())

    def test_classification_does_not_leak_into_the_next_load(self):
        with ConsoleProgress(threshold=NEVER) as progress:
            progress.model_classified("acquisition", "org/some-model")
            progress.phase_start("model")
            progress.phase_done("model")
            progress.phase_start("model")

            assert "Preparing embedding model" in _render_active(progress)


class TestThrottling:
    def test_events_never_drive_repaints(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=SLOW, refresh_hz=0.01) as progress:
            progress.phase_start("embed", total=100_000)
            _wait_for(lambda: progress._live is not None)
            before = buffer.getvalue()

            for _ in range(100_000):
                progress.advance()

            assert buffer.getvalue() == before

    def test_refresh_rate_is_handed_to_the_live_display(self):
        console, _ = _console()
        with ConsoleProgress(console, threshold=SLOW, refresh_hz=7) as progress:
            progress.phase_start("files")
            _wait_for(lambda: progress._live is not None)

            assert progress._live.refresh_per_second == 7


class TestNonInteractiveConsole:
    def test_slow_phase_prints_static_lines_without_control_codes(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.model_classified("acquisition", "org/some-model")
            progress.phase_start("model")
            _wait_for(lambda: "org/some-model" in buffer.getvalue())
            progress.phase_done("model")

        text = buffer.getvalue()
        assert "\x1b" not in text
        assert "Downloading embedding model (one-time): org/some-model..." in text
        assert "\u2713 Downloaded embedding model" in text

    def test_fast_phase_prints_nothing(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=NEVER) as progress:
            progress.phase_start("files")
            progress.phase_done("files")

        assert buffer.getvalue() == ""


class TestCompletionLines:
    def test_a_single_indexed_file_is_singular(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("files")
            progress.advance()
            _wait_for(lambda: "Indexing files..." in buffer.getvalue())
            progress.phase_done("files")

        assert "\u2713 Indexed 1 file\n" in buffer.getvalue()


class TestDumbTerminal:
    def test_dumb_terminal_gets_static_lines_not_a_silent_live_display(self):
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100, _environ={"TERM": "dumb"})
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("resolve")
            _wait_for(lambda: "Resolving graph..." in buffer.getvalue())
            assert progress._live is None
            progress.phase_done("resolve")

        assert "\u2713 Resolved graph" in buffer.getvalue()


class TestColour:
    def test_no_colour_request_strips_colour_escapes(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100, color_system="standard")
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("embed", total=4)
            progress.advance(2)
            _wait_for(lambda: "Embedding" in buffer.getvalue())
            progress.phase_done("embed")

        raw = buffer.getvalue()
        assert "\u2713 Embedded 4 nodes" in _plain(raw)
        assert _COLOUR_SGR.search(raw) is None

    def test_colour_is_used_when_not_requested_off(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100, color_system="standard")
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("files")
            _wait_for(lambda: progress._live is not None)
            progress.phase_done("files")

        assert _COLOUR_SGR.search(buffer.getvalue()) is not None


class TestPhaseBookkeeping:
    def test_completing_a_phase_that_is_not_active_is_ignored(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("files")
            _wait_for(lambda: progress._live is not None)

            progress.phase_done("embed")

            assert progress._live is not None
            assert progress._phase == "files"
        assert "\u2713" not in _plain(buffer.getvalue())

    def test_a_stale_timer_cannot_paint_a_later_phase(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=NEVER) as progress:
            progress.phase_start("files")
            stale = progress._generation
            progress.phase_done("files")
            progress.phase_start("resolve")

            progress._paint(stale)

            assert progress._live is None
        assert buffer.getvalue() == ""

    def test_starting_a_phase_discards_an_unfinished_one(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=NEVER) as progress:
            progress.phase_start("files")
            progress.advance(5)
            progress.phase_start("embed", total=3)

            assert progress._phase == "embed"
            assert progress._count == 0
        assert buffer.getvalue() == ""

    def test_rendering_with_no_active_phase_is_empty(self):
        console, _ = _console()
        progress = ConsoleProgress(console, threshold=NEVER)

        assert _render_active(progress).strip() == ""

    def test_resolve_phase_has_a_spinner_line_and_a_completion_line(self):
        console, buffer = _console()
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("resolve")
            _wait_for(lambda: "Resolving graph" in buffer.getvalue())
            progress.phase_done("resolve")

        assert "\u2713 Resolved graph" in _plain(buffer.getvalue())

    def test_embedding_announces_its_size_in_a_log(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("embed", total=4863)
            _wait_for(lambda: "Embedding 4,863 nodes..." in buffer.getvalue())
            progress.phase_done("embed")

        assert "\u2713 Embedded 4,863 nodes" in buffer.getvalue()

    def test_indexing_files_is_announced_in_a_log_and_counted_on_completion(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("files")
            progress.advance(219)
            _wait_for(lambda: "Indexing files..." in buffer.getvalue())
            progress.phase_done("files")

        assert "\u2713 Indexed 219 files" in buffer.getvalue()

    def test_unclassified_model_load_reads_as_preparing(self):
        console, buffer = _console(terminal=False)
        with ConsoleProgress(console, threshold=SLOW) as progress:
            progress.phase_start("model")
            _wait_for(lambda: "Preparing embedding model..." in buffer.getvalue())
            progress.phase_done("model")

        assert "\u2713 Prepared embedding model" in buffer.getvalue()
