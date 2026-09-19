import errno
import io
import itertools
import os
import pty
import re
import select
import sys
import termios
import threading
import time
import tty

import pytest
from rich.console import Console

from indexter import banner
from indexter._logo import LOGO
from indexter.banner import (
    DURATION_SECONDS,
    FRAMES_PER_SECOND,
    PALETTES,
    Palette,
    detect_background,
    parse_colorfgbg,
    parse_osc11,
    query_terminal_background,
    render_frame,
    show_banner,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_HEX = re.compile(r"^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$")

# Distinct colours per role, so a test can tell the glint from the gradient and the sparkles.
TEST_PALETTE = Palette(start="#101010", end="#202020", shine="#ff0000", sparkle="#00ff00")
TAGLINE = "v9.9.9 · code search for agents"


def _gradient_reds(text) -> list[int]:
    """The red channel of every gradient/glint span (sparkles are bold, the tagline is dim)."""
    reds = []
    for span in text.spans:
        match = _HEX.match(str(span.style))
        if match:
            reds.append(int(match.group(1), 16))
    return reds


class TestRenderFrame:
    def test_still_header_shows_the_art_unchanged(self):
        text = render_frame(0.0, TEST_PALETTE, final=True)

        assert text.plain == "\n".join(LOGO) + "\n"

    def test_still_header_has_no_glint(self):
        assert max(_gradient_reds(render_frame(0.0, TEST_PALETTE, final=True))) <= 0x20

    def test_first_and_last_animation_frames_are_glint_free(self):
        for t in (0.0, DURATION_SECONDS):
            assert max(_gradient_reds(render_frame(t, TEST_PALETTE))) <= 0x20

    def test_glint_crosses_the_art_midway(self):
        text = render_frame(DURATION_SECONDS / 2, TEST_PALETTE)

        assert max(_gradient_reds(text)) > 0xC0

    def test_gradient_runs_from_start_to_end_colour(self):
        text = render_frame(0.0, Palette("#000000", "#ffffff", "#ff0000", "#00ff00"), final=True, art=("███",))

        assert [str(span.style) for span in text.spans] == ["#000000", "#808080", "#ffffff"]

    def test_gradient_on_a_single_column_takes_the_start_colour(self):
        text = render_frame(0.0, Palette("#000000", "#ffffff", "#ff0000", "#00ff00"), final=True, art=("█",))

        assert [str(span.style) for span in text.spans] == ["#000000"]

    def test_sparkles_twinkle_then_rest_on_their_glyph(self):
        first = render_frame(0.0, TEST_PALETTE).plain
        later = render_frame(0.1, TEST_PALETTE).plain
        rested = render_frame(0.0, TEST_PALETTE, final=True).plain

        assert first != later
        assert rested.count("✦") + rested.count("✧") == sum(line.count(g) for line in LOGO for g in "✦✧")

    def test_sparkles_take_the_sparkle_colour(self):
        text = render_frame(0.0, TEST_PALETTE)

        assert any(str(span.style) == "bold #00ff00" for span in text.spans)

    def test_tagline_is_centred_under_the_art(self):
        plain = render_frame(0.0, TEST_PALETTE, tagline=TAGLINE, final=True).plain
        last = plain.splitlines()[-1]

        assert last.strip() == TAGLINE
        assert len(last) - len(last.lstrip()) == (max(map(len, LOGO)) - len(TAGLINE)) // 2

    def test_both_palettes_are_defined(self):
        assert set(PALETTES) == {"dark", "light"}


class TestLogoArt:
    def test_is_a_sized_block_of_art(self):
        assert LOGO
        assert max(map(len, LOGO)) <= 52
        assert all(line == line.rstrip() for line in LOGO)

    def test_has_sparkles_above_the_glasses(self):
        assert any(g in line for line in LOGO[:3] for g in "✦✧")


class TestParseOsc11:
    def test_four_digit_components(self):
        assert parse_osc11("\x1b]11;rgb:ffff/0000/8080\x07") == pytest.approx((1.0, 0.0, 0x8080 / 0xFFFF))

    def test_two_digit_components(self):
        assert parse_osc11("\x1b]11;rgb:ff/00/80\x1b\\") == pytest.approx((1.0, 0.0, 0x80 / 0xFF))

    def test_one_digit_components(self):
        assert parse_osc11("rgb:f/0/8") == pytest.approx((1.0, 0.0, 8 / 15))

    @pytest.mark.parametrize("reply", ["", "garbage", "\x1b]11;\x07", "rgb:zz/00/00"])
    def test_no_colour_in_reply(self, reply):
        assert parse_osc11(reply) is None


class TestParseColorfgbg:
    @pytest.mark.parametrize("value", ["15;0", "15;default;0", "7;6", "0;8"])
    def test_dark_backgrounds(self, value):
        assert parse_colorfgbg(value) == "dark"

    @pytest.mark.parametrize("value", ["0;15", "0;7", "0;default;15", "0;9"])
    def test_light_backgrounds(self, value):
        assert parse_colorfgbg(value) == "light"

    @pytest.mark.parametrize("value", [None, "", "default;default", "0;99", "0;-1", "nonsense"])
    def test_unusable_values(self, value):
        assert parse_colorfgbg(value) is None


class TestDetectBackground:
    def test_dark_reply(self):
        assert detect_background({}, lambda: "\x1b]11;rgb:1e1e/1e1e/1e1e\x07") == "dark"

    def test_light_reply(self):
        assert detect_background({}, lambda: "\x1b]11;rgb:ffff/ffff/ffff\x07") == "light"

    def test_reply_beats_the_environment(self):
        assert detect_background({"COLORFGBG": "15;0"}, lambda: "rgb:ffff/ffff/ffff") == "light"

    def test_silent_terminal_falls_back_to_colorfgbg(self):
        assert detect_background({"COLORFGBG": "0;15"}, lambda: None) == "light"

    def test_unparseable_reply_falls_back_to_colorfgbg(self):
        assert detect_background({"COLORFGBG": "0;15"}, lambda: "garbage") == "light"

    def test_nothing_known_assumes_dark(self):
        assert detect_background({}, lambda: None) == "dark"

    def test_reads_the_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv("COLORFGBG", "0;15")

        assert detect_background(query=lambda: None) == "light"


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
class TestQueryTerminalBackground:
    @pytest.fixture
    def pty_pair(self):
        master, slave = pty.openpty()
        yield master, slave
        for fd in (master, slave):
            try:
                os.close(fd)
            except OSError:
                pass

    OSC11_REPLY = b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07"
    DA1_REPLY = b"\x1b[?62;22c"

    @staticmethod
    def _terminal_answers(master: int, reply: bytes, delay: float = 0.0) -> threading.Thread:
        def answer():
            os.read(master, 64)  # blocks until the query arrives
            time.sleep(delay)
            os.write(master, reply)

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        return thread

    def test_returns_the_terminals_reply(self, pty_pair):
        master, slave = pty_pair
        thread = self._terminal_answers(master, self.OSC11_REPLY + self.DA1_REPLY)

        reply = query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        assert reply is not None
        assert parse_osc11(reply) == pytest.approx((0x1E1E / 0xFFFF,) * 3)

    def test_a_slow_reply_is_still_read(self, pty_pair):
        master, slave = pty_pair
        thread = self._terminal_answers(master, self.OSC11_REPLY + self.DA1_REPLY, delay=0.2)

        reply = query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        assert reply is not None
        assert parse_osc11(reply) == pytest.approx((0x1E1E / 0xFFFF,) * 3)

    def test_nothing_is_left_pending_after_a_complete_answer(self, pty_pair):
        master, slave = pty_pair
        thread = self._terminal_answers(master, self.OSC11_REPLY + self.DA1_REPLY)

        query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        assert not select.select([slave], [], [], 0.05)[0]

    def test_a_terminal_without_osc11_answers_promptly(self, pty_pair):
        master, slave = pty_pair
        thread = self._terminal_answers(master, self.DA1_REPLY)

        started = time.monotonic()
        reply = query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        assert time.monotonic() - started < 1.0
        assert reply is not None
        assert parse_osc11(reply) is None

    def test_keys_typed_after_the_reply_are_kept(self, pty_pair):
        master, slave = pty_pair
        thread = self._terminal_answers(master, self.OSC11_REPLY + self.DA1_REPLY + b"ls\n")

        query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        assert os.read(slave, 16) == b"ls\n"

    def test_typed_ahead_input_skips_the_query(self, pty_pair):
        master, slave = pty_pair
        os.write(master, b"ls\n")
        assert select.select([slave], [], [], 2.0)[0]

        assert query_terminal_background(os.ttyname(slave), timeout=0.2) is None

        assert os.read(slave, 16) == b"ls\n"
        # The master sees only the line discipline's echo of the keys, never the query.
        assert b"]11;?" not in os.read(master, 64)

    def test_silent_terminal_gives_none_after_the_timeout(self, pty_pair):
        _, slave = pty_pair

        assert query_terminal_background(os.ttyname(slave), timeout=0.05) is None

    def test_a_reply_after_the_timeout_is_not_left_for_the_shell(self, pty_pair, monkeypatch):
        master, slave = pty_pair
        # A sweep wide enough that the reply lands well inside it, never on its edge.
        monkeypatch.setattr(banner, "DRAIN_QUIET_SECONDS", 0.3)
        monkeypatch.setattr(banner, "DRAIN_LIMIT_SECONDS", 0.3)
        thread = self._terminal_answers(master, self.OSC11_REPLY + self.DA1_REPLY, delay=0.1)

        reply = query_terminal_background(os.ttyname(slave), timeout=0.01)
        thread.join(timeout=2.0)

        assert reply is None  # we gave up before it answered
        # A leaked reply holds no newline, so canonical mode hands over nothing until the
        # user hits Enter -- and then it arrives on the front of their command. Look at the
        # queue the way the shell eventually will. TCSANOW, not setraw's default TCSAFLUSH:
        # that discards the very input under test, and waits on output the reply's own echo
        # has already stalled.
        tty.setraw(slave, termios.TCSANOW)
        assert not select.select([slave], [], [], 0.3)[0]

    def test_a_silent_terminal_does_not_stall_on_the_drain(self, pty_pair):
        _, slave = pty_pair

        started = time.monotonic()
        query_terminal_background(os.ttyname(slave), timeout=0.05)

        assert time.monotonic() - started < 0.5

    def test_terminal_mode_is_restored(self, pty_pair):
        master, slave = pty_pair
        before = termios.tcgetattr(slave)
        thread = self._terminal_answers(master, b"\x1b]11;rgb:0000/0000/0000\x07" + self.DA1_REPLY)

        query_terminal_background(os.ttyname(slave), timeout=2.0)
        thread.join(timeout=2.0)

        after = termios.tcgetattr(slave)
        # PENDIN is set by the kernel itself when input is left queued across the switch back to
        # canonical mode; it is a status flag, not a mode that we changed.
        pendin = getattr(termios, "PENDIN", 0)
        after[3] &= ~pendin
        assert after == before

    def test_terminal_mode_is_restored_when_the_sweep_fails(self, pty_pair, monkeypatch):
        master, slave = pty_pair
        before = termios.tcgetattr(slave)

        def hung_up(*args):
            raise OSError(errno.EIO, "Input/output error")  # a pty master that went away

        monkeypatch.setattr(banner, "_drain", hung_up)
        thread = self._terminal_answers(master, self.OSC11_REPLY)  # no DA1, so the sweep runs

        assert query_terminal_background(os.ttyname(slave), timeout=0.05) is None
        thread.join(timeout=2.0)

        after = termios.tcgetattr(slave)
        pendin = getattr(termios, "PENDIN", 0)
        after[3] &= ~pendin
        assert after == before

    def test_a_query_that_does_not_fit_gives_none_at_once(self, pty_pair, monkeypatch):
        _, slave = pty_pair
        # A terminal that took part of the query can never answer it, and waiting
        # out the timeout for a reply that is not coming only makes init slower.
        monkeypatch.setattr(banner, "_write_all", lambda fd, data, deadline: len(data) - 1)

        started = time.monotonic()
        assert query_terminal_background(os.ttyname(slave), timeout=2.0) is None
        assert time.monotonic() - started < 1.0

    def test_closed_terminal_gives_none(self, pty_pair):
        master, slave = pty_pair
        path = os.ttyname(slave)
        os.close(master)

        assert query_terminal_background(path, timeout=0.05) is None

    def test_no_terminal_gives_none(self):
        assert query_terminal_background("/nonexistent/tty") is None

    def test_background_job_leaves_the_terminal_alone(self, pty_pair, monkeypatch):
        _, slave = pty_pair
        monkeypatch.setattr(os, "tcgetpgrp", lambda fd: os.getpgrp() + 1)

        assert query_terminal_background(os.ttyname(slave), timeout=0.05) is None

    def test_termios_failure_gives_none(self, pty_pair, monkeypatch):
        _, slave = pty_pair

        def boom(fd):
            raise termios.error(25, "Inappropriate ioctl for device")

        monkeypatch.setattr(termios, "tcgetattr", boom)

        assert query_terminal_background(os.ttyname(slave), timeout=0.05) is None

    def test_platform_without_termios_gives_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "termios", None)

        assert query_terminal_background() is None


class TestWriteAll:
    @pytest.fixture
    def pipe(self):
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        yield read_fd, write_fd
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def test_writes_every_byte(self, pipe):
        read_fd, write_fd = pipe

        assert banner._write_all(write_fd, b"\x1b]11;?\x07", time.monotonic() + 2.0) == 7
        assert os.read(read_fd, 16) == b"\x1b]11;?\x07"

    def test_a_full_pipe_stops_at_the_deadline(self, pipe):
        _, write_fd = pipe
        try:
            while True:
                os.write(write_fd, b"\0" * 4096)
        except BlockingIOError:
            pass

        started = time.monotonic()
        assert banner._write_all(write_fd, b"\x1b]11;?\x07", time.monotonic() + 0.1) == 0
        assert time.monotonic() - started < 1.0

    def test_a_short_write_is_finished_once_there_is_room(self, pipe):
        read_fd, write_fd = pipe
        size = os.fpathconf(write_fd, "PC_PIPE_BUF")
        try:
            while True:
                os.write(write_fd, b"\0" * size)
        except BlockingIOError:
            pass

        def make_room():
            time.sleep(0.05)
            os.read(read_fd, size * 4)

        thread = threading.Thread(target=make_room, daemon=True)
        thread.start()
        assert banner._write_all(write_fd, b"\x1b]11;?\x07", time.monotonic() + 2.0) == 7
        thread.join(timeout=2.0)


class _AsciiStream(io.StringIO):
    encoding = "ascii"


class _DyingStream(io.StringIO):
    """A terminal that hangs up part-way through the animation."""

    def __init__(self, writes_before_hangup: int = 3):
        super().__init__()
        self._remaining = writes_before_hangup

    def write(self, text: str) -> int:
        self._remaining -= 1
        if self._remaining < 0:
            raise OSError(errno.EIO, "Input/output error")
        return super().write(text)


def _console(*, width: int = 80, height: int = 30, no_color: bool = False, terminal: bool = True, stream=None):
    buffer = stream if stream is not None else io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=terminal,
        width=width,
        height=height,
        no_color=no_color,
        color_system="truecolor" if terminal else None,
    )
    return console, buffer


class TestShowBanner:
    @pytest.fixture(autouse=True)
    def _no_real_terminal(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        # Never let a test ask the developer's actual terminal for its colours.
        monkeypatch.setattr(banner, "detect_background", lambda: "dark")

    @staticmethod
    def _play(console, **kwargs) -> list[float]:
        ticks = itertools.count(0, 0.3)
        sleeps: list[float] = []
        show_banner(console, version="9.9.9", clock=lambda: next(ticks), sleep=sleeps.append, **kwargs)
        return sleeps

    def test_animates_then_leaves_the_still_header(self):
        console, buffer = _console()

        sleeps = self._play(console)
        shown = _ANSI.sub("", buffer.getvalue())

        assert len(sleeps) > 1
        assert set(sleeps) == {1 / FRAMES_PER_SECOND}
        assert LOGO[4] in shown
        assert shown.count(TAGLINE) >= 1
        assert shown.rstrip().endswith(TAGLINE)

    def test_the_background_picks_the_palette(self):
        dark, dark_out = _console()
        light, light_out = _console()

        self._play(dark, background="dark")
        self._play(light, background="light")

        assert dark_out.getvalue() != light_out.getvalue()

    def test_an_unspecified_background_is_detected(self, monkeypatch):
        monkeypatch.setattr(banner, "detect_background", lambda: "light")
        detected, detected_out = _console()
        explicit, explicit_out = _console()

        self._play(detected)
        self._play(explicit, background="light")

        assert detected_out.getvalue() == explicit_out.getvalue()

    def test_no_colour_prints_the_still_header_once_without_animating(self):
        console, buffer = _console(no_color=True)

        sleeps = self._play(console)
        out = buffer.getvalue()

        assert sleeps == []
        assert out.count(TAGLINE) == 1
        assert LOGO[4] in _ANSI.sub("", out)
        assert not re.search(r"\x1b\[[0-9;]*(?:3[0-7]|38|9[0-7])[0-9;]*m", out)

    def test_no_colour_does_not_ask_the_terminal(self, monkeypatch):
        def detected():
            raise AssertionError("must not ask about a background no colour of ours will use")

        monkeypatch.setattr(banner, "detect_background", detected)
        console, buffer = _console(no_color=True)

        self._play(console)

        assert LOGO[4] in _ANSI.sub("", buffer.getvalue())

    def test_a_terminal_that_goes_away_mid_draw_is_not_an_error(self):
        console, _ = _console(stream=_DyingStream())

        self._play(console)  # the command it introduces must survive

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"terminal": False},
            {"width": 40},
            {"height": 10},
            {"stream": _AsciiStream()},
        ],
        ids=["not-a-terminal", "too-narrow", "too-short", "ascii-only"],
    )
    def test_prints_nothing_where_it_cannot_draw(self, kwargs, monkeypatch):
        def detected():
            raise AssertionError("must not ask the terminal when there is nothing to draw")

        monkeypatch.setattr(banner, "detect_background", detected)
        console, buffer = _console(**kwargs)

        sleeps = self._play(console)

        assert buffer.getvalue() == ""
        assert sleeps == []

    def test_prints_nothing_on_a_dumb_terminal(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        console, buffer = _console()

        self._play(console)

        assert buffer.getvalue() == ""

    def test_defaults_to_stderr(self, capsys):
        show_banner(version="9.9.9")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""  # capsys stderr is not a terminal, so nothing is drawn
