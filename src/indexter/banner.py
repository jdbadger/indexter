"""The logo banner `indexter init` shows the first time it indexes a repository.

The art (`indexter._logo`) is drawn in grayscale with a gradient, and one
slanted glint sweeps across it while the sparkles twinkle; then it settles
into a still header. Nothing is drawn unless stderr is a real terminal wide
and tall enough to hold it -- there is no fallback art, so a pipe, a log file
or a dumb terminal never sees escape sequences or half a logo.

Which of two palettes to use depends on whether the terminal background is
dark or light, found by asking the terminal (OSC 11, followed by a
device-attributes query that every terminal answers, so the reply is
known to be complete), then `COLORFGBG`, then assuming dark. Rendering is a
pure function of the elapsed time, and the player takes its clock and sleep as
parameters, so tests run without waiting.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from indexter._logo import LOGO

if TYPE_CHECKING:
    from rich.console import Console
    from rich.text import Text

Background = Literal["dark", "light"]

DURATION_SECONDS = 1.2
FRAMES_PER_SECOND = 30
GLINT_WIDTH = 8.0
GLINT_SLANT = 1.5
SPARKLES = "✦✧"
TWINKLE = "·✧✦✧"
QUERY_TIMEOUT_SECONDS = 0.5
DRAIN_QUIET_SECONDS = 0.05
DRAIN_LIMIT_SECONDS = 0.25


@dataclass(frozen=True)
class Palette:
    """Hex colours: the logo runs `start` to `end` left to right, the glint
    passes through `shine`, and the sparkles are `sparkle`."""

    start: str
    end: str
    shine: str
    sparkle: str


PALETTES: dict[Background, Palette] = {
    "dark": Palette(start="#8a8a8a", end="#d4d4d4", shine="#ffffff", sparkle="#ffffff"),
    "light": Palette(start="#111111", end="#4a4a4a", shine="#b8b8b8", sparkle="#3a3a3a"),
}

_RGB = tuple[int, int, int]


def _rgb(colour: str) -> _RGB:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _mix(a: _RGB, b: _RGB, fraction: float) -> _RGB:
    return (
        round(a[0] + (b[0] - a[0]) * fraction),
        round(a[1] + (b[1] - a[1]) * fraction),
        round(a[2] + (b[2] - a[2]) * fraction),
    )


def _hex(colour: _RGB) -> str:
    r, g, b = colour
    return f"#{r:02x}{g:02x}{b:02x}"


def render_frame(
    t: float,
    palette: Palette,
    *,
    tagline: str = "",
    final: bool = False,
    art: tuple[str, ...] = LOGO,
) -> Text:
    """The banner `t` seconds into the animation. `final` is the still header:
    no glint, and the sparkles at rest. The glint is fully off the art at
    t=0 and t=DURATION_SECONDS, so the first and last frames are the plain one.
    """
    from rich.text import Text

    width = max(map(len, art))
    ramp = max(1, width - 1)  # the last column lands on `end`, not one step short of it
    span = width + len(art) * GLINT_SLANT
    centre = -GLINT_WIDTH + (t / DURATION_SECONDS) * (span + 2 * GLINT_WIDTH)
    start, end, shine = _rgb(palette.start), _rgb(palette.end), _rgb(palette.shine)

    out = Text(no_wrap=True)
    sparkle_index = 0
    for y, line in enumerate(art):
        for x, ch in enumerate(line):
            if ch == " ":
                out.append(" ")
            elif ch in SPARKLES:
                glyph = ch if final else TWINKLE[(int(t * 10) + sparkle_index * 2) % len(TWINKLE)]
                sparkle_index += 1
                out.append(glyph, style=f"bold {palette.sparkle}")
            else:
                glint = 0.0 if final else max(0.0, 1 - abs(x + y * GLINT_SLANT - centre) / GLINT_WIDTH)
                out.append(ch, style=_hex(_mix(_mix(start, end, x / ramp), shine, glint * 0.85)))
        out.append("\n")
    if tagline:
        out.append(" " * max(0, (width - len(tagline)) // 2) + tagline, style="dim")
    return out


_OSC11_REPLY = re.compile(r"rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})")


def parse_osc11(reply: str) -> tuple[float, float, float] | None:
    """The `(r, g, b)`, each 0..1, in a terminal's answer to the OSC 11
    background-colour query (`ESC ] 11 ; rgb:RRRR/GGGG/BBBB BEL`), or None if
    the reply holds no colour."""
    match = _OSC11_REPLY.search(reply)
    if match is None:
        return None
    r, g, b = (int(part, 16) / (16 ** len(part) - 1) for part in match.groups())
    return r, g, b


def parse_colorfgbg(value: str | None) -> Background | None:
    """The background named by `COLORFGBG` (`fg;bg`, sometimes `fg;default;bg`),
    or None when it is unset or unusable."""
    if not value:
        return None
    try:
        index = int(value.rsplit(";", 1)[-1])
    except ValueError:
        return None
    if index in range(0, 7) or index == 8:
        return "dark"
    if index == 7 or index in range(9, 16):
        return "light"
    return None


_QUERY = b"\x1b]11;?\x07\x1b[c"  # background colour, then primary device attributes (DA1)
_DA1_REPLY = re.compile(rb"\x1b\[\?[0-9;]*c")


def _write_all(fd: int, data: bytes, deadline: float) -> int:
    """Write `data` to a non-blocking `fd` and return how much of it went out,
    giving up at `deadline`. A short write would cut the query off mid-escape,
    and a terminal left parsing an unterminated OSC swallows what we print
    next -- the banner, and part of the shell's following prompt."""
    import select

    sent = 0
    while sent < len(data):
        try:
            sent += os.write(fd, data[sent:])
        except BlockingIOError:
            pass
        if sent < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                break
    return sent


def _drain(fd: int, quiet: float, limit: float) -> None:
    """Read and discard whatever the terminal is still sending, until it has
    been quiet for `quiet` seconds or `limit` seconds have passed in all."""
    import select

    deadline = time.monotonic() + limit
    while (remaining := deadline - time.monotonic()) > 0:
        if not select.select([fd], [], [], min(quiet, remaining))[0]:
            return
        if not os.read(fd, 64):
            return  # EOF: the terminal is gone, nothing more is coming


def query_terminal_background(tty_path: str = "/dev/tty", timeout: float = QUERY_TIMEOUT_SECONDS) -> str | None:
    """Ask the terminal for its background colour and return its raw reply, or
    None if there is no terminal to ask, it stays silent for `timeout`, input
    is already waiting to be read, or the platform cannot do it.

    The colour query is followed by a DA1 query. Terminals answer in order and
    nearly all answer DA1, so once that reply is in, the colour reply is either
    already in or never coming: reading can stop there without guessing at a
    short timeout, and nothing is left to be echoed or read by the shell. Bytes
    are read one at a time so that keys typed after the reply stay queued. Keys
    typed while the query is in flight are still read; typed-ahead input that
    is already waiting makes us skip the query instead.

    The terminal's mode is always restored afterwards. If it never finished
    answering, what it sends in the next `DRAIN_QUIET_SECONDS` is read and
    thrown away, so the near miss -- an answer landing just after we gave up --
    does not reach the shell as garbage at its next prompt. That is a sweep,
    not a guarantee: an answer slower than that still escapes, and a link that
    slow is better served by a longer `QUERY_TIMEOUT_SECONDS`, which waits the
    same time but keeps the answer. The sweep costs a key typed during it, and
    costs a terminal that never answers one `DRAIN_QUIET_SECONDS` on top of the
    timeout.
    """
    try:
        import select
        import termios
        import tty
    except ImportError:  # Windows
        return None

    try:
        # O_NONBLOCK: opening a hung-up terminal must fail or return, never wait for carrier.
        fd = os.open(tty_path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return None
    late_reply_possible = False
    try:
        try:
            if os.tcgetpgrp(fd) != os.getpgrp():
                return None  # a background job must not touch the terminal's mode
        except OSError:
            pass
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd, termios.TCSANOW)
            if select.select([fd], [], [], 0)[0]:
                return None  # the user typed ahead; do not read their keys
            deadline = time.monotonic() + timeout
            sent = _write_all(fd, _QUERY, deadline)
            late_reply_possible = sent > 0
            if sent < len(_QUERY):
                return None  # a half-written query is one the terminal cannot answer
            reply = b""
            while _DA1_REPLY.search(reply) is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                    break
                chunk = os.read(fd, 1)
                if not chunk:
                    break
                reply += chunk
            else:
                late_reply_possible = False  # the DA1 reply closes the exchange
            return reply.decode("ascii", "replace") or None
        finally:
            try:
                if late_reply_possible:
                    # Still raw, so a read takes whatever is there at once.
                    _drain(fd, DRAIN_QUIET_SECONDS, DRAIN_LIMIT_SECONDS)
            finally:
                # Restoring comes last and unconditionally: a drain that raises
                # (EIO, when the terminal hung up mid-sweep) must not leave the
                # device raw, with neither echo nor line editing.
                #
                # Not TCSAFLUSH: that waits for the query to drain, which never
                # happens if the terminal is not reading.
                termios.tcsetattr(fd, termios.TCSANOW, saved)
                if late_reply_possible:
                    termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        return None
    finally:
        os.close(fd)


def detect_background(
    env: Mapping[str, str] | None = None,
    query: Callable[[], str | None] = query_terminal_background,
) -> Background:
    """`"light"` or `"dark"`: what the terminal reports, else `COLORFGBG`, else dark."""
    reply = query()
    colour = parse_osc11(reply) if reply else None
    if colour is not None:
        r, g, b = colour
        return "dark" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "light"
    return parse_colorfgbg((os.environ if env is None else env).get("COLORFGBG")) or "dark"


def show_banner(
    console: Console | None = None,
    *,
    version: str,
    background: Background | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    art: tuple[str, ...] = LOGO,
) -> None:
    """Play the banner on `console` (stderr by default) and leave the still
    header behind. Prints nothing unless the console is a capable terminal at
    least as large as the art; under `NO_COLOR` it prints the still header
    once, uncoloured, without animating -- and without asking the terminal
    anything, since no colour of ours survives. A terminal that goes away
    mid-draw is not an error: this is decoration, never a reason to fail the
    command it introduces."""
    from rich.console import Console
    from rich.live import Live

    console = console if console is not None else Console(stderr=True)
    tagline = f"v{version} · code search for agents"
    width = max(map(len, art))
    if (
        not console.is_terminal
        or console.is_dumb_terminal
        or console.options.ascii_only
        or console.width < width
        or console.height < len(art) + 3
    ):
        return

    try:
        console.print()
        if console.no_color:
            # Every colour is stripped on the way out, so which palette this is
            # cannot show: do not ask the terminal about a background that
            # nothing will use.
            console.print(render_frame(0.0, PALETTES["dark"], tagline=tagline, final=True, art=art))
        else:
            palette = PALETTES[background if background is not None else detect_background()]
            with Live(
                render_frame(0.0, palette, tagline=tagline, art=art),
                console=console,
                auto_refresh=False,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                start = clock()
                while (elapsed := clock() - start) < DURATION_SECONDS:
                    live.update(render_frame(elapsed, palette, tagline=tagline, art=art), refresh=True)
                    sleep(1 / FRAMES_PER_SECOND)
                live.update(render_frame(DURATION_SECONDS, palette, tagline=tagline, final=True, art=art), refresh=True)
        console.print()
    except OSError:
        return  # the terminal went away mid-draw; decoration never fails a command
