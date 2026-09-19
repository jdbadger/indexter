"""Progress reporting for long-running index work.

Callers that want narration pass a `Progress`; everything else gets
`NullProgress`, so silence is the default and nothing can emit output by
omission. That matters for the MCP server, where stdout is the JSON-RPC
transport. `RecordingProgress` is the test double, in the spirit of
`FakeEmbedder`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.live import Live

Phase = Literal["model", "files", "resolve", "embed"]
LoadKind = Literal["cached", "acquisition"]


@runtime_checkable
class Progress(Protocol):
    def model_classified(self, kind: LoadKind, model_name: str) -> None:
        """The embedding model load is about to begin, and is `kind`."""

    def phase_start(self, phase: Phase, total: int | None = None) -> None:
        """`phase` began. `total` is the amount of work when known up front."""

    def advance(self, n: int = 1) -> None:
        """`n` more units of the current phase are done."""

    def phase_done(self, phase: Phase) -> None:
        """`phase` finished."""


class NullProgress:
    """Reports nothing: no rendering, no terminal detection."""

    def model_classified(self, kind: LoadKind, model_name: str) -> None:
        pass

    def phase_start(self, phase: Phase, total: int | None = None) -> None:
        pass

    def advance(self, n: int = 1) -> None:
        pass

    def phase_done(self, phase: Phase) -> None:
        pass


class RecordingProgress:
    """Test double: captures every event in `events`, in order, as tuples --
    `("model", kind, model_name)`, `("start", phase, total)`,
    `("advance", phase, n)`, `("done", phase)`.
    """

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self._active: Phase | None = None

    def model_classified(self, kind: LoadKind, model_name: str) -> None:
        self.events.append(("model", kind, model_name))

    def phase_start(self, phase: Phase, total: int | None = None) -> None:
        self._active = phase
        self.events.append(("start", phase, total))

    def advance(self, n: int = 1) -> None:
        self.events.append(("advance", self._active, n))

    def phase_done(self, phase: Phase) -> None:
        self.events.append(("done", phase))
        if self._active == phase:
            self._active = None

    def started(self) -> list[Phase]:
        return [e[1] for e in self.events if e[0] == "start"]

    def advanced(self, phase: Phase) -> int:
        return sum(e[2] for e in self.events if e[0] == "advance" and e[1] == phase)


PAINT_THRESHOLD_SECONDS = 0.3
REFRESH_HZ = 12.0
_BAR_WIDTH = 28


def _format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


class ConsoleProgress:
    """Narrates phases on a `rich` console (stderr by default).

    A phase paints nothing until it has run past `threshold` seconds, so fast
    phases leave no trace. Events only update counters; painting happens on
    `Live`'s own clock at `refresh_hz`, however fast events arrive. Only the
    active phase animates, and a finished, painted phase becomes a static
    line. On a non-terminal console nothing animates: a slow phase prints one
    line when it crosses the threshold and one when it completes.

    Use as a context manager (or call `close()`) so a phase that never
    finishes -- an error mid-run -- doesn't leave its live line behind.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        threshold: float = PAINT_THRESHOLD_SECONDS,
        refresh_hz: float = REFRESH_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from rich.console import Console
        from rich.spinner import Spinner

        self._console = console if console is not None else Console(stderr=True)
        self._threshold = threshold
        self._refresh_hz = refresh_hz
        self._clock = clock
        ascii_only = self._console.options.ascii_only
        self._check = "OK" if ascii_only else "\u2713"
        self._spinner = Spinner("line" if ascii_only else "dots", style="cyan")

        self._lock = threading.Lock()
        self._generation = 0
        self._phase: Phase | None = None
        self._total: int | None = None
        self._count = 0
        self._started_at = 0.0
        self._painted = False
        self._timer: threading.Timer | None = None
        self._live: Live | None = None
        self._kind: LoadKind | None = None
        self._model_name = ""

    def __enter__(self) -> ConsoleProgress:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def model_classified(self, kind: LoadKind, model_name: str) -> None:
        self._kind = kind
        self._model_name = model_name

    def phase_start(self, phase: Phase, total: int | None = None) -> None:
        with self._lock:
            self._end_phase()
            self._phase = phase
            self._total = total
            self._count = 0
            self._started_at = self._clock()
            self._painted = False
            self._timer = threading.Timer(self._threshold, self._paint, args=(self._generation,))
            self._timer.daemon = True
            self._timer.start()

    def advance(self, n: int = 1) -> None:
        self._count += n

    def phase_done(self, phase: Phase) -> None:
        with self._lock:
            if self._phase != phase:
                return
            painted = self._painted
            line = self._completion_line()
            self._end_phase()
            if phase == "model":
                self._kind = None
        if painted:
            self._console.print(line)

    def close(self) -> None:
        with self._lock:
            self._end_phase()

    def _end_phase(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._phase = None
        self._painted = False

    def _paint(self, generation: int) -> None:
        from rich.live import Live
        from rich.text import Text

        with self._lock:
            if generation != self._generation or self._phase is None:
                return
            self._painted = True
            if self._console.is_terminal and not self._console.is_dumb_terminal:
                self._live = Live(
                    console=self._console,
                    get_renderable=self._active_line,
                    refresh_per_second=self._refresh_hz,
                    transient=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live.start()
            else:
                suffix = f": {self._model_name}" if self._phase == "model" and self._model_name else ""
                self._console.print(Text(f"{self._heading()}{suffix}..."))

    def _heading(self) -> str:
        if self._phase == "model":
            if self._kind == "acquisition":
                return "Downloading embedding model (one-time)"
            if self._kind == "cached":
                return "Loading embedding model"
            return "Preparing embedding model"
        if self._phase == "files":
            return "Indexing files"
        if self._phase == "resolve":
            return "Resolving graph"
        return f"Embedding {self._total:,} nodes" if self._total else "Embedding"

    def _completion_line(self) -> RenderableType:
        from rich.text import Text

        if self._phase == "model":
            verb = {"cached": "Loaded", "acquisition": "Downloaded"}.get(self._kind or "", "Prepared")
            body = f"{verb} embedding model"
        elif self._phase == "files":
            body = f"Indexed {self._count:,} {'file' if self._count == 1 else 'files'}"
        elif self._phase == "resolve":
            body = "Resolved graph"
        else:
            body = f"Embedded {self._total or self._count:,} nodes"
        line = Text()
        line.append(self._check, style="green")
        line.append(f" {body}")
        return line

    def _active_line(self) -> RenderableType:
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        if self._phase is None:
            return Text("")

        elapsed = _format_elapsed(self._clock() - self._started_at)
        spinner = cast("Text", self._spinner.render(self._clock()))  # a text-frame spinner renders as Text

        if self._phase == "embed" and self._total:
            total, done = self._total, min(self._count, self._total)
            bar_width = max(8, min(_BAR_WIDTH, self._console.width - 46))
            detail = Text(f"{done * 100 // total}% \u00b7 {done:,}/{total:,} \u00b7 {elapsed}", style="dim")
            grid = Table.grid(padding=(0, 1))
            for _ in range(3):
                grid.add_column(no_wrap=True)
            grid.add_column(no_wrap=True, overflow="ellipsis")
            grid.add_row(spinner, "Embedding", ProgressBar(total=total, completed=done, width=bar_width), detail)
            return grid

        if self._phase == "files":
            detail = f"\u00b7 {self._count:,} {'file' if self._count == 1 else 'files'}"
        elif self._phase == "model":
            # The model name goes last: on a narrow terminal it is the part to lose.
            name = f" \u00b7 {self._model_name}" if self._model_name else ""
            detail = f"\u00b7 {elapsed}{name}"
        else:
            detail = ""
        return Text.assemble(
            spinner, " ", self._heading(), " " if detail else "", (detail, "dim"), no_wrap=True, overflow="ellipsis"
        )
