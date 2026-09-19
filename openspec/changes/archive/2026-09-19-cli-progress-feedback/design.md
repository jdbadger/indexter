## Context

`indexter init` on this repository (219 files, 4,863 nodes) takes 19 seconds. Measured by phase:

| Phase | Time | Share |
|---|---:|---:|
| import + argv + settings + open db | 0.41 s | 2% |
| walk · read · hash · parse · compose · write | 1.49 s | 8% |
| graph resolution | 0.14 s | 1% |
| **embedding backlog** | **17.28 s** | **91%** |
| — of which: model load | ~1.5–4.7 s | |
| — of which: 152 batches × 32 texts | ~12.6 s | |

Three facts from this shape the whole design.

**The dominant phase has a known total.** `_embedding_backlog` (`sync.py:336`) selects every
vector-less node before the loop starts, so the count is known upfront. At ~83 ms per batch it
repaints ~12×/s — a genuine percentage bar with a real ETA, not a decorative one. The walk phase is
the opposite: `Walker.walk()` is a generator with no count until exhausted, and pre-counting would
double the cheapest phase to decorate it.

**One caller must never print.** `sync_repo` has three callers. `search/hybrid.py:419` and
`search/neighbors.py:455` run inside the MCP server, where **stdout is the JSON-RPC transport**. A
stray progress byte corrupts the protocol.

**The model load is three different silences, not one.** Measured:

| Case | Duration | Current output |
|---|---|---|
| Cold (nothing cached) | 6.2 s on fibre → minutes on slow links | **nothing at all** |
| Warm, online (today's path) | ~1.5 s construct | auth warning + misleading bar |
| Warm, `local_files_only=True` | ~0.2 s construct | — |

The cold case is silent because `sentence_transformers/util/file_io.py:186` hardcodes
`"tqdm_class": disabled_tqdm`, suppressing the download bar. The `Loading weights: 100%` bar users
do see is weight *initialization*, which completes instantly and tells them nothing about the
transfer they are actually waiting on.

## Goals / Non-Goals

**Goals:**
- The user knows, within ~300 ms of any wait beginning, what indexter is doing and that it is alive.
- A cold first run explains that it is a one-time download and shows elapsed time.
- stdout stays byte-identical, so agents and existing tests are unaffected.
- The MCP path is silent by construction, not by convention.
- No new dependencies.

**Non-Goals:**
- A determinate progress bar for the model download (see Decision 4 — not achievable).
- Progress for `list`, `remove`, `skill`, or `mcp`; none of them have a perceptible wait.
- Restructuring the indexing pipeline, parallelising embedding, or changing what is indexed.
- Progress inside the MCP server, including MCP's own progress-notification protocol.

## Decisions

### Decision 1: Narration on stderr, results on stdout

Progress renders to stderr; the existing summary stays on stdout, unchanged.

This gives `indexter init . 2>/dev/null` clean machine-readable output for an agent, while
`indexter init . > out.txt` still shows a human the progress. It also means the existing stdout
assertions in `test_cli.py` remain valid without modification, which is a strong signal the split is
cutting along the real seam.

*Alternative considered:* everything on stdout with `--quiet`. Rejected — it makes silence opt-in for
every non-interactive caller, and indexter is explicitly a tool that agents invoke.

### Decision 2: An injected observer, defaulting to silence

A small `Progress` protocol is threaded `index_repository → sync_repo → _embedding_backlog`, as an
optional parameter defaulting to a no-op implementation.

Silence is therefore the default for every caller that does not opt in, which makes the MCP
constraint structural rather than a rule to remember. It also follows a pattern the codebase already
uses: `FakeEmbedder` exists so tests can assert "the model was never loaded", and a recording
`FakeProgress` lets a test assert "MCP search emitted zero progress events".

*Alternative considered:* making `sync_repo` a generator of events. Rejected — it changes the return
shape for all three callers to serve one of them.

*Alternative considered:* `typer.echo` inside `sync_repo` guarded by a flag. Rejected — it puts a
protocol-corrupting bug one forgotten argument away.

### Decision 3: Probe the cache locally before deciding anything

`huggingface_hub.try_to_load_from_cache()` answers "is this model on disk?" in **0.016 ms with no
network**. It branches the load into cold and warm before either is committed to.

This is the same idiom `_load_tokenizer` already uses at `embed.py:98` (`local_files_only=True`, then
fall back to download). The model path simply never got the same treatment.

### Decision 4: No byte total on the cold path — elapsed time only

A determinate download bar is not achievable, and this is worth stating rather than rediscovering:

- **Polling the cache directory is wrong.** Measured, it reports `0.1 MB → 181.8 MB` in a single
  250 ms step against an 87 MB real payload, because xet blobs are sparse and snapshot symlinks
  double-count.
- **Hugging Face themselves decline it.** From `_xet_progress_reporting.py`: *"Transfer byte count is
  hard to predict (dedup/compression), so we omit a total and show bytes only."*
- **Pre-fetching to own the bar is a trap.** A naive `snapshot_download` dry-run for the default
  model reports **30 files, 976.9 MB** — TensorFlow, Rust, ONNX and quantized variants — against the
  **87 MB** `sentence-transformers` actually fetches. Replicating its `modules.json`-driven selection
  would couple us to its internals for a cosmetic gain.

So the cold path shows a spinner, the model name, and **elapsed time**. An elapsed counter alone
converts "is this frozen?" into "it has been working for 42 seconds", which is the substance of the
reassurance. A size estimate is also model-dependent and would be wrong for any non-default
`embedding_model`.

### Decision 5: `local_files_only=True` once the cache hit is confirmed

`SentenceTransformer` accepts `local_files_only`. Measured across runs:

```
local_files_only=False:  construct 1.44 s / 1.52 s
local_files_only=True:   construct 0.25 s / 0.13 s
```

~1.3 s saved on every `init`, `reindex`, and first search. Today that time is spent asking Hugging
Face about a file already visible on disk.

This is a real semantic shift — a cached model is never silently refreshed — and it is accepted
deliberately. `embedding_model` is pinned in configuration, and changing it already forces a full
re-index because `compute_fingerprint` (`sync.py:286`) includes it. There is no supported workflow in
which a silent upstream model change should alter an existing index.

### Decision 6: Suppress the borrowed noise, after import

```python
huggingface_hub.utils.logging.set_verbosity_error()   # the auth warning
huggingface_hub.utils.disable_progress_bars()         # the weight-loading bar
```

Both verified. **Both must run after `huggingface_hub` is imported** — HF reconfigures its root
logger at import time, so levels set beforehand are silently clobbered. Given the deliberate lazy
imports in `embed.py`, this ordering is a live footgun and belongs immediately after the lazy import
inside `_load_model`, not at module scope.

Note `disable_progress_bars()` is programmatic and process-local; the `HF_HUB_DISABLE_PROGRESS_BARS`
environment variable is avoided because it would also mute the download bar for any other consumer in
the process.

### Decision 7: Lazy painting at a ~300 ms threshold

A phase paints nothing until it has been running longer than the threshold. This yields the
`init`/`reindex` asymmetry for free: a no-op reindex does no embedding work (`_embedding_backlog`
returns before touching the embedder at `sync.py:339`), so it finishes under the threshold and prints
only its summary.

It also composes with Decision 5: `local_files_only` drops the warm load to ~0.2 s, which falls
*under* the threshold. The model spinner therefore appears only when the wait is genuinely real —
a cold download, or a slow disk — with no flag deciding when to animate.

### Decision 8: Rendering is `rich`, already present

`rich` 15.0.0 is already a hard transitive dependency via **both** `typer` and `fastmcp-slim`. It
also handles `NO_COLOR`, dumb terminals, and TTY detection for free. `Console(stderr=True)`
auto-detects and degrades.

Render shape (Decision: the "measured" level — phase lines that resolve, only the active one
animates):

```
⠹ Indexing files · 143 parsed
✓ Indexed 219 files · 4,863 nodes · 6,994 refs
✓ Resolved graph · 10,014 edges
⠸ Embedding ━━━━━━━━━━━━━━╸──────────  61% · 2,976/4,863 · 0:08
✓ Embedded 4,863 nodes
```

Rejected the fuller boxed-panel treatment: delightful once, irritating the fortieth time, and it does
not survive scrollback or narrow terminals.

### Decision 9: `--quiet` and `--progress` override detection

`--quiet` forces narration off; `--progress` forces it on even when stderr is not a TTY (useful for
CI logs that are recorded but not interactive). Default is auto-detection. The two are mutually
exclusive and conflict is an error rather than a silent precedence rule.

## Risks / Trade-offs

**[A cached model never auto-updates under `local_files_only`]** → Accepted per Decision 5;
`embedding_model` is pinned and a change to it already forces a re-index. `reindex --full` after
clearing the HF cache remains the escape hatch.

**[The cache probe and the actual load could disagree]** → The probe checks a marker file while the
load needs the full snapshot, so a partially-populated cache could probe warm and then fail under
`local_files_only`. Mitigation: catch the local-entry-not-found failure and retry once without
`local_files_only`, reporting it as a download. This keeps a corrupt or truncated cache recoverable
rather than permanently broken.

**[Progress output leaks into MCP stdio]** → Mitigated structurally by Decision 2 (silence is the
default, opt-in at the CLI). Backed by a test asserting the MCP search path emits zero events, and by
narration going to stderr even when enabled.

**[Rendering slows the hot loop]** → At ~83 ms per embedding batch a repaint per batch is ~12 Hz,
well within what `rich` handles. The walk phase can tick per file (219 files in 1.5 s ≈ 150 Hz), so
its counter should be throttled to a fixed refresh rate rather than repainting per event.

**[Suppressing HF logging hides a real failure]** → `set_verbosity_error()` keeps errors; only
warnings and below are muted. `ModelAcquisitionError` already wraps and re-raises acquisition
failures with actionable text, so the failure path is unaffected.

**[Snapshot tests capture ANSI escapes]** → Narration goes to stderr and `CliRunner` can separate the
streams; assertions stay on stdout. Any test that does exercise narration should force a
non-interactive console so output is deterministic.

## Migration Plan

Additive and independently landable, in the order the measurements justify:

1. **Embedding quiet + `local_files_only` + cache probe.** Standalone value — a ~1.3 s speedup and the
   removal of the borrowed noise — with no rendering work and no CLI surface change.
2. **The `Progress` protocol and its null implementation**, threaded through `sync_repo`. No
   behavior change; callers unchanged.
3. **The `rich` renderer and the CLI wiring**, including `--quiet` / `--progress`.

Rollback is per-step; step 1 is the only one with an observable behavior change and it is reversible
by dropping the `local_files_only` argument.

## Open Questions

- Should the cold-download path surface an extra reassurance line after a longer interval (~30 s) on
  very slow links, or is a running elapsed counter sufficient on its own?
- Should `--progress` also force color, or only force painting? Forcing painting while respecting
  `NO_COLOR` is the more conservative reading and is the assumed default.
