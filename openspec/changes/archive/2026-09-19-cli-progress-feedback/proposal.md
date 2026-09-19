## Why

The CLI is the human API for indexter, and it says nothing while it works. A cold `indexter init` on
a 219-file repository takes 19 seconds, of which 91% is the embedding backlog, and the only output
before the final summary is leakage from other libraries: a Hugging Face "unauthenticated requests"
warning that reads like an error, and a `Loading weights` bar that measures weight initialization
rather than anything the user is waiting on. Indexter's own voice is silent until it is finished.

Worse, a first-ever run downloads an 87 MB model with the progress bar explicitly disabled by
`sentence-transformers` (`tqdm_class: disabled_tqdm`), so the terminal is frozen for the entire
transfer — seconds on a fast link, minutes on a slow one, on the user's very first contact with the
tool. That silence is invisible to anyone with a warm cache, which is why it has survived.

## What Changes

- **New progress narration, on stderr.** `init` and `reindex` narrate their phases to stderr while
  results stay on stdout, so `2>/dev/null` yields clean machine-readable output and `> file` still
  shows progress on screen. Agents and CI are unaffected by default because narration degrades to
  silence when stderr is not a TTY.
- **Phase lines that resolve.** Each phase prints one line that animates while active and resolves
  to a `✓` line that stays in scrollback. Only the active line animates. The embedding backlog
  renders a true percentage bar — its total is known upfront from the backlog query — while the walk
  phase, whose total is unknowable without a second pass, shows a running counter.
- **Lazy painting.** Nothing is painted until a phase has run longer than a short threshold
  (~300 ms), so fast paths stay quiet. This is what makes `reindex` naturally less chatty than
  `init` without separate code paths: a no-op reindex does no embedding work, so it paints nothing
  but its summary.
- **The embedding model load stops being silent.** A local cache probe distinguishes a cold first
  run from a warm load before committing to either. A cold run announces that it is downloading the
  model for the first time and shows elapsed time; a warm load announces loading. Neither promises a
  byte total — a determinate download bar is not achievable, because Hugging Face's xet transfers
  dedupe and compress, and `huggingface_hub` itself omits the total for this reason.
- **Borrowed noise is suppressed.** The Hugging Face auth warning and the misleading weight-loading
  bar are silenced, so the only output during an index is indexter's.
- **A confirmed cache hit skips the network.** When the local probe proves the model is on disk, the
  model is constructed with `local_files_only=True`, removing a ~1.3 s round-trip that currently
  runs on every `init`, `reindex`, and first search.
- **`--quiet` / `--progress`.** An explicit flag forces narration off or on, overriding TTY
  detection.

No breaking changes: stdout content is unchanged, and progress is opt-in at the call site, so the
MCP server's sync-on-search path — where stdout is the JSON-RPC transport — stays silent by
construction rather than by remembering.

## Capabilities

### New Capabilities
- `cli-progress`: Progress narration for long-running CLI commands — the stdout/stderr split, the
  phase lifecycle and its rendering, lazy painting, TTY and `NO_COLOR` degradation, and the
  `--quiet`/`--progress` overrides.

### Modified Capabilities
- `repo-management-cli`: `init` and `reindex` gain progress narration on stderr while their existing
  stdout summaries stay byte-identical, and both accept the narration-control flags.
- `embedding`: Model loading gains a local cache probe that reports whether a load is a first-time
  download or a cached load, suppresses the backend libraries' own console output, and uses
  `local_files_only` once the cache is confirmed. The lazy-load guarantee is unchanged.
- `index-sync`: `sync_repo` accepts an optional progress observer, defaulting to one that reports
  nothing, and reports phase transitions and embedding-batch completions through it.

## Impact

- **Code**: `src/indexter/cli.py` (both indexing commands and a new rendering layer),
  `src/indexter/index/sync.py` (thread an observer through `index_repository`, `sync_repo`, and
  `_embedding_backlog`), `src/indexter/index/embed.py` (cache probe, `local_files_only`, noise
  suppression, load reporting).
- **Callers that must stay silent**: `src/indexter/search/hybrid.py:419` and
  `src/indexter/search/neighbors.py:455` call `sync_repo` on every search from inside the MCP
  server. They pass no observer and must emit nothing.
- **Dependencies**: none added. `rich` 15.0.0 is already a hard transitive dependency via both
  `typer` and `fastmcp-slim`.
- **Behavior**: `local_files_only` on a cache hit means a cached model is never silently updated
  mid-life. This is considered correct — `embedding_model` is pinned in configuration, and changing
  it already forces a full re-index through the index fingerprint.
- **Tests**: `src/indexter/tests/test_cli.py` asserts CLI output; stdout assertions should remain
  valid precisely because narration goes to stderr.
