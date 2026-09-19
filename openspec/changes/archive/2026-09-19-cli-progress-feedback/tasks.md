## 1. Embedding: quiet the borrowed noise and skip the network on a cache hit

Standalone value — a ~1.3 s speedup per run and the removal of foreign console output — with no
rendering work and no CLI surface change. Landable and releasable on its own.

- [x] 1.1 Add a `probe_model_cache(model_name)` helper in `index/embed.py` that uses `huggingface_hub.try_to_load_from_cache` against a marker file and returns whether the model is cached, performing no network access
- [x] 1.2 Add a `_quiet_backends()` helper that calls `huggingface_hub.utils.logging.set_verbosity_error()` and `huggingface_hub.utils.disable_progress_bars()`, and call it immediately after the lazy `huggingface_hub`/`sentence_transformers` imports inside `_load_model` — not at module scope, since importing the hub reconfigures its logging
- [x] 1.3 Pass `local_files_only=True` to `SentenceTransformer(...)` in `SentenceTransformerEmbedder._load_model` when the probe reports a cache hit
- [x] 1.4 Catch the local-entry-not-found failure from that construction and retry once without `local_files_only`, reclassifying the load as a first-time acquisition, so an incomplete cache recovers
- [x] 1.5 Apply the same probe and suppression to `FastEmbedEmbedder._load_model`, or record in the code why the fastembed backend cannot use them
- [x] 1.6 Test: a cached model constructs with local files only and makes no network request
- [x] 1.7 Test: an incomplete cache that probes as a hit falls back, is reclassified as an acquisition, and succeeds
- [x] 1.8 Test: no hub authentication warning and no weight-loading bar appear on either stream during a load
- [x] 1.9 Test: `ModelAcquisitionError` still names the model and states that network access is needed once, with suppression active
- [x] 1.10 Verify the existing lazy-load guarantees still hold: construction loads nothing, tokenizer loads without the model, model loads once

## 2. The progress protocol and its null implementation

No behavior change; every existing caller keeps working untouched.

- [x] 2.1 Create `src/indexter/progress.py` with a `Progress` protocol covering phase start (with an optional known total), advance, and phase completion, plus a model-load classification event
- [x] 2.2 Implement `NullProgress` as the default: it performs no rendering and no terminal detection
- [x] 2.3 Implement `RecordingProgress` as a test double that captures the event sequence, in the spirit of `FakeEmbedder`
- [x] 2.4 Test: `NullProgress` writes nothing to either stream and does no terminal detection

## 3. Thread the observer through the indexing pipeline

- [x] 3.1 Add an optional `progress: Progress | None = None` parameter to `index_repository`, `sync_repo`, and `_embedding_backlog` in `index/sync.py`, defaulting to the non-reporting observer
- [x] 3.2 Report the file-indexing phase from `sync_repo`'s walk loop with no known total, advancing per file
- [x] 3.3 Report the resolution phase around the `resolve_repo` call, and skip reporting entirely when resolution is not due
- [x] 3.4 Report the embedding phase from `_embedding_backlog`: emit the total row count before the first batch, then advance after each batch
- [x] 3.5 Ensure `_embedding_backlog`'s early return for an empty backlog reports no phase and still loads neither model nor tokenizer
- [x] 3.6 Give the embedder access to the observer so the model-load classification from task 1.1 is reported before the load begins
- [x] 3.7 Confirm `search/hybrid.py:419` and `search/neighbors.py:455` pass no observer and remain silent
- [x] 3.8 Test: a sync with a `RecordingProgress` reports the indexing, resolution, and embedding phases in order
- [x] 3.9 Test: a sync with no resolution due reports no resolution phase
- [x] 3.10 Test: the embedding total is reported before the first batch and the advancing count ends equal to it
- [x] 3.11 Test: the same repository synced with and without an observer produces equivalent reports and database contents
- [x] 3.12 Test: a search-triggered and a neighbors-triggered sync each emit zero progress events

## 4. The terminal renderer

- [x] 4.1 Implement `ConsoleProgress` in `progress.py` using `rich` with `Console(stderr=True)` — already a hard transitive dependency via both `typer` and `fastmcp-slim`, so nothing is added to `pyproject.toml`
- [x] 4.2 Implement lazy painting: a phase paints nothing until it has run past a ~300 ms threshold, and a phase completing sooner produces no output
- [x] 4.3 Throttle repaints to a bounded refresh rate so the per-file walk tick (~150 Hz) does not drive rendering
- [x] 4.4 Render a determinate phase as a proportional bar with completed and total counts; render an indeterminate phase as a running count with no percentage or bar
- [x] 4.5 Animate at most one phase at a time and resolve each finished phase to a static completion line that stays in scrollback
- [x] 4.6 Render the model-load classification: a first-time acquisition names the model, says the acquisition happens once, and shows elapsed time with no byte total, percentage, or estimate; a cached load says it is loading and names the model
- [x] 4.7 Honour the environment's no-colour request independently of whether narration is enabled
- [x] 4.8 Test: a phase completing under the threshold paints nothing
- [x] 4.9 Test: a phase exceeding the threshold paints an active line and then a completion line
- [x] 4.10 Test: acquisition narration contains no byte total, percentage, or time-remaining estimate
- [x] 4.11 Test: narration renders without colour escapes when the environment requests it

## 5. Wire up the CLI

- [x] 5.1 Add mutually exclusive `--quiet` and `--progress` options to `init` and `reindex` in `cli.py`, failing non-zero with a message naming both when they conflict
- [x] 5.2 Select `ConsoleProgress` when narration is enabled — by stderr being a TTY, or forced by `--progress` — and `NullProgress` otherwise, then pass it to `index_repository`
- [x] 5.3 Keep `_render_index_result` and the per-file error listing writing to stdout exactly as they do today, changing neither content nor ordering
- [x] 5.4 Test: stdout is byte-identical between a run with narration enabled and one with it disabled
- [x] 5.5 Test: no phase label, spinner frame, bar, or completion marker ever appears on stdout
- [x] 5.6 Test: non-interactive stderr with no flag produces no narration
- [x] 5.7 Test: `--quiet` on an interactive terminal silences narration and still writes the summary
- [x] 5.8 Test: `--progress` with redirected stderr writes narration to that destination
- [x] 5.9 Test: `--quiet --progress` exits non-zero naming both flags, with no indexing performed
- [x] 5.10 Test: a no-op `reindex` with narration enabled paints no phase and still writes its summary
- [x] 5.11 Test: `reindex --full` with narration enabled narrates the indexing and embedding passes as `init` does
- [x] 5.12 Verify the existing `test_cli.py` stdout assertions pass unmodified

## 6. Verify against the real thing

- [x] 6.1 Run `indexter init` on this repository with a warm cache and confirm the narration matches the design's render shape and that the run is ~1.3 s faster than before
- [x] 6.2 Run `indexter init` against an empty `HF_HOME` and confirm the cold acquisition is narrated with elapsed time rather than being silent
- [x] 6.3 Run `indexter reindex` with no changes and confirm only the summary appears
- [x] 6.4 Run the MCP server, issue a search that triggers a real sync, and confirm the JSON-RPC stream is uncorrupted
- [x] 6.5 Run `indexter init . 2>/dev/null` and `indexter init . | cat` and confirm both produce clean, parseable stdout
- [x] 6.6 Update `README.md` with the narration behavior and the two new flags
- [x] 6.7 Add a CHANGELOG entry covering the narration, the borrowed-noise suppression, and the cached-load speedup
