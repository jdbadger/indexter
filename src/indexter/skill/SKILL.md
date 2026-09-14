---
name: indexter
description: Search this codebase by meaning, not just keywords, and walk its call/import/inheritance graph from a search result. Use before grepping for a symbol name you're only guessing at, or when tracing what calls, imports, or inherits from something.
---

# indexter

indexter indexes this repository and exposes two tools, `search` and
`neighbors` (in Claude Code, `mcp__indexter__search` and
`mcp__indexter__neighbors`). Every call syncs the index against the files on
disk first, so results always reflect the current state of the repository --
there's no separate reindex step to remember.

## `search`: find code by what it does

Reach for `search` before grepping when you don't already know the exact file
or symbol name. It ranks hybrid (semantic + keyword) matches and returns real
code for each hit, plus a `related` line naming its closest callers, callees
and containing scope.

Parameters: `query` (required), `repo`, `kind`, `language`, `path`, `limit`.

- Phrase `query` as the *behavior*, in plain words: "where retries are
  decided for a failed request", not `retry`. A symbol name you do know is
  also a fine query -- keyword and semantic matches are combined.
- `kind` and `language` narrow results to one or more values (e.g.
  `kind="function"`, `language=["python", "rust"]`) -- use them once you know
  the shape of what you want and the first search is too broad.
- `path` restricts results to files under a prefix.
- `limit` caps how many results come back.

Each result carries a node ID and a `path:start-end` span. Read the span with
your own file-reading tool instead of searching again -- the snippet already
tells you whether it's worth reading further, and the ID is what `neighbors`
needs next.

## `neighbors`: walk the call/import/inheritance graph

Once you have a node ID from `search` (or from a previous `neighbors` call),
use `neighbors` to answer "what calls this", "what does this import", "what
inherits from this", or "what does this contain" -- directly, instead of
re-running `search` and hoping the graph shows up in `related`.

Parameters: `node_id` (required), `repo`, `direction` (`in` | `out` | `both`,
default `both`), `edges` (any of `calls`, `imports`, `inherits`, `contains`;
default all four), `depth` (1-3, default 1), `limit` (default 20).

- `direction="in"` asks "who calls/imports/inherits this"; `direction="out"`
  asks "what does this call/import/inherit". Narrow `edges` to one relation
  when the default mix of all four is noisy -- e.g. `edges=["calls"],
  direction="in"` for "who calls this function".
- Raise `depth` past 1 only when one hop isn't enough (e.g. "what eventually
  calls into this, a couple of levels up"). Each extra hop can pull in many
  more nodes; anything past `limit` is counted as omitted rather than
  dropped silently.
- Neighbors reached only through a highly-connected node, or only through an
  ambiguous call resolution, are listed but not walked further -- that's
  expected, not a bug.

## When a call fails

- **"No index found for ..."** means indexter hasn't indexed this repository,
  or `repo` points at the wrong one. Don't run `indexter init` yourself -- it
  downloads an embedding model on first use and can take a while. Tell the
  user what happened and let them decide.
- **A stale node ID** ("no node with ID ...") means the code moved since that
  ID was returned. Search again rather than guessing a corrected ID; the
  error lists current IDs with the same name in that file when it can find
  one.
- Any other error names the bad parameter and its valid values -- fix the
  call and retry.
