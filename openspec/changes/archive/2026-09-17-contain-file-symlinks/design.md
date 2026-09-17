## Context

A security review of v0.2.0 found that `Walker._walk_dir` (`src/indexter/walk.py`) checks containment only for symlinked directories:

```text
is_dir?  ── yes ── ignored("rel/")? ── symlink && !within_repo? ── descend
   │
   no ── is_file()  (follows link) ── _consider_file ── entry.stat() (follows link)
                                                           │
sync_repo ── read_file(repo, rel) ── (repo / rel).read_text()   (follows link)
search    ── read_snippets ── read_file(...)                    (follows link again)
```

There are two gaps:

1. **Escaping file symlinks.** A file symlink whose target is outside the repository is indexed. Its content reaches `nodes_fts.body`, `nodes.embed_text` and MCP `search` snippets.
2. **Symlinks to ignored paths.** The directory containment check asks only whether the target is under the repo root. `link -> .git` is inside the repo, so the walker descends it, even though `.git/` is always ignored. The same happens for file symlinks to gitignored secrets (`notes.md -> .env`). Ignore rules are matched against the link's path, never the target's.

The threat is a repository the user has indexed that contains attacker-controlled commits, combined with an LLM client that is open to prompt injection from that repository's content. Search re-syncs before answering, so a `git pull` is enough to trigger the attack.

## Goals / Non-Goals

**Goals:**

- Content reachable only through a symlink never enters the index or a snippet. This covers content outside the repository and content excluded by the walk's ignore rules.
- Keep indexing in-repo, non-ignored symlinks, so behaviour stays the same for repos that use them legitimately (for example `CLAUDE.md -> AGENTS.md`, or shared fixtures).
- Rows indexed through a bad link before the fix are removed automatically on the next sync.
- A second check at read time, so the snippet path, which trusts paths stored in the database, cannot be turned into an arbitrary read.

**Non-Goals:**

- **Repo-layer config files and `.gitignore` read through symlinks.**
  - `indexter.toml`, `pyproject.toml` and `.gitignore` symlinked out of the repo are still read. Their contents never reach search output.
  - `ConfigError` messages name keys and sources, not values.
  - Ignore patterns only narrow the walk.
  - Revisit this if config values ever appear in tool output.
- **Hard links.** A hard link is the same inode, so it can't be told apart from a regular file without heuristics, and git does not create hard links on checkout.
- **Race-free (`O_NOFOLLOW`/`openat`) traversal against a local attacker** who swaps paths between check and read. That attacker already has local write access.
- **Nested `.gitignore` files.** The walker only honours the root `.gitignore` today, and this change matches that.
- **Windows.** Git's default `core.symlinks=false` checks symlinks out as plain text files, so nothing is followed there.

## Decisions

### 1. A symlink is admitted only if its target would be walked by its real path

For every entry where `entry.is_symlink()` is true, resolve it with `Path.resolve(strict=True)`. Admit it only if both of these hold:

- the resolved path is under `repo_resolved`
- the resolved path, taken relative to `repo_resolved` as a POSIX path, does not match the ignore matcher (with a trailing `/` for directories)

The same predicate replaces `_symlink_dir_within_repo`, and is applied to files and directories alike.

The check runs **before** `is_file()` and `_consider_file`, so an escaping target is never stat'd. A broken link fails `resolve(strict=True)` and is skipped, which matches the current behaviour.

Why this rule over plain containment: the invariant "a link can't reveal anything the walk wouldn't index by its real path" covers `.git/`, `.env` and outside targets with one rule. Plain containment closes the reviewer's finding but leaves the `.env` variant open, and that variant needs no path guessing.

`GitIgnoreSpec.match_file` already matches descendants of ignored directory patterns (`secrets/` matches `secrets/key.txt`, `.git/` matches `.git/config`). This was verified against the installed pathspec, so there is no need to check ancestors separately.

**Alternatives considered:**

- **Never follow file symlinks.** This is simplest, but it silently drops legitimate in-repo links, and it treats files differently from directories, which the existing spec scenario "Symlink within the repository" keeps.
- **Resolve with `os.path.realpath` without `strict`.** This would make broken links look contained. `strict=True` makes them fail closed.

### 2. `read_file` checks containment, not ignore rules

`read_file(repo_path, relpath)` resolves `Path(repo_path).resolve()` and `(Path(repo_path) / relpath).resolve(strict=True)`. It returns `None` without opening the file unless the second is under the first. Its signature stays the same.

`read_file` takes no settings, and it is called from `search/results.py`, where the ignore matcher isn't available. Adding one would widen the API for a second layer of defence. The walker is the main control. `read_file` guarantees that a path from the database can never read outside the repo, even if a link was swapped in after indexing or a bad `relpath` got stored.

A contained `..` path, such as `a/../b.py`, still resolves inside the repo and stays allowed.

**Why `None` and not an exception:** callers already treat `None` as unreadable. `sync_repo` records it as an unreadable file, and `read_snippets` renders a `None` snippet. This keeps call sites unchanged.

**Considered:** a separate `resolve_within_repo` helper shared by the walker and `read_file`. Adopt it if the predicate code is duplicated. It is an implementation detail.

### 3. Cleaning up existing indexes relies on normal removal handling

A link the walker now rejects is missing from `seen_paths` on the next sync, so `sync_repo` calls `remove_file`. That deletes the file's nodes and, through FK cascade, its FTS and vector rows. Search syncs before querying, so cleanup happens on the user's first search after upgrading.

**Rejected alternative:** bumping the settings fingerprint to force a reparse. A forced reparse does not remove anything the walker still yields, so it adds cost and closes nothing.

### 4. Resolving the repo root once per walk, and once per read

`walk()` already computes `repo_resolved`. `read_file` resolves the repo root on each call. That is one extra `realpath` per changed file during sync and per distinct file during snippet rendering, which is negligible next to parsing and embedding.

## Risks / Trade-offs

- **[Risk] Repos that deliberately symlink files from outside the checkout stop indexing them.** → Mitigation: this matches the existing directory behaviour and the spec's intent. Note it in the CHANGELOG `### Security` entry, and log a debug line per skipped link (`Skipping symlink outside repository or to ignored path`).
- **[Risk] The repo root itself is reached through a symlink** (for example macOS `/tmp` → `/private/tmp`), so link targets compared against the unresolved root would all look external. → Mitigation: always compare against `repo_resolved`, never `self.repo_path`. Add a test that walks a repo through a symlinked root path.
- **[Risk] A symlink to a directory inside an allowed directory symlink.** → Mitigation: the check runs per entry on the entry's own full resolution, so nesting doesn't matter. Add a test for a chain of links.
- **[Trade-off] Ignore-matching the target means configured `ignore_patterns` now also control which links are followed.** That is the intended behaviour: ignoring a path should mean it is not indexed, whatever route reaches it.
- **[Risk] Hash or ID stability for in-repo links is unchanged**, because paths are still the link's path. No reindex churn.

## Migration Plan

- No schema, fingerprint or config changes. Ship in the next patch release (0.2.1) with a CHANGELOG `### Security` entry describing the exposure and noting that the next search purges affected rows.
- **Rollback** is a plain revert. Previously excluded links would be re-indexed on the next sync.

## Open Questions

- Should `indexter init` or `reindex` report in their summary how many symlinks were skipped, so users can see why a linked file disappeared? This proposal logs at debug level only. It is a small follow-up if wanted.
