## Why

The walker checks symlinked directories but not symlinked files. `entry.is_file()` and `entry.stat()` follow links, and `read_file()` reads through them. A repository can therefore commit a file symlink to a path outside the repo, such as `docs/notes.md -> ../../.ssh/id_ed25519`. That file gets indexed, and its contents come back as a snippet from the MCP `search` tool.

Pulling a malicious commit into a repo that is already indexed is enough, because search re-syncs before answering. Prompt-injection text in the same repo can then ask the agent to search for the secret and send it out through other tools. This breaks the existing file-walking requirement "The walker SHALL NOT follow symlinks that resolve outside the repository."

There is a worse variant that the security review did not cover: a symlink whose target is *inside* the repository but ignored. Examples are `notes.md -> .env` and `notes.md -> .git/config`, where the config may hold a token embedded in a remote URL. The containment check accepts these, so gitignored local secrets are indexed without any path guessing. A directory symlink such as `link -> .git` is descended today for the same reason.

## What Changes

- **Symlinked files.** The walker SHALL skip a symlinked file whose resolved target is outside the repository root. It currently indexes that file.
- **Symlinks to ignored paths.** The walker SHALL skip any symlink, file or directory, whose resolved target is inside the repository but matches the ignore rules for the walk. Those rules are `.git/`, the configured patterns and the root `.gitignore`. A link may only reach content that the walk would index by its real path anyway.
- **Content reads.** Reading contents (`read_file`) SHALL refuse a path that resolves outside the repository root. This protects the snippet read at search time, which uses paths stored in the database and does not run the walker.
- **Existing indexes.** Content indexed through such a symlink before this change is removed on the next sync, which `search` triggers automatically. The file no longer appears in the walk, so the existing removed-file handling deletes its rows. No migration or fingerprint bump is needed.
- **Unchanged.** Symlinks, file or directory, whose targets are inside the repository and not ignored are still followed. They are indexed under the link's own path, as today.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `file-walking`: The "Traversal safety" requirement is extended from symlinked directories to symlinked files, and it now also rejects links whose targets are ignored paths. The "Content reading and hashing" requirement gains a containment rule for the read step.

## Impact

- **Code**:
  - `src/indexter/walk.py`: `Walker._walk_dir`, `_symlink_dir_within_repo` (generalised) and `read_file`.
  - No changes in `index/sync.py` or `search/results.py`. They get the protection through `Walker` and `read_file`.
- **Tests**: `src/indexter/tests/test_walk.py` gets new safety cases. An end-to-end sync/search test checks that an escaping link's content never reaches the index or a snippet, and that rows indexed before the fix are removed.
- **Behaviour**: Repositories that deliberately symlink files from outside the checkout will no longer index those files. An example is a shared config pulled in from a sibling directory. This matches how directory symlinks already behave.
- **Docs**: CHANGELOG `[Unreleased]` gets a `### Security` entry.
- **Dependencies / APIs / schema**: No changes.
