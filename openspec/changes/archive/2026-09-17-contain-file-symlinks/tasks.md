## 1. Walker: symlink admission

- [x] 1.1 In `src/indexter/walk.py`, replace `_symlink_dir_within_repo` with a single predicate (e.g. `_symlink_admitted(entry, repo_resolved, is_dir)`). It resolves with `resolve(strict=True)` and requires the target to be under `repo_resolved`. It also requires the target's repo-relative POSIX path, with a trailing `/` for directories, to not match `self._matcher`. It returns False on `OSError`/`ValueError`.
- [x] 1.2 In `Walker._walk_dir`, apply the predicate to every entry where `entry.is_symlink()`. Do this before `is_file()` and `_consider_file`, and for directories alongside the existing link-path ignore check. Log a debug line for each skipped link.
- [x] 1.3 Confirm that all comparisons use `repo_resolved`, never `self.repo_path`, so a repo reached through a symlinked root still admits its own in-repo links.

## 2. Content read containment

- [x] 2.1 In `read_file`, resolve the repo root and `Path(repo_path) / relpath` with `strict=True`. Return `None` without opening the file if resolution fails or the target is not under the resolved root. Keep the signature and the `(content, hash)` return type unchanged.
- [x] 2.2 Update the `read_file` docstring to say it refuses paths that resolve outside the repository.

## 3. Walker and read tests (`src/indexter/tests/test_walk.py`)

- [x] 3.1 `TestSafety`: a file symlink with a relative `..` target outside the repo is not yielded.
- [x] 3.2 `TestSafety`: a file symlink with an absolute target outside the repo is not yielded. Assert that the target is never stat'd, for example by monkeypatching `_consider_file` or checking with an unreadable target.
- [x] 3.3 `TestSafety`: a chain of links where an in-repo link points at a second in-repo link that escapes is not yielded.
- [x] 3.4 `TestSafety`: a file symlink to a gitignored in-repo file (`notes.md -> .env`, with `.env` in `.gitignore`) is not yielded.
- [x] 3.5 `TestSafety`: a file symlink to a file under a gitignored directory, and a file symlink to a configured `ignore_patterns` match, are not yielded.
- [x] 3.6 `TestSafety`: a directory symlink `link -> .git` and a file symlink `cfg -> .git/config` are not yielded or descended.
- [x] 3.7 `TestSafety`: an in-repo, non-ignored file symlink is still yielded under the link's path. Keep `test_symlink_within_repo_is_fine` passing.
- [x] 3.8 `TestSafety`: walking through a symlinked repo root path still yields the in-repo links.
- [x] 3.9 `read_file` tests: it returns `None` for an escaping file symlink, for a `../outside.py` relpath, and for a broken link. It still reads a contained `a/../b.py`.

## 4. End-to-end tests

- [x] 4.1 In `src/indexter/index/tests/test_sync_repo.py`: sync a repo with an escaping file symlink and assert its path is absent from `files`, and its content is absent from `nodes_fts`.
- [x] 4.2 In `test_sync_repo.py`: simulate a pre-fix index by inserting the link's rows through `write_file` for the link path, then sync. Assert that the path is reported as removed and its nodes and FTS rows are gone.
- [x] 4.3 In `src/indexter/search/tests/test_results.py`, next to `test_vanished_file_yields_no_snippet`: replace an indexed file with a symlink that escapes the repo and assert `read_snippets` yields `None` for it.

## 5. Spec, docs and verification

- [x] 5.1 Add a `### Security` entry under `[Unreleased]` in `CHANGELOG.md`. It should cover the exposure (file symlinks outside the repo, and symlinks to ignored paths such as `.env` and `.git/`, were indexed and returned by `search`), say that the next search purges affected rows, and note that out-of-repo file links are no longer indexed.
- [x] 5.2 Run `just lint` and `just test`, and make sure both pass.
- [x] 5.3 Run `openspec validate contain-file-symlinks` and make sure it passes. When archiving, merge the MODIFIED requirements into `openspec/specs/file-walking/spec.md`; do not sync them verbatim.
