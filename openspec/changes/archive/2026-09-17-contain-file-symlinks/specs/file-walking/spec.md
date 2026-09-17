## MODIFIED Requirements

### Requirement: Traversal safety

The walker SHALL NOT follow symlinks, whether to files or to directories, that resolve outside the repository. It SHALL NOT follow symlinks whose resolved target, taken as a path relative to the repository root, matches the walk's ignore rules. The walker SHALL survive unreadable directories and files that disappear mid-walk.

#### Scenario: Symlink escaping the repository

- **WHEN** the repository contains a symlinked directory whose target resolves outside the repository root
- **THEN** the symlink is not descended into and the walk continues

#### Scenario: Symlinked file escaping the repository

- **WHEN** the repository contains a symlinked file whose target resolves outside the repository root, whether through a relative `..` chain or an absolute path
- **THEN** the symlink is not yielded, its target is never stat'd or read, and the walk continues

#### Scenario: Symlink resolving through a chain of links

- **WHEN** a symlink points to another symlink inside the repository, which in turn resolves outside the repository root
- **THEN** the first symlink is not yielded or descended into

#### Scenario: Symlink to an ignored path inside the repository

- **WHEN** the repository contains a symlink, to a file or a directory, whose target is inside the repository root but matches an ignore rule, such as a gitignored `.env`, a file under a gitignored directory, or `.git/` and anything beneath it
- **THEN** the symlink is not yielded or descended into, even though the link's own path is not ignored

#### Scenario: Symlink within the repository

- **WHEN** the repository contains a symlinked directory or file whose target is inside the repository root and not ignored
- **THEN** the walk continues without error and the target's content is yielded under the link's path

#### Scenario: Broken symlink

- **WHEN** the repository contains a symlink whose target does not exist
- **THEN** it is skipped and the walk continues

#### Scenario: Unreadable directory

- **WHEN** a directory in the repository cannot be read due to permissions
- **THEN** it is skipped, the remaining files are still yielded, and the walk does not raise

#### Scenario: File disappears mid-walk

- **WHEN** a file is removed between being listed and being stat'd
- **THEN** it is skipped and the walk continues

#### Scenario: Previously indexed escaping symlink is removed

- **WHEN** a repository was indexed while a symlinked file resolving outside it was being followed, and the repository is synced again
- **THEN** that path is no longer walked, and its file, nodes and search entries are removed from the index

### Requirement: Content reading and hashing

Reading a file's contents SHALL be a separate step from walking. The read step SHALL return the decoded text together with a SHA-256 hash of the relative path and content combined. It SHALL report failure rather than raising when the file cannot be decoded. It SHALL report failure without reading anything when the relative path resolves outside the repository root.

#### Scenario: Reading returns content and hash

- **WHEN** an eligible file is read
- **THEN** its decoded text is returned along with a SHA-256 hash computed over the relative path and content combined

#### Scenario: Hash changes when content changes

- **WHEN** the same path is read before and after its content is modified
- **THEN** the two hashes differ

#### Scenario: Hash changes when the path changes

- **WHEN** identical content is read at two different relative paths
- **THEN** the two hashes differ

#### Scenario: Encoding fallback

- **WHEN** a file is not valid UTF-8 but decodes as Latin-1
- **THEN** the read succeeds using the fallback encoding

#### Scenario: Undecodable file

- **WHEN** a file cannot be decoded by any supported encoding
- **THEN** the read reports failure instead of raising, and the file is treated as not indexable

#### Scenario: Walking does not read contents

- **WHEN** a repository is walked without any file being explicitly read
- **THEN** no file contents are loaded

#### Scenario: Read refuses a path resolving outside the repository

- **WHEN** a read is requested for a relative path that is a symlink resolving outside the repository root, or that contains `..` segments escaping it
- **THEN** the read reports failure and the target's contents are not returned

#### Scenario: Snippets are not read through an escaping symlink

- **WHEN** a file that was indexed as a regular file is replaced by a symlink resolving outside the repository before its snippet is read at search time
- **THEN** the snippet is reported as unreadable instead of containing the target's contents
