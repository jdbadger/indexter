## ADDED Requirements

### Requirement: Synchronous repository traversal

The walker SHALL traverse a repository synchronously and yield one entry per candidate file, as an ordinary generator. Entries SHALL carry the file's repository-relative path, size in bytes, modification time, and lowercased extension, and SHALL NOT carry file contents.

#### Scenario: Files are yielded with stat metadata

- **WHEN** a repository containing source files is walked
- **THEN** one entry is yielded per candidate file, each carrying its relative path, size, modification time, and extension, and no file contents

#### Scenario: Traversal is a plain generator

- **WHEN** the walk is iterated
- **THEN** it can be consumed without an event loop, and iteration is lazy rather than materializing the whole repository first

#### Scenario: Paths are relative to the repository root

- **WHEN** a file nested several directories deep is yielded
- **THEN** its path is relative to the repository root, using forward-slash separators, and never absolute

#### Scenario: Empty repository

- **WHEN** a repository with no eligible files is walked
- **THEN** the walk completes without error and yields nothing

### Requirement: Ignore-pattern filtering

The walker SHALL skip paths matching gitignore-style patterns drawn from configured ignore patterns and the repository's `.gitignore`. Directories matching an ignore pattern SHALL be pruned without descending into them.

#### Scenario: Configured patterns are applied

- **WHEN** a file matches a pattern from the resolved configuration's ignore patterns
- **THEN** it is not yielded

#### Scenario: Repository .gitignore is honoured

- **WHEN** the repository root contains a `.gitignore` naming a file or directory
- **THEN** matching paths are not yielded

#### Scenario: Ignored directories are pruned, not descended

- **WHEN** a directory matches an ignore pattern and contains files that would otherwise be eligible
- **THEN** none of its contents are yielded and the directory is not traversed

#### Scenario: Missing .gitignore is not an error

- **WHEN** the repository has no `.gitignore`
- **THEN** the walk proceeds using only the configured patterns

### Requirement: Content-independent file filtering

The walker SHALL skip files that can be excluded without reading them: known binary extensions, minified files, files larger than the configured maximum size, and zero-byte files.

#### Scenario: Binary extensions are skipped

- **WHEN** the repository contains files with known binary extensions such as images, archives, or compiled objects
- **THEN** none of them are yielded

#### Scenario: Minified files are skipped

- **WHEN** a file's name contains `.min.` or ends in `.min`
- **THEN** it is not yielded

#### Scenario: Oversized files are skipped

- **WHEN** a file is larger than the configured maximum file size
- **THEN** it is not yielded

#### Scenario: Empty files are skipped

- **WHEN** a file is zero bytes
- **THEN** it is not yielded

### Requirement: Traversal safety

The walker SHALL NOT follow symlinks that resolve outside the repository, and SHALL survive unreadable directories and files that disappear mid-walk.

#### Scenario: Symlink escaping the repository

- **WHEN** the repository contains a symlinked directory whose target resolves outside the repository root
- **THEN** the symlink is not descended into and the walk continues

#### Scenario: Symlink within the repository

- **WHEN** the repository contains a symlinked directory whose target is inside the repository root
- **THEN** the walk continues without error

#### Scenario: Broken symlink

- **WHEN** the repository contains a symlink whose target does not exist
- **THEN** it is skipped and the walk continues

#### Scenario: Unreadable directory

- **WHEN** a directory in the repository cannot be read due to permissions
- **THEN** it is skipped, the remaining files are still yielded, and the walk does not raise

#### Scenario: File disappears mid-walk

- **WHEN** a file is removed between being listed and being stat'd
- **THEN** it is skipped and the walk continues

### Requirement: Content reading and hashing

Reading a file's contents SHALL be a separate step from walking. The read step SHALL return the decoded text together with a SHA-256 hash of the relative path and content combined, and SHALL report failure rather than raising when the file cannot be decoded.

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
