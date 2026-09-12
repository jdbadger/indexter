# storage-paths Specification

## Purpose

Replacing the old `repos.json` registry means a repository's database location must be derivable
by pure computation from the repository path alone — no persisted mapping to go stale or need
migrating, and no lookup step before a database can be opened.

This capability defines that derivation (a deterministic `<slug>-<hash12>.db` filename under the
XDG data directory) and the XDG-aware resolution of the application's data and config directories,
and establishes that the reverse direction — which repository a database belongs to — is never
computed from the filename, only read from that database's own stored metadata.

## Requirements

### Requirement: Deterministic database path per repository

The system SHALL map a repository path to exactly one database file path by pure computation, with no registry, index file, or persisted mapping. The database filename SHALL be `<slug>-<hash12>.db`, where `<slug>` is the canonical repository directory name reduced to lowercase alphanumerics and hyphens, and `<hash12>` is the first 12 hexadecimal characters of the SHA-256 digest of the canonical repository path string.

#### Scenario: Same repository resolves to the same database path

- **WHEN** the database path is derived twice for the same repository, in separate processes
- **THEN** both derivations return the identical path

#### Scenario: Different repositories resolve to different database paths

- **WHEN** the database path is derived for two repositories with the same directory name but different parent directories
- **THEN** the two paths differ in their hash component

#### Scenario: Non-canonical input is canonicalized first

- **WHEN** the database path is derived from a relative path, a path containing `.`/`..` segments, a trailing slash, or a symlink to the repository
- **THEN** the returned path is identical to the one derived from the fully resolved repository path

#### Scenario: Directory name is made filesystem-safe

- **WHEN** the repository directory name contains characters outside lowercase alphanumerics and hyphens, such as spaces, dots, or uppercase letters
- **THEN** the slug component contains only lowercase alphanumerics and hyphens, and the hash component is still computed from the unmodified canonical path

### Requirement: XDG-aware application directories

The system SHALL locate its data directory at `$XDG_DATA_HOME/indexter`, defaulting to `~/.local/share/indexter`, and its configuration directory at `$XDG_CONFIG_HOME/indexter`, defaulting to `~/.config/indexter`. Directories SHALL be created on demand when something is written to them, and SHALL NOT be created merely by asking for their path.

#### Scenario: XDG environment variables are honoured

- **WHEN** `XDG_DATA_HOME` is set to a directory
- **THEN** the data directory is that directory joined with `indexter`

#### Scenario: Defaults apply when XDG variables are unset or empty

- **WHEN** `XDG_DATA_HOME` and `XDG_CONFIG_HOME` are unset, or set to an empty string
- **THEN** the data directory is `~/.local/share/indexter` and the config directory is `~/.config/indexter`

#### Scenario: Asking for a path has no side effects

- **WHEN** the data directory path is requested and no directory exists there
- **THEN** the path is returned and no directory is created on disk

### Requirement: Reverse lookup from database to repository

The system SHALL NOT attempt to invert the path hash. The repository a database belongs to SHALL be read from that database's own stored metadata.

#### Scenario: Repository identity comes from the database

- **WHEN** the repository path for an existing database file is requested
- **THEN** the value is read from the database's stored `repo_path` metadata, not computed from the filename
