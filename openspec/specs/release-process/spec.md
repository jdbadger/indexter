# release-process Specification

## Purpose

The rewrite's version, `0.1.0`, is lower than the `0.1.2` of the old Qdrant-based tool already
published to PyPI under the same name, and the package has no license file, changelog or path from
a clean tree to a published release. This capability is that path: version numbers that only ever
increase past what is already published, a `just release` recipe that bumps the version and tags a
commit without pushing or publishing anything, a changelog that tells a 0.1.x user what changed and
how to upgrade, package metadata complete enough to publish, and a publish workflow — triggered by
a GitHub release — that builds, smoke-tests and publishes with trusted publishing to TestPyPI or
PyPI depending on whether the release is a prerelease.

## Requirements

### Requirement: Versions increase past the previously published package

Every published version SHALL be greater than every version of `indexter` already on PyPI; the first release of the rewrite SHALL be `0.2.0`, preceded by release candidates `0.2.0rcN`. Versions SHALL be changed only through the release recipe, never by hand.

#### Scenario: First release candidate

- **WHEN** the maintainer runs `just release minor 1` with the project at `0.1.0`
- **THEN** the project version becomes `0.2.0rc1`

#### Scenario: Next release candidate

- **WHEN** the maintainer runs `just release "" 2` with the project at `0.2.0rc1`
- **THEN** the project version becomes `0.2.0rc2`

#### Scenario: Final release from a candidate

- **WHEN** the maintainer runs `just release` with the project at `0.2.0rc2`
- **THEN** the project version becomes `0.2.0`

### Requirement: The release recipe commits and tags, and nothing else

`just release [bump] [rc]` SHALL refuse to run with uncommitted changes; SHALL compute the new version with `uv version` (a `bump` of major, minor or patch, optionally with a release-candidate suffix; an `rc` alone to advance the candidate number; neither to promote a candidate to final); SHALL fail with a usage message when asked to promote a version that is not a prerelease; SHALL write the version to `pyproject.toml` and `uv.lock`, commit exactly those two files as `release: v<version>`, and create the annotated tag `v<version>`. It SHALL NOT push or publish. It SHALL work with the BSD tools on macOS as well as on Linux.

#### Scenario: Dirty working tree

- **WHEN** `just release minor 1` is run with uncommitted changes
- **THEN** it exits non-zero without changing the version, committing or tagging

#### Scenario: Promoting a final version

- **WHEN** `just release` is run with the project at `0.2.0`
- **THEN** it exits non-zero with a message explaining that a bump is required

#### Scenario: Successful release candidate

- **WHEN** `just release minor 1` succeeds
- **THEN** the latest commit is `release: v0.2.0rc1` touching only `pyproject.toml` and `uv.lock`, the tag `v0.2.0rc1` points at it, and nothing has been pushed

### Requirement: A final release requires a changelog entry

`CHANGELOG.md` SHALL follow the Keep a Changelog format, with unreleased changes under `## [Unreleased]`. For a final (non-prerelease) version, `just release` SHALL fail before changing anything unless `CHANGELOG.md` contains a `## [<version>]` heading for the version being released. Release candidates SHALL NOT require one.

#### Scenario: Missing entry for a final release

- **WHEN** `just release` would produce `0.2.0` and `CHANGELOG.md` has no `## [0.2.0]` heading
- **THEN** it exits non-zero naming the missing heading, and the version, commits and tags are unchanged

#### Scenario: Release candidate without an entry

- **WHEN** `just release minor 1` is run and `CHANGELOG.md` only has `## [Unreleased]`
- **THEN** the release candidate is created

### Requirement: The changelog explains upgrading from 0.1

The changelog entry for the rewrite SHALL state that it replaces the 0.1.x tool, and SHALL tell a 0.1.x user that the old commands, MCP tools, registry and Qdrant storage are gone; that configuration keys from 0.1 are rejected and must be removed from the global configuration file; and that each repository must be indexed again and the MCP server re-registered. The README SHALL point to this changelog entry rather than duplicating its steps.

#### Scenario: Upgrading user with an old config key

- **WHEN** a 0.1.x user reads the changelog entry after `indexter` reports an unknown setting such as `default_root`
- **THEN** the upgrade notes tell them to delete that key from `~/.config/indexter/config.toml`

### Requirement: Package metadata is complete for publishing

The built distributions SHALL include the MIT license file declared through `license-files`, the README as the long description, project URLs whose changelog link resolves to `CHANGELOG.md` in the repository, and trove classifiers that describe a command-line tool for Python 3.11–3.13 without claiming a typed library API.

#### Scenario: License in the wheel metadata

- **WHEN** the wheel's `.dist-info` is inspected
- **THEN** it contains the `LICENSE` file and its metadata declares the MIT license expression

#### Scenario: Classifiers

- **WHEN** the package metadata is inspected
- **THEN** the classifiers include Python 3.11, 3.12 and 3.13 and do not include `Typing :: Typed`

### Requirement: Publishing a GitHub release publishes the package

A publish workflow SHALL run when a GitHub release is published. It SHALL build the distributions once, run the same distribution checks and smoke test as CI, and fail before publishing unless the release tag equals `v` followed by the project version. It SHALL publish with trusted publishing and no stored credentials: to TestPyPI when the release is marked as a prerelease, and to PyPI otherwise, each from its own deployment environment.

#### Scenario: Prerelease

- **WHEN** a GitHub release for tag `v0.2.0rc1` is published as a prerelease with the project at `0.2.0rc1`
- **THEN** the distributions are published to TestPyPI and not to PyPI

#### Scenario: Final release

- **WHEN** a GitHub release for tag `v0.2.0` is published, not as a prerelease, with the project at `0.2.0`
- **THEN** the distributions are published to PyPI and not to TestPyPI

#### Scenario: Tag does not match the version

- **WHEN** a GitHub release for tag `v0.2.1` is published while the tagged commit's project version is `0.2.0`
- **THEN** the workflow fails before any publish job runs, naming both versions
