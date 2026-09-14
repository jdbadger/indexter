# agent-skill Specification

## Purpose

An agent only reaches for `search` and `neighbors` if something tells it these tools exist and
how to use them well — otherwise it falls back to grepping for names it's only guessing at, or
never discovers `neighbors` at all. This capability is that guidance: a packaged `SKILL.md`
shipped inside the package itself, and the `indexter skill` command that gets it in front of an
agent, whether printed for inspection or installed into Claude Code's skills directory.

It owns what the skill teaches — reaching for `search` before grepping, phrasing a query as
behavior rather than a symbol name, reading a response's ranked entries versus its `related`
section, following `path:start-end` locations and node IDs, using `neighbors` and its
`direction`, `edges`, `depth` and `limit` parameters, that every call synchronizes the index
first, and asking the user before running `indexter init` on an unindexed repository — and the
install command's behavior: printing to stdout by default, writing to
`<config>/skills/indexter/SKILL.md` with `--install`, respecting `CLAUDE_CONFIG_DIR` and `--dir`,
and refusing to clobber a locally edited file without `--force`.

## Requirements

### Requirement: A skill file ships with the package

The package SHALL include `indexter/skill/SKILL.md`, readable through the package's resources in an installed wheel. It SHALL begin with frontmatter giving `name: indexter` and a `description` of when an agent should use it. Its body SHALL tell an agent: to use `search` before grepping when it does not know the file or symbol name; to phrase a query as the behavior or concept in plain words; what the `kind`, `language`, `path` and `limit` filters do; how to read a response — ranked entries versus the `related` section, `path:start-end` locations to read, and node IDs; to use `neighbors` with a node ID for callers, callees, importers and subclasses, describing its `direction`, `edges`, `depth` and `limit` parameters; that every call synchronizes the index first; and, when a repository has no index, to ask the user before running `indexter init`. It SHALL mention Claude Code's `mcp__indexter__search` and `mcp__indexter__neighbors` tool names.

#### Scenario: Frontmatter

- **WHEN** the packaged `SKILL.md` is parsed
- **THEN** its frontmatter has `name` equal to `indexter` and a non-empty `description`

#### Scenario: Covers both tools and every neighbors parameter

- **WHEN** the packaged `SKILL.md` is read
- **THEN** it names `search`, `neighbors`, `mcp__indexter__search`, `mcp__indexter__neighbors`, and each of `direction`, `edges`, `depth` and `limit`

#### Scenario: Present in the built wheel

- **WHEN** the package is built as a wheel
- **THEN** the wheel contains `indexter/skill/SKILL.md`

### Requirement: `indexter skill` prints or installs the skill

`indexter skill` SHALL write the packaged `SKILL.md` to stdout byte-for-byte and exit zero. `indexter skill --install` SHALL write it to `<config>/skills/indexter/SKILL.md`, where `<config>` is the `CLAUDE_CONFIG_DIR` environment variable when set and `~/.claude` otherwise, creating directories as needed, and print the path written. `--dir PATH` SHALL replace the `<config>/skills/indexter` directory. When the target file already exists with identical content, the command SHALL report it is up to date and exit zero; when it exists with different content, the command SHALL leave it unchanged and exit non-zero with a message naming `--force`, unless `--force` is given, in which case it SHALL overwrite it. `--force` and `--dir` without `--install` SHALL be rejected.

#### Scenario: Print

- **WHEN** `indexter skill` is run
- **THEN** stdout is exactly the packaged `SKILL.md` and the process exits zero

#### Scenario: Install to Claude Code's user directory

- **WHEN** `indexter skill --install` is run with `CLAUDE_CONFIG_DIR` unset
- **THEN** `~/.claude/skills/indexter/SKILL.md` contains the packaged file and its path is printed

#### Scenario: Config directory override

- **WHEN** `indexter skill --install` is run with `CLAUDE_CONFIG_DIR=/x`
- **THEN** the file is written to `/x/skills/indexter/SKILL.md`

#### Scenario: Custom directory

- **WHEN** `indexter skill --install --dir /agents/skills/indexter` is run
- **THEN** the file is written to `/agents/skills/indexter/SKILL.md`

#### Scenario: Already up to date

- **WHEN** `indexter skill --install` is run and the target already holds identical content
- **THEN** the command reports it is up to date and exits zero

#### Scenario: Local edits are protected

- **WHEN** `indexter skill --install` is run and the target holds different content
- **THEN** the file is unchanged and the command exits non-zero naming `--force`

#### Scenario: Forced overwrite

- **WHEN** `indexter skill --install --force` is run and the target holds different content
- **THEN** the target is replaced with the packaged file

#### Scenario: Force without install

- **WHEN** `indexter skill --force` is run
- **THEN** the command exits non-zero and writes nothing
