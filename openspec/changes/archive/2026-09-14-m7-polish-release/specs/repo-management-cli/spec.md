## MODIFIED Requirements

### Requirement: Single CLI entry point

The system SHALL expose one console command, `indexter`, backed by a command group. Running it with no arguments or with `--help` SHALL list the available commands. Running it with `--version` SHALL print `indexter <version>`, where `<version>` is the installed package version, and exit zero without running any command. Every command SHALL exit non-zero on failure and zero on success.

#### Scenario: Help lists the commands

- **WHEN** `indexter --help` is run
- **THEN** the output lists the available commands and the process exits zero

#### Scenario: Unknown command fails clearly

- **WHEN** a command name that does not exist is invoked
- **THEN** the process exits non-zero with a message naming the unknown command

#### Scenario: Version

- **WHEN** `indexter --version` is run
- **THEN** the output is `indexter` followed by the installed package version, and the process exits zero

#### Scenario: Version takes precedence over a command

- **WHEN** `indexter --version list` is run
- **THEN** the version is printed, the `list` command does not run, and the process exits zero
