## ADDED Requirements

### Requirement: Layered configuration resolution

The system SHALL resolve configuration from four layers, each overriding the one before it: packaged defaults, the global user configuration file, the per-repository configuration, and explicit arguments supplied by the caller. Layers SHALL be merged key by key, so a layer that sets one key does not discard values set by lower layers.

#### Scenario: Defaults apply when no configuration files exist

- **WHEN** configuration is resolved for a repository with no global config file and no repository config
- **THEN** every setting holds its packaged default value

#### Scenario: Repository config overrides global config

- **WHEN** the global config and the repository config both set the same key to different values
- **THEN** the resolved configuration holds the repository value

#### Scenario: Partial overrides leave other keys intact

- **WHEN** the repository config sets exactly one key
- **THEN** that key takes the repository value and all other keys retain their global or default values

#### Scenario: Explicit arguments win over every file

- **WHEN** a caller passes an explicit value for a setting that is also set in the global and repository config files
- **THEN** the resolved configuration holds the caller's value

### Requirement: Global configuration file location

The system SHALL read global configuration from `config.toml` in the configuration directory. A missing global config file SHALL NOT be an error.

#### Scenario: Missing global config is not an error

- **WHEN** configuration is resolved and no global config file exists
- **THEN** resolution succeeds using defaults and any repository-level values

#### Scenario: Malformed global config is an error

- **WHEN** the global config file is not valid TOML
- **THEN** resolution fails with an error naming the file path and the parse error

### Requirement: Per-repository configuration sources

The system SHALL read per-repository configuration from `indexter.toml` at the repository root if that file exists, and otherwise from the `[tool.indexter]` table of the repository's `pyproject.toml`. The two sources SHALL NOT be merged with each other.

#### Scenario: Dedicated repository config file is used

- **WHEN** the repository root contains `indexter.toml`
- **THEN** its top-level keys are used as the repository configuration layer

#### Scenario: pyproject table is used as a fallback

- **WHEN** the repository root has no `indexter.toml` but its `pyproject.toml` contains a `[tool.indexter]` table
- **THEN** that table is used as the repository configuration layer

#### Scenario: Both sources present

- **WHEN** the repository root contains both `indexter.toml` and a `[tool.indexter]` table in `pyproject.toml`
- **THEN** `indexter.toml` is used in full, the `[tool.indexter]` table is ignored entirely, and a warning names the file that won

#### Scenario: Neither source present

- **WHEN** the repository has neither file
- **THEN** the repository configuration layer is empty and resolution succeeds

### Requirement: Unknown and invalid settings are rejected

The system SHALL reject configuration containing keys it does not recognize, and SHALL reject values of the wrong type or outside the allowed range for their setting. The error SHALL name the offending key and the file it came from.

#### Scenario: Unknown key is rejected

- **WHEN** a configuration file contains a key that is not a recognized setting
- **THEN** resolution fails with an error naming that key and the source file

#### Scenario: Wrong value type is rejected

- **WHEN** a configuration file sets a numeric setting to a string
- **THEN** resolution fails with an error naming that key, the source file, and the expected type

### Requirement: Configuration is a typed, immutable object

Resolved configuration SHALL be exposed as a validated, immutable typed object rather than a raw dictionary, so that consumers get attribute access and static type checking.

#### Scenario: Resolved configuration is typed

- **WHEN** configuration is resolved
- **THEN** the result is a validated settings object whose fields are accessible as typed attributes

#### Scenario: Resolved configuration cannot be mutated

- **WHEN** a caller attempts to assign to a field of a resolved configuration object
- **THEN** the assignment raises an error
