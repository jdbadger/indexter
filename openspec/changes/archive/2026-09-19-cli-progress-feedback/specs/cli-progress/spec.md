## ADDED Requirements

### Requirement: Narration and results occupy separate streams

Progress narration SHALL be written to stderr. Command results — the existing summaries and error
listings — SHALL remain on stdout, unchanged in content and ordering. No progress narration SHALL
ever appear on stdout.

#### Scenario: Results survive discarding narration

- **WHEN** an indexing command runs with stderr discarded
- **THEN** stdout contains exactly the summary it would have produced with narration disabled

#### Scenario: Narration survives redirecting results

- **WHEN** an indexing command runs with stdout redirected to a file and stderr attached to a terminal
- **THEN** the progress narration is displayed on the terminal and the file contains only the summary

#### Scenario: Narration never reaches stdout

- **WHEN** narration is enabled for any command
- **THEN** no phase label, spinner frame, progress bar, or completion marker appears on stdout

### Requirement: Narration is enabled by terminal detection and overridable by flags

Narration SHALL be enabled by default only when stderr is an interactive terminal. `--quiet` SHALL
disable narration regardless of detection. `--progress` SHALL enable narration regardless of
detection. Supplying both SHALL fail with a message naming the conflict and exit non-zero. Colour
and styling SHALL be suppressed when the environment requests it, independently of whether narration
itself is enabled.

#### Scenario: Non-interactive stderr is silent by default

- **WHEN** an indexing command runs with stderr redirected to a pipe or file and no narration flag
- **THEN** no narration is produced and the command behaves exactly as it does today

#### Scenario: Quiet overrides an interactive terminal

- **WHEN** an indexing command runs with `--quiet` and stderr attached to a terminal
- **THEN** no narration is produced and the summary is still written to stdout

#### Scenario: Progress overrides non-interactive stderr

- **WHEN** an indexing command runs with `--progress` and stderr redirected to a file
- **THEN** narration is written to that file

#### Scenario: Conflicting flags fail

- **WHEN** an indexing command is given both `--quiet` and `--progress`
- **THEN** the process exits non-zero with a message naming both flags and no indexing is performed

#### Scenario: Colour is suppressed on request

- **WHEN** narration is enabled and the environment requests no colour
- **THEN** narration is rendered without colour escape sequences

### Requirement: Phases paint only once they are slow

A phase SHALL NOT paint anything until it has been running longer than a short threshold. A phase
that completes within the threshold SHALL produce no narration at all. Once painting has begun, the
phase SHALL repaint at a bounded refresh rate regardless of how frequently it reports events.

#### Scenario: A fast phase stays silent

- **WHEN** a phase completes in less time than the painting threshold
- **THEN** no line is painted for that phase

#### Scenario: A slow phase paints and resolves

- **WHEN** a phase runs for longer than the painting threshold and then completes
- **THEN** an active line is painted while it runs and is replaced by a completion line that remains in the output

#### Scenario: High-frequency events are throttled

- **WHEN** a phase reports progress events faster than the refresh rate
- **THEN** the display repaints at the bounded rate rather than once per event

### Requirement: Only the active phase animates

At most one phase SHALL be animated at a time. A completed phase SHALL be rendered as a static
completion line that remains in the terminal's scrollback. A phase whose total is known in advance
SHALL render a proportional progress bar with completed and total counts. A phase whose total is not
known in advance SHALL render a running count and SHALL NOT render a proportional bar or a
percentage.

#### Scenario: Completed phases stop animating

- **WHEN** a phase completes and a subsequent phase begins
- **THEN** the completed phase is a static line and only the new phase animates

#### Scenario: Determinate phase shows proportion

- **WHEN** a phase with a known total reports partial progress
- **THEN** the display shows a proportional bar with the completed count and the total

#### Scenario: Indeterminate phase shows a count

- **WHEN** a phase with no known total reports partial progress
- **THEN** the display shows a running count without a percentage or proportional bar

### Requirement: Narration is off unless a caller opts in

The progress facility SHALL be supplied to the indexing pipeline by its caller. When no progress
facility is supplied, the pipeline SHALL produce no narration on any stream. The default SHALL be
the non-reporting facility, so a caller that does not opt in cannot emit output by omission.

#### Scenario: Omitting the facility is silent

- **WHEN** the indexing pipeline is invoked without a progress facility
- **THEN** nothing is written to stdout or stderr by the progress mechanism

#### Scenario: The non-reporting facility does no work

- **WHEN** the non-reporting facility receives phase and progress events
- **THEN** it performs no rendering and no terminal detection
