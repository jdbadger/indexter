## ADDED Requirements

### Requirement: A local cache probe classifies the model load before it begins

Before loading the model, the system SHALL determine whether the configured model is already present
in the local model cache, using a check that performs no network access. The result SHALL classify
the load as either a first-time acquisition or a cached load, and SHALL be reported to the caller's
progress facility before the load begins, so the wait can be described while it happens rather than
after.

#### Scenario: Cached model is classified without network

- **WHEN** the configured model is present in the local cache
- **THEN** the probe reports a cached load without making any network request

#### Scenario: Absent model is classified as an acquisition

- **WHEN** the configured model is absent from the local cache
- **THEN** the probe reports a first-time acquisition before any download begins

#### Scenario: Classification precedes the wait

- **WHEN** a model load begins
- **THEN** the classification is reported to the progress facility before the loading work starts

### Requirement: A confirmed cached model loads without contacting the network

When the cache probe confirms the model is present locally, the system SHALL load it with local files
only, performing no network round-trip. When loading with local files only fails because the cached
copy is incomplete, the system SHALL retry once without that restriction, reclassifying the load as a
first-time acquisition, so an incomplete cache is recoverable rather than permanently broken.

#### Scenario: Cache hit skips the network

- **WHEN** the model is present in the local cache and is loaded
- **THEN** the model is constructed with local files only and no network request is made

#### Scenario: Incomplete cache recovers

- **WHEN** the probe reports a cached model but loading with local files only fails because files are missing
- **THEN** the load is retried once without that restriction, is reported as a first-time acquisition, and succeeds if the missing files can be fetched

#### Scenario: A pinned model is not silently refreshed

- **WHEN** a cached model is loaded and a newer revision exists upstream
- **THEN** the cached revision is used and no upstream revision is fetched

### Requirement: Model acquisition reports elapsed time without promising a total

While acquiring a model for the first time, the system SHALL report that this is a one-time
acquisition, name the model, and report elapsed time. It SHALL NOT report a byte total, a percentage,
or an estimated completion time for the acquisition, because transfer sizes cannot be predicted
reliably. While loading an already-cached model, the system SHALL report that it is loading and name
the model.

#### Scenario: First acquisition is described as one-time

- **WHEN** a model is acquired for the first time with narration enabled
- **THEN** the narration names the model, states that the acquisition happens once, and shows elapsed time

#### Scenario: No fabricated totals

- **WHEN** a model acquisition is narrated
- **THEN** the narration contains no byte total, no percentage, and no estimated time remaining

#### Scenario: Cached load is described as loading

- **WHEN** an already-cached model is loaded slowly enough to be painted
- **THEN** the narration says it is loading and names the model

### Requirement: The model backend's own console output is suppressed

The system SHALL suppress the model backend libraries' own console output — including hub
authentication warnings and weight-loading progress bars — so that the only output produced during an
index is the system's own. Suppression SHALL be applied after those libraries are imported, since
importing them reconfigures their logging. Error-level messages SHALL NOT be suppressed, and
suppression SHALL be process-local rather than applied through environment variables.

#### Scenario: No borrowed warnings

- **WHEN** a model is loaded
- **THEN** no hub authentication warning appears on any stream

#### Scenario: No borrowed progress bars

- **WHEN** a model is loaded
- **THEN** no weight-loading progress bar from the backend library appears on any stream

#### Scenario: Errors still surface

- **WHEN** the backend library reports an error-level condition during a load
- **THEN** that condition is not suppressed and still reaches the caller

#### Scenario: Acquisition failures remain actionable

- **WHEN** a model cannot be acquired because it is absent and cannot be downloaded
- **THEN** the failure still names the model and states that network access is needed once, unaffected by suppression
