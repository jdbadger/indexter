# embedding Specification

## Purpose

The composer produces text; something has to turn it into vectors comparable to a query
embedding at search time. That requires selecting a backend, loading a model and its tokenizer
without paying that cost when nothing needs embedding, and verifying the model's output matches
the configured dimension before writing to `vectors`.

This capability owns the `Embedder` interface: backend selection between sentence-transformers
and fastembed by configuration, lazy and independent loading of the tokenizer and the model,
L2-normalized batched embedding matching the configured dimension, and actionable errors for an
unsupported backend, a dimension mismatch, or a model that needs to be downloaded.
## Requirements
### Requirement: Embedding backend is selected by configuration

The system SHALL provide an embedder for the configured `embedding_backend`: `sentence-transformers` (the default) or `fastembed`. Both SHALL embed with the configured `embedding_model`. Selecting `fastembed` when it is not installed SHALL fail with an error naming the `onnx` extra that provides it.

#### Scenario: Default backend

- **WHEN** an embedder is created with default settings
- **THEN** it uses sentence-transformers with the configured model

#### Scenario: fastembed backend

- **WHEN** an embedder is created with `embedding_backend = "fastembed"` and fastembed is installed
- **THEN** it embeds with fastembed using the configured model

#### Scenario: fastembed not installed

- **WHEN** an embedder is created with `embedding_backend = "fastembed"` and fastembed is not importable
- **THEN** the first use fails with an error that names the `onnx` extra

#### Scenario: Unknown backend is rejected

- **WHEN** settings name a backend other than the two supported ones
- **THEN** settings resolution fails with a configuration error naming the key

### Requirement: Model and tokenizer load lazily and independently

Creating an embedder SHALL NOT load the tokenizer, the model, or the model's machine-learning framework. The tokenizer SHALL be loaded on first request for it, from the model repository's tokenizer file, without importing the model's framework. The model SHALL be loaded on the first embedding request. Each SHALL be loaded at most once per embedder instance.

#### Scenario: Construction is free

- **WHEN** an embedder is created and not used
- **THEN** neither the tokenizer nor the model has been loaded and no deep-learning framework has been imported by it

#### Scenario: Tokenizer without the model

- **WHEN** only the tokenizer is requested from a sentence-transformers embedder
- **THEN** the tokenizer is returned and the model has not been loaded

#### Scenario: Loads happen once

- **WHEN** an embedder embeds several batches
- **THEN** the model is loaded exactly once

### Requirement: Embeddings are normalized and match the configured dimension

The embedder SHALL return one vector per input text, in input order, L2-normalized, as 32-bit floats serialized in the form the vector table accepts. On first model load it SHALL compare the model's output dimension with `embedding_dim` and, if they differ, fail with an error naming both dimensions and the setting to change. Texts SHALL be embedded in batches of `embed_batch_size`, and the model's own maximum sequence length SHALL be set to `embed_max_tokens`.

#### Scenario: Output order and count

- **WHEN** a list of texts is embedded
- **THEN** exactly one vector per text is returned, in the same order

#### Scenario: Vectors are unit length

- **WHEN** any text is embedded
- **THEN** the returned vector has an L2 norm of 1 within floating-point tolerance

#### Scenario: Dimension mismatch fails loudly

- **WHEN** the configured model produces 384-dimensional vectors but `embedding_dim` is 768
- **THEN** the first embedding request fails with an error stating both 384 and 768 and naming `embedding_dim`

#### Scenario: Empty input

- **WHEN** an empty list of texts is embedded
- **THEN** an empty list is returned without loading the model

### Requirement: Model acquisition failures are actionable

When the configured model or its tokenizer is absent from the local model cache, the system SHALL download it. When that download fails, the system SHALL raise an error that names the model and states that network access is needed once to fetch it.

#### Scenario: Cached model works offline

- **WHEN** the model is present in the local cache and there is no network access
- **THEN** the tokenizer and model load successfully

#### Scenario: Missing model without network

- **WHEN** the model is not cached and cannot be downloaded
- **THEN** the failure names the model and says it must be downloaded once with network access

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

