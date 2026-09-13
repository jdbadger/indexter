## ADDED Requirements

### Requirement: A question set with expected answers

The repository SHALL contain an eval question set naming a target repository and between 15 and 20 natural-language questions, each with one or more expected repository-relative file paths. Questions SHALL describe behavior without naming the symbols or modules that implement it.

#### Scenario: Question set loads

- **WHEN** the eval reads the question set
- **THEN** it obtains the target repository and every question with a non-empty list of expected files

#### Scenario: Expected file missing from the target

- **WHEN** a question's expected file does not exist in the target repository
- **THEN** the eval reports that question as invalid by name instead of scoring it as a miss

### Requirement: Questions are scored by hit@5 and MRR@10

For each question the eval SHALL run a search with default limit and filters and SHALL score hit@5 as whether any of the first 5 selected entries is in an expected file, and MRR@10 as the reciprocal of the position of the first such entry within the first 10, or 0. It SHALL print per-question results, totals per variant, rendered response sizes, and search latency (median and maximum), and SHALL record the target repository's commit and whether its working tree was dirty.

#### Scenario: Per-question and total scores

- **WHEN** the eval completes a variant
- **THEN** it prints, for each question, the position of the first correct entry or a miss, and the variant's hit@5 count and mean MRR@10

#### Scenario: Target state recorded

- **WHEN** the eval runs against a repository with uncommitted changes
- **THEN** its output records the commit and that the working tree was dirty

### Requirement: Variants are compared under the same conditions

The eval SHALL support these variants: the production configuration; raw-code embeddings, in which each node is embedded from its own source text truncated to the same token budget while full-text rows are unchanged; vector-only and keyword-only ranking; ranking without test-file demotion; and an alternative embedding model of the configured dimension. Each variant SHALL build its index with settings from the packaged defaults plus that variant's overrides, never from the user's global or repository configuration, into a data directory specific to the variant, and SHALL NOT read or modify the user's own databases. Variants sharing an index SHALL reuse it.

#### Scenario: User configuration is ignored

- **WHEN** the user's global configuration sets a different embedding model
- **THEN** the production variant still indexes with the packaged default model

#### Scenario: Real databases untouched

- **WHEN** the eval runs against a repository the user has already initialized
- **THEN** the user's database for that repository is neither read nor modified

#### Scenario: Raw variant changes only embeddings

- **WHEN** the raw variant indexes the target
- **THEN** its nodes' full-text rows equal the production variant's, and its embedded texts are the nodes' source text

#### Scenario: Ranking variants reuse the index

- **WHEN** the vector-only, keyword-only and no-demotion variants run after the production variant
- **THEN** none of them re-embeds the target

### Requirement: The eval runs from one command, outside CI

`just eval` SHALL run the eval over all variants except alternative models, with options to choose variants, the target repository, the cache directory, and to print each question's rendered response. The eval SHALL NOT be part of the test suite or its coverage.

#### Scenario: One command

- **WHEN** a developer runs `just eval` with the target repository present
- **THEN** every default variant is indexed as needed, scored, and summarized

#### Scenario: Showing responses

- **WHEN** the eval is run with its show option
- **THEN** each question's rendered search response is printed with its score

### Requirement: The composed-versus-raw decision follows a fixed rule

The composed-summary embeddings SHALL remain the production choice unless the raw variant, with hybrid ranking, answers at least 2 more questions at hit@5 and has a mean MRR@10 at least as high. The outcome and the numbers it rests on SHALL be recorded with the change that introduces the eval.

#### Scenario: Raw does not clear the bar

- **WHEN** raw answers 1 more question at hit@5 than composed
- **THEN** composed remains the production choice

#### Scenario: Raw clears the bar

- **WHEN** raw answers 2 more questions at hit@5 and its mean MRR@10 is higher
- **THEN** the recorded outcome is that raw wins, and changing composition is proposed as a separate change
