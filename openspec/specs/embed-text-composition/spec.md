# embed-text-composition Specification

## Purpose

M2 produces parsed nodes with byte ranges and no text of their own. Embedding and full-text
search both need actual text per node, and the plan requires that text to be deterministic,
inspectable, and shaped to fit the embedding model's token budget — without ever calling a
language model.

This capability owns composition: turning a parsed file into per-node embed text, qualified
names, name words, and full-text bodies. It orders sections most-meaningful-first (label,
signature, docstring prose, body, structured documentation), splits identifiers into words,
provides per-kind variants for containers, files, sections, data blocks, and chunks, truncates
to the model's token budget at a token boundary using the model's own tokenizer, and excludes
child byte ranges from each node's full-text body.

## Requirements

### Requirement: Composition is deterministic and file-scoped

The system SHALL compose embed text for every node of a parsed file from that file's relative path, its content, and its parse result, without calling a language model. Composing the same inputs with the same tokenizer and budget SHALL always produce identical text. For each node the composer SHALL also produce the node's `qualified_name`, its `name_words`, its full-text body, and an `embed_hash` that is a hash of the composed text.

#### Scenario: Composition is reproducible

- **WHEN** the same file content is composed twice with the same tokenizer and budget
- **THEN** every node's embed text, qualified name, name words, body and embed hash are identical across both runs

#### Scenario: Embed hash follows the text

- **WHEN** two nodes compose to the same embed text
- **THEN** they have the same embed hash, and any difference in embed text yields a different embed hash

#### Scenario: Line shifts do not change composed text

- **WHEN** blank lines are inserted above a function without changing the function itself
- **THEN** that function's embed text and embed hash are unchanged

### Requirement: Sections are ordered most-meaningful-first

For code symbols, the composed text SHALL consist of the following sections in this order, omitting any that are empty: a label; the signature, or the declaration header line when the node has no signature; the docstring's prose; a prefix of the body; and the docstring's structured parameter, return, and exception documentation. The body section SHALL NOT repeat the header or docstring already emitted, SHALL collapse runs of blank lines, and SHALL retain comments and string literals.

#### Scenario: Full section order for a documented method

- **WHEN** a method with a signature, a docstring containing a prose summary and an `Args:` block, and a body is composed within budget
- **THEN** the text contains, in order, the label, the signature, the prose summary, the body, and the `Args:` block

#### Scenario: Body does not repeat the docstring

- **WHEN** a Python function with a docstring is composed
- **THEN** the docstring text appears once, in its docstring section, and not again in the body section

#### Scenario: String literals are kept

- **WHEN** a function body contains an error message string literal
- **THEN** the literal's text appears in the composed body section

#### Scenario: Blank lines are collapsed

- **WHEN** a body contains several consecutive blank lines
- **THEN** the composed body contains no run of more than one blank line

#### Scenario: Structured documentation in other conventions

- **WHEN** a JavaScript function's JSDoc contains `@param` tags or a Rust function's doc comment contains an `# Arguments` section
- **THEN** those parts are placed in the structured-documentation section after the body, and the remaining doc text is placed in the prose section

### Requirement: Labels identify the node in words

The label SHALL contain the node's kind, its qualified name, its name split into lowercase words, and its file path together with the path split into words. Identifier splitting SHALL separate snake_case, kebab-case, camelCase and PascalCase boundaries, runs of capitals (acronyms), and digit runs. A node's `qualified_name` SHALL be its scope path and name joined with `.`; for a file node it SHALL be the relative path. A node's `name_words` SHALL be its name's split words; for a file node they SHALL be the relative path's split words.

#### Scenario: Snake case is split

- **WHEN** a function named `get_user_by_email` is composed
- **THEN** its name words are `get user by email` and they appear in its label

#### Scenario: Camel case and acronyms are split

- **WHEN** a class named `HTTPServer2` is composed
- **THEN** its name words are `http server 2`

#### Scenario: Qualified name includes the full scope

- **WHEN** a method `login` with scope path `("AuthHandler",)` is composed
- **THEN** its qualified name is `AuthHandler.login` and the label contains its kind, `AuthHandler.login`, and the file path's words

#### Scenario: File node naming

- **WHEN** the file node of `src/auth/handlers.py` is composed
- **THEN** its qualified name is `src/auth/handlers.py` and its name words are the words of that path

### Requirement: Per-kind composition variants

Container and non-code kinds SHALL use a variant in place of the body section. Class, struct, trait, interface and enum nodes SHALL list the names of their member nodes grouped by kind. File nodes SHALL list their top-level symbols by kind, followed by the file's text not covered by any symbol. Section nodes SHALL use their heading or scope path in the label and their own prose as the body. Data nodes SHALL use their key path in the label and a slice of their source as the body. Chunk nodes SHALL use the file path and line range in the label and the chunk's raw text as the body.

#### Scenario: A class lists its members instead of its body

- **WHEN** a class with methods `login` and `validate` is composed
- **THEN** its composed text names `login` and `validate` and does not contain their bodies

#### Scenario: A file lists its top-level symbols

- **WHEN** the file node of a module defining a class and two functions is composed
- **THEN** its composed text names the class and both functions, followed by the module's imports and top-level statements

#### Scenario: A markdown section uses its heading path

- **WHEN** a section with heading path `Setup > Prerequisites` is composed
- **THEN** its label contains `Setup > Prerequisites` and its body is that section's prose

#### Scenario: A data block uses its key path

- **WHEN** a YAML mapping nested under `services.web` is composed
- **THEN** its label contains the key path and its body is a slice of the mapping's source

#### Scenario: A chunk carries its location

- **WHEN** a chunk covering lines 10–40 of `notes.txt` is composed
- **THEN** its label contains the path and the line range and its body is the chunk's text

### Requirement: Composed text fits the model's token budget

The composed text SHALL be counted with the embedding model's own tokenizer, with the tokenizer's built-in truncation and padding disabled, and SHALL NOT exceed the configured `embed_max_tokens` minus the model's special tokens. When the joined sections exceed the budget, the text SHALL be cut at a token boundary, so that trailing sections are dropped before earlier ones and the last surviving section is cut rather than removed.

#### Scenario: Within-budget text is untouched

- **WHEN** a short function's joined sections fit the budget
- **THEN** the composed text contains every section in full

#### Scenario: Over-budget text drops the tail first

- **WHEN** a function with a long body and an `Args:` block exceeds the budget
- **THEN** the `Args:` block is absent, the label and signature are present in full, and the body is cut

#### Scenario: Cut falls on a token boundary

- **WHEN** composed text is truncated
- **THEN** re-tokenizing the result yields no more tokens than the budget and its final token is a complete token from the original encoding

#### Scenario: Tokenizer defaults do not cap the count

- **WHEN** the model's tokenizer file is configured with truncation or padding at a smaller length than the budget
- **THEN** token counts reflect the text's real length and a text longer than that configured length is truncated to the budget, not to the tokenizer file's length

### Requirement: Full-text bodies exclude child nodes

Each node's full-text body SHALL be the source text of its byte range with the byte ranges of its child nodes removed, so that each part of the source is indexed in full text on the innermost node containing it.

#### Scenario: A method body is indexed once

- **WHEN** a file containing a class with one method is composed
- **THEN** the method's body text appears in the method's full-text body and not in the class's or the file's

#### Scenario: Module-level code is indexed on the file node

- **WHEN** a script file with imports and top-level statements but no symbols is composed
- **THEN** the file node's full-text body contains the whole file

#### Scenario: Class attributes are indexed on the class

- **WHEN** a class defines a class-level attribute and a method
- **THEN** the attribute's text appears in the class's full-text body and the method's text does not
