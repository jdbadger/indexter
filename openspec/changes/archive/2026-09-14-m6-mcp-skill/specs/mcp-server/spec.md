## ADDED Requirements

### Requirement: The server runs over stdio from one command

The system SHALL provide an `indexter mcp` command that runs an MCP server over the stdio transport until the client disconnects. It SHALL accept an optional `--repo PATH` default repository, which MUST be an existing directory — otherwise the command SHALL exit non-zero with a message naming the path, without starting a server — and which need not be indexed. While serving, the process SHALL write nothing to stdout except MCP protocol messages, and SHALL NOT print a startup banner.

#### Scenario: Tools listed over stdio

- **WHEN** `indexter mcp` is started as a subprocess in a directory with no index and a client lists tools over stdio
- **THEN** the client receives the tool list and every line the process wrote to stdout is a protocol message

#### Scenario: Default repository is not a directory

- **WHEN** `indexter mcp --repo /does/not/exist` is run
- **THEN** the command exits non-zero naming the path and no server starts

#### Scenario: Default repository not yet indexed

- **WHEN** `indexter mcp --repo <dir>` is run for an existing directory with no index
- **THEN** the server starts

### Requirement: The server exposes exactly two tools and instructions

The server SHALL be named `indexter`, SHALL expose exactly the tools `search` and `neighbors`, and SHALL provide instructions of at most two sentences that describe combined meaning-and-keyword search returning real code with graph context, and direct the agent to use `search` when it does not know a file or symbol name and `neighbors` on a returned node ID. Both tools SHALL be annotated read-only, idempotent and not open-world.

#### Scenario: Tool list

- **WHEN** a client lists the server's tools
- **THEN** exactly `search` and `neighbors` are listed, each annotated `readOnlyHint`, `idempotentHint` and not `openWorldHint`

#### Scenario: Instructions

- **WHEN** a client initializes a session
- **THEN** the server's instructions name both `search` and `neighbors` in no more than two sentences

### Requirement: The search tool wraps hybrid search

The `search` tool SHALL take `query` (required), and optionally `repo`, `kind` (one kind or a list), `language` (one language or a list), `path` and `limit`, with a description on every parameter; the `kind` and `language` descriptions SHALL list their valid values. It SHALL run the hybrid search with the resolved repository's settings and return the search response's rendered text unchanged. Its tool metadata SHALL include `"anthropic/alwaysLoad": true`.

#### Scenario: Search returns rendered results

- **WHEN** `search` is called with a query that matches a function in the indexed repository
- **THEN** the tool result is the text the search rendering produces for that query and repository, including the function's `path:start-end` and snippet

#### Scenario: Filters pass through

- **WHEN** `search` is called with `kind=["function", "method"]`, `language="python"`, `path="src"` and `limit=3`
- **THEN** the search runs with those filters and limit

#### Scenario: Always loaded in Claude Code

- **WHEN** a client lists tools
- **THEN** the `search` tool's `_meta` contains `"anthropic/alwaysLoad": true`

#### Scenario: Valid values in the schema

- **WHEN** a client reads the `search` tool's input schema
- **THEN** the `kind` description lists every filterable node kind and the `language` description lists every registered parser language

### Requirement: The neighbors tool wraps graph walking

The `neighbors` tool SHALL take `node_id` (required), and optionally `repo`, `direction`, `edges`, `depth` and `limit`, with a description on every parameter, and SHALL return the rendered neighbors response for the resolved repository. Its tool metadata SHALL include an `"anthropic/searchHint"` string and SHALL NOT include `"anthropic/alwaysLoad"`.

#### Scenario: Neighbors returns rendered results

- **WHEN** `neighbors` is called with a node ID returned by `search` and `direction="in"`
- **THEN** the tool result is the rendered list of that node's incoming neighbors

#### Scenario: Deferred in Claude Code

- **WHEN** a client lists tools
- **THEN** the `neighbors` tool's `_meta` has a string `"anthropic/searchHint"` and no `"anthropic/alwaysLoad"`

### Requirement: Each call resolves its repository

For every tool call the server SHALL choose a starting path: the call's `repo` argument if given (a relative path resolved against the server's working directory), otherwise the server's `--repo` if given, otherwise the server's working directory. The repository SHALL be the nearest of the starting path and its ancestors that has an index database. When none has one, the call SHALL fail with an error naming the starting path and `indexter init`, and SHALL NOT create a database. Settings SHALL be loaded for the resolved repository on every call.

#### Scenario: Working directory is the repository

- **WHEN** the server runs with an indexed repository as its working directory and `search` is called without `repo`
- **THEN** that repository is searched

#### Scenario: Subdirectory resolves upward

- **WHEN** the server's working directory is `src/auth` inside an indexed repository and a tool is called without `repo`
- **THEN** the enclosing repository is used

#### Scenario: Explicit repo wins

- **WHEN** the server was started with `--repo A` and `search` is called with `repo` set to indexed repository `B`
- **THEN** repository `B` is searched

#### Scenario: Server default used

- **WHEN** the server was started with `--repo A` for indexed repository `A`, its working directory is elsewhere, and `search` is called without `repo`
- **THEN** repository `A` is searched

#### Scenario: Nothing indexed

- **WHEN** a tool is called and neither the starting path nor any ancestor has an index
- **THEN** the call fails with an error naming the starting path and `indexter init`, and no database file is created

#### Scenario: Repository configuration changes apply without restart

- **WHEN** a repository's `indexter.toml` changes `search_limit` between two `search` calls
- **THEN** the second call uses the new limit

### Requirement: Embedders are reused and warmed

The server SHALL create at most one embedder per distinct combination of embedding backend, model, dimension, batch size and token budget, and reuse it for every later call with the same combination. When a default repository resolves at startup (from `--repo` or the working directory), the server SHALL load that repository's embedder and embed one text in the background without delaying the session handshake; a warm-up failure SHALL be written to stderr and SHALL NOT stop the server. With no repository resolvable at startup, no embedder SHALL be created until a call needs one.

#### Scenario: One embedder across calls

- **WHEN** two `search` calls and a `neighbors` call target repositories with identical embedding settings
- **THEN** exactly one embedder is created

#### Scenario: Different model, different embedder

- **WHEN** two calls target repositories configured with different embedding models
- **THEN** two embedders are created

#### Scenario: Warm-up at startup

- **WHEN** the server starts with an indexed default repository
- **THEN** its embedder is created and has embedded a text before any tool call arrives, and the handshake did not wait for it

#### Scenario: Warm-up failure

- **WHEN** warm-up raises an embedding error
- **THEN** the error is written to stderr, the server keeps serving, and the next tool call reports the error as a tool error

#### Scenario: No warm-up without a repository

- **WHEN** the server starts in a directory with no index and no `--repo`
- **THEN** no embedder is created

### Requirement: Tool calls are serialized

The server SHALL run at most one tool call's repository resolution, sync, retrieval and rendering at a time, including warm-up; concurrent calls SHALL wait and then run in turn.

#### Scenario: Concurrent searches

- **WHEN** two `search` calls on the same repository arrive concurrently
- **THEN** both succeed, and the second's sync runs after the first call has finished

### Requirement: Failures are tool errors an agent can act on

Search errors, neighbors errors, configuration errors, database errors and embedding errors raised by a tool call SHALL be returned to the client as tool errors carrying the error's message.

#### Scenario: Invalid filter

- **WHEN** `search` is called with `kind="func"`
- **THEN** the client receives a tool error whose text names the `kind` filter, the value `func` and the valid kinds

#### Scenario: Invalid configuration

- **WHEN** the resolved repository's `indexter.toml` contains an unknown setting
- **THEN** the client receives a tool error naming the setting and the file

#### Scenario: Unknown node

- **WHEN** `neighbors` is called with a node ID that is not in the index
- **THEN** the client receives a tool error saying the node was not found
