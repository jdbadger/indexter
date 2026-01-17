"""
MCP prompt implementations for Indexter.

Prompts provide reusable templates for common agent workflows.
"""

SEARCH_WORKFLOW_PROMPT = """\
# Indexter Code Search Workflow

When searching code in a repository using Indexter:

1. **List available repositories** using the `list_repositories` tool to see which
   repositories are configured and available for searching.

2. **Use filters effectively** - The `search_repository` tool supports filters:
   - `document_path`: Limit search to a specific directory (use trailing `/` for prefix match)
   - `language`: Filter by language (e.g., 'python', 'javascript')  
   - `node_type`: Filter by code structure ('function', 'class', 'method')
   - `node_name`: Filter by specific symbol name
   - `parent_scope`: Filter by parent scope (e.g., 'AuthHandler' to find methods within that class)
   - `has_documentation`: Find documented or undocumented code
   - `limit`: Specify the maximum number of results to return (defaults to 10)

3. **Get repository details** - Use the `get_repository` tool to get metadata
   for a specific repository, including indexing status and document counts.

4. **Handle errors** - If a repo is not found, check available repos with the `list_repositories` tool.

Note: The `search_repository` tool automatically ensures the 
repository index is up to date before searching.

## Example Workflow

```
# 1. Check available repos
repos = call_tool("list_repositories")

# 2. Get details for a specific repo
repo_info = call_tool("get_repository", name="my-repo")

# 3. Search with filters
results = call_tool("search_repository", 
    name="my-repo",
    query="authentication middleware",
    language="python",
    node_type="function"
)

# 4. Search within a specific class
methods = call_tool("search_repository",
    name="my-repo", 
    query="validation logic",
    parent_scope="UserValidator",
    node_type="method"
)
```
"""


def get_search_workflow() -> str:
    """Get the search workflow prompt template."""
    return SEARCH_WORKFLOW_PROMPT
