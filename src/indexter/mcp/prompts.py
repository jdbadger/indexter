"""
MCP prompt implementations for Indexter.

Prompts provide reusable templates for common agent workflows.
"""

SEARCH_WORKFLOW_PROMPT = """\
# Indexter Code Search Workflow

When searching code in a repository using Indexter:

1. **List available repositories** using the `list_repos` tool to see which
   repositories are configured and available for searching.

2. **Use filters effectively** - The `search_repo` tool supports filters:
   - `file_path`: Limit search to a specific directory (use trailing `/` for prefix match)
   - `language`: Filter by language (e.g., 'python', 'javascript')  
   - `node_type`: Filter by code structure ('function', 'class', 'method')
   - `node_name`: Filter by specific symbol name
   - `has_documentation`: Find documented or undocumented code
   - `limit`: Specify the maximum number of results to return (defaults to 10)

3. **Handle errors** - If a repo is not found, check available repos with the `list_repos` tool.

Note: The `search_repo` tool automatically ensures the 
repository index is up to date before searching.

## Example Workflow

```
# 1. Check available repos
repos = call_tool("list_repos")

# 2. Search with filters
results = call_tool("search_repo", 
    name="my-repo",
    query="authentication middleware",
    language="python",
    node_type="function"
)
```
"""


def get_search_workflow() -> str:
    """Get the search workflow prompt template."""
    return SEARCH_WORKFLOW_PROMPT
