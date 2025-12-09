"""MCP prompt implementations for Indexter.

Prompts provide reusable templates for common agent workflows.
"""

SEARCH_WORKFLOW_PROMPT = """\
# Indexter Code Search Workflow

When searching code in a repository using Indexter:

1. **List available repositories** using the `repos://list` resource to see
   which repos are configured.

2. **Sync before searching** - Always call `index` before `search` to ensure
   the vector index reflects the current file state. Sync is incremental and
   fast for unchanged files.

3. **Use filters effectively** - The `search` tool supports filters:
   - `file_path`: Limit search to a specific directory (use trailing `/` for prefix match)
   - `language`: Filter by language (e.g., 'python', 'javascript')  
   - `node_type`: Filter by code structure ('function', 'class', 'method')
   - `node_name`: Filter by specific symbol name
   - `has_documentation`: Find documented or undocumented code

4. **Handle errors** - If a repo is not found, check available repos with the list resource.

## Example Workflow

```
# 1. Check available repos
repos = read_resource("repos://list")

# 2. Sync to get latest state
index_result = call_tool("index_repo", name="my-project")

# 3. Search with filters
results = call_tool("search_repo", 
    name="my-project",
    query="authentication middleware",
    language="python",
    node_type="function"
)
```
"""


def get_search_workflow_prompt() -> str:
    """Get the search workflow prompt template."""
    return SEARCH_WORKFLOW_PROMPT
