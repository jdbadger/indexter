from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """MCP server settings."""

    model_config = SettingsConfigDict(env_prefix="INDEXTER_MCP_")

    host: str = "localhost"
    port: int = 8765
    # Number of results to return from search
    default_top_k: int = 10
