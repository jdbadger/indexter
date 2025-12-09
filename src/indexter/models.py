"""Data models for indexter."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from .config.repo import RepoConfig, RepoFileConfig, load_repos_config, save_repos_config
from .exceptions import (
    InvalidGitRepositoryError,
    RepoExistsError,
    RepoNotFoundError,
    validate_git_repository,
)
from .parsers import get_parser
from .store import store
from .walker import Walker

logger = logging.getLogger(__name__)


class IndexResult(BaseModel):
    """Result of a sync operation."""

    files_synced: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    files_checked: int = 0
    skipped_files: int = 0
    nodes_added: int = 0
    nodes_deleted: int = 0
    nodes_updated: int = 0
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    errors: list[str] = Field(default_factory=list)


class NodeMetadata(BaseModel):
    """Metadata for a parsed node."""

    hash: str  # content hash of the document to support change detection

    # Repo info
    repo_path: str  # absolute path to the repo root

    # Document info
    document_path: str  # path to the document within the repo

    # Node info
    language: str  # programming language
    node_type: str  # e.g., 'function', 'class'
    node_name: str  # name of the function/class/module, etc.
    start_byte: int  # start byte within the document
    end_byte: int  # end byte within the document
    start_line: int  # start line within the document
    end_line: int  # end line within the document
    documentation: str | None = None  # docstring or comments
    parent_scope: str | None = None  # enclosing scope or class
    signature: str | None = None  # function or method signature
    extra: dict[str, str] = Field(
        default_factory=dict
    )  # attributes specific to language sytanx or semantics .e.g. decorators, attributes


class Node(BaseModel):
    """A parsed and chunked node ready for embedding."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    metadata: NodeMetadata


class Document(BaseModel):
    """Document ."""

    path: str
    size_bytes: int
    mtime: float
    content: str
    hash: str


class Repo(BaseModel):
    repo_config: RepoConfig

    @computed_field
    @property
    def name(self) -> str:
        """Name of the repository."""
        return self.repo_config.name

    @computed_field
    @property
    def path(self) -> str:
        """Absolute path to the repository root."""
        return str(self.repo_config.path)

    @computed_field
    @property
    def collection_name(self) -> str:
        """Name of the VectorStore collection for this repo."""
        return self.repo_config.collection_name

    @classmethod
    async def init(cls, repo_path: Path) -> Repo:
        """Add a new repository to the configuration.

        Args:
            repo_path: Path to the git repository root.

        Returns:
            The added Repo instance.

        Raises:
            RepoExistsError: If a repo with the same name already exists.
            ValueError: If the path is not a git repository.
        """
        repo_configs = await load_repos_config()
        resolved_path = repo_path.resolve()

        # Validate that the path is a git repository
        try:
            validate_git_repository(resolved_path)
        except InvalidGitRepositoryError as e:
            raise ValueError(f"{resolved_path} is not a git repository") from e

        # Create new config to get the derived name
        repo_config = RepoConfig(path=resolved_path)

        # Check if name already exists
        for existing_config in repo_configs:
            if existing_config.name == repo_config.name:
                if existing_config.path.resolve() == resolved_path:
                    # Same repo, already configured
                    logger.info(f"Repository already configured: {repo_config.name}")
                    return cls(repo_config=existing_config)
                else:
                    # Different repo with same name
                    raise RepoExistsError(
                        f"A repository named '{repo_config.name}' already exists "
                        f"at {existing_config.path}. Rename the directory to use a unique name."
                    )

        repo_configs.append(repo_config)
        await save_repos_config(repo_configs)

        logger.info(f"Added repository: {repo_config.name} ({resolved_path})")
        return cls(repo_config=repo_config)

    @classmethod
    async def get(cls, name: str) -> Repo:
        """Get a repository by name.

        Args:
            name: The repository name (directory name containing .git).

        Returns:
            The Repo instance.

        Raises:
            RepoNotFoundError: If no repo with the given name exists.
        """
        repo_configs = await load_repos_config()
        for repo_config in repo_configs:
            if repo_config.name == name:
                return cls(repo_config=repo_config)
        raise RepoNotFoundError(f"Repository not found: {name}")

    @classmethod
    async def list(cls) -> list[Repo]:
        """List all repositories in the configuration."""
        repo_configs = await load_repos_config()
        return [cls(repo_config=rc) for rc in repo_configs]

    @classmethod
    async def remove(cls, name: str) -> bool:
        """Remove a repository by name.

        Args:
            name: The repository name to remove.

        Returns:
            True if the repository was removed.

        Raises:
            RepoNotFoundError: If no repo with the given name exists.
        """
        repo = await cls.get(name)

        # Delete collection from store
        await store.delete_collection(repo.collection_name)

        # Remove from config
        repo_configs = await load_repos_config()
        new_repo_configs = [r for r in repo_configs if r.name != name]
        await save_repos_config(new_repo_configs)
        logger.info(f"Removed repository: {name}")
        return True

    async def get_document_hashes(self) -> dict[str, str]:
        """Get document content hashes for the repository."""
        document_hashes: dict[str, str] = {}
        walker = await Walker.create(Path(self.path))
        async for doc in walker.walk():
            doc = Document.model_validate(doc)
            document_hashes[doc.path] = doc.hash
        return document_hashes

    async def search(
        self,
        query: str,
        limit: int = 10,
        file_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        has_documentation: bool | None = None,
    ) -> list[dict]:
        """Search nodes in the repository's collection using semantic search.

        Args:
            query: Search query text.
            limit: Maximum number of results to return (default: 10).
            file_path: Filter by file path (exact match or prefix with trailing /).
            language: Filter by programming language.
            node_type: Filter by node type (e.g., 'function', 'class').
            node_name: Filter by node name.
            has_documentation: Filter by documentation presence (True/False).

        Returns:
            List of search results with scores and metadata.
        """
        return await store.search(
            collection_name=self.collection_name,
            query=query,
            limit=limit,
            file_path=file_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            has_documentation=has_documentation,
        )

    async def status(self) -> dict:
        """Get the current status of the repository in the store.

        Returns:
            Dict with counts of documents and nodes indexed.
        """
        local_hashes = list((await self.get_document_hashes()).values())
        stored_hashes = list((await store.get_document_hashes(self.collection_name)).values())
        num_documents = len(stored_hashes)
        num_documents_stale = len([h for h in stored_hashes if h not in local_hashes])
        num_nodes = await store.count_nodes(self.collection_name)
        return {
            "repository": self.name,
            "path": self.path,
            "nodes_indexed": num_nodes,
            "documents_indexed": num_documents,
            "documents_indexed_stale": num_documents_stale,
        }

    async def index(self, full: bool = False) -> IndexResult:
        """Index the repository: parse, chunk, and store nodes.

        Performs incremental indexing by comparing document content hashes:
        - New files: parse and add nodes
        - Modified files: delete old nodes, parse and add new nodes
        - Deleted files: remove nodes from store
        - Unchanged files: skip

        Returns:
            IndexResult with counts of files/nodes processed and any errors.
        """
        result = IndexResult()

        # Load per-repo configuration
        repo_config = await RepoFileConfig.from_repo(Path(self.path))
        max_sync_files = repo_config.max_sync_files
        upsert_batch_size = repo_config.upsert_batch_size

        # On full sync, recreate the collection
        if full:
            await store.delete_collection(self.collection_name)
            logger.info(f"Performing full sync for repository: {self.name}")

        # Ensure collection exists
        await store.ensure_collection(self.collection_name)

        # Initialize walker
        walker = await Walker.create(Path(self.path))

        # Get stored document hashes for change detection
        stored_hashes = await store.get_document_hashes(self.collection_name)
        stored_paths = set(stored_hashes.keys())

        # Track what we've walked and what needs processing
        walked_paths: set[str] = set()
        files_to_process: list[dict] = []  # (doc_dict, is_new)

        # Walk the repository and identify changes
        async for doc in walker.walk():
            doc = Document.model_validate(doc)
            result.files_checked += 1
            walked_paths.add(doc.path)

            stored_hash = stored_hashes.get(doc.path)

            if stored_hash is None:
                # New file
                files_to_process.append({"doc": doc, "is_new": True})
            elif stored_hash != doc.hash:
                # Modified file
                files_to_process.append({"doc": doc, "is_new": False})
            # else: unchanged, skip

        # Identify deleted files
        deleted_paths = list(stored_paths - walked_paths)

        # Respect max_sync_files limit
        if len(files_to_process) > max_sync_files:
            result.skipped_files = len(files_to_process) - max_sync_files
            files_to_process = files_to_process[:max_sync_files]
            logger.warning(
                f"Sync limited to {max_sync_files} files, skipping {result.skipped_files} files"
            )

        # Delete nodes for modified files (before re-adding)
        if modified_paths := [f["doc"].path for f in files_to_process if not f["is_new"]]:
            await store.delete_by_document_paths(self.collection_name, modified_paths)

        # Parse and upsert nodes in batches
        pending_nodes: list[Node] = []

        for file_info in files_to_process:
            doc = file_info["doc"]
            is_new = file_info["is_new"]

            try:
                parser = get_parser(doc.path)
                if parser is None:
                    continue

                # Parse document into nodes
                file_nodes: list[Node] = []
                for content, metadata in parser.parse(doc.content):
                    node = Node(
                        content=content,
                        metadata=NodeMetadata(
                            hash=doc.hash,
                            repo_path=self.path,
                            document_path=doc.path,
                            language=metadata.get("language", "unknown"),
                            node_type=metadata.get("node_type", "unknown"),
                            node_name=metadata.get("node_name") or "",
                            start_byte=metadata.get("start_byte", 0),
                            end_byte=metadata.get("end_byte", 0),
                            start_line=metadata.get("start_line", 0),
                            end_line=metadata.get("end_line", 0),
                            documentation=metadata.get("documentation"),
                            parent_scope=metadata.get("parent_scope"),
                            signature=metadata.get("signature"),
                            extra=metadata.get("extra", {}),
                        ),
                    )
                    file_nodes.append(node)

                if file_nodes:
                    pending_nodes.extend(file_nodes)
                    result.files_synced.append(doc.path)

                    if is_new:
                        result.nodes_added += len(file_nodes)
                    else:
                        result.nodes_updated += len(file_nodes)

                    # Batch upsert when we have enough nodes
                    if len(pending_nodes) >= upsert_batch_size:
                        await store.upsert_nodes(self.collection_name, pending_nodes)
                        pending_nodes = []

            except Exception as e:
                error_msg = f"Failed to parse {doc.path}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        # Upsert any remaining nodes
        if pending_nodes:
            await store.upsert_nodes(self.collection_name, pending_nodes)

        # Delete nodes for removed files
        if deleted_paths:
            await store.delete_by_document_paths(self.collection_name, deleted_paths)
            result.files_deleted = deleted_paths
            # We don't know exact node count deleted, but track file count
            result.nodes_deleted = len(deleted_paths)  # Approximation

        result.indexed_at = datetime.now(UTC)

        logger.info(
            f"Sync complete for {self.name}: "
            f"{len(result.files_synced)} files synced, "
            f"{len(result.files_deleted)} files deleted, "
            f"+{result.nodes_added} ~{result.nodes_updated} -{result.nodes_deleted} nodes"
        )

        return result
