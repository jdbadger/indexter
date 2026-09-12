import pytest

from indexter.config import ConfigError, Settings, load_settings


@pytest.fixture
def global_config_dir(monkeypatch, tmp_path):
    cfg_dir = tmp_path / "config-home"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    return cfg_dir / "indexter"


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return d


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestLayering:
    def test_defaults_with_no_files(self, global_config_dir, repo):
        settings = load_settings(repo=repo)
        assert settings == Settings()

    def test_repo_overrides_global(self, global_config_dir, repo):
        write(global_config_dir / "config.toml", 'embedding_model = "global-model"\n')
        write(repo / "indexter.toml", 'embedding_model = "repo-model"\n')
        settings = load_settings(repo=repo)
        assert settings.embedding_model == "repo-model"

    def test_partial_override_preserves_other_keys(self, global_config_dir, repo):
        write(repo / "indexter.toml", "embed_batch_size = 64\n")
        settings = load_settings(repo=repo)
        assert settings.embed_batch_size == 64
        assert settings.embedding_dim == Settings().embedding_dim

    def test_explicit_args_win(self, global_config_dir, repo):
        write(global_config_dir / "config.toml", "embed_batch_size = 16\n")
        write(repo / "indexter.toml", "embed_batch_size = 32\n")
        settings = load_settings(repo=repo, embed_batch_size=99)
        assert settings.embed_batch_size == 99


class TestGlobalConfig:
    def test_missing_global_is_fine(self, global_config_dir):
        settings = load_settings()
        assert settings == Settings()

    def test_malformed_global_errors(self, global_config_dir):
        write(global_config_dir / "config.toml", "not valid toml [[[")
        with pytest.raises(ConfigError, match="config.toml"):
            load_settings()


class TestRepoConfig:
    def test_dedicated_file_used(self, global_config_dir, repo):
        write(repo / "indexter.toml", 'embedding_model = "from-dedicated"\n')
        settings = load_settings(repo=repo)
        assert settings.embedding_model == "from-dedicated"

    def test_pyproject_fallback(self, global_config_dir, repo):
        write(
            repo / "pyproject.toml",
            '[tool.indexter]\nembedding_model = "from-pyproject"\n',
        )
        settings = load_settings(repo=repo)
        assert settings.embedding_model == "from-pyproject"

    def test_both_present_dedicated_wins_with_warning(self, global_config_dir, repo):
        write(repo / "indexter.toml", 'embedding_model = "from-dedicated"\n')
        write(
            repo / "pyproject.toml",
            '[tool.indexter]\nembedding_model = "from-pyproject"\n',
        )
        with pytest.warns(UserWarning, match="indexter.toml"):
            settings = load_settings(repo=repo)
        assert settings.embedding_model == "from-dedicated"

    def test_neither_present(self, global_config_dir, repo):
        settings = load_settings(repo=repo)
        assert settings == Settings()

    def test_no_repo_given(self, global_config_dir):
        settings = load_settings()
        assert settings == Settings()


class TestValidation:
    def test_unknown_key_rejected(self, global_config_dir, repo):
        write(repo / "indexter.toml", "not_a_real_setting = 1\n")
        with pytest.raises(ConfigError, match="not_a_real_setting"):
            load_settings(repo=repo)

    def test_unknown_key_names_source_file(self, global_config_dir, repo):
        write(repo / "indexter.toml", "bogus = 1\n")
        with pytest.raises(ConfigError, match="indexter.toml"):
            load_settings(repo=repo)

    def test_unknown_key_in_pyproject_fallback_names_pyproject(self, global_config_dir, repo):
        write(repo / "pyproject.toml", "[tool.indexter]\nbogus = 1\n")
        with pytest.raises(ConfigError, match="pyproject.toml"):
            load_settings(repo=repo)

    def test_unknown_key_in_global_config_names_global_file(self, global_config_dir, repo):
        write(global_config_dir / "config.toml", "bogus = 1\n")
        with pytest.raises(ConfigError, match="config.toml"):
            load_settings(repo=repo)

    def test_wrong_type_rejected(self, global_config_dir, repo):
        write(repo / "indexter.toml", 'embed_batch_size = "not a number"\n')
        with pytest.raises(ConfigError, match="embed_batch_size"):
            load_settings(repo=repo)

    def test_unknown_key_from_explicit_override(self, global_config_dir, repo):
        with pytest.raises(ConfigError, match="explicit override"):
            load_settings(repo=repo, not_a_real_setting=1)

    def test_frozen_model_rejects_assignment(self):
        settings = Settings()
        with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on frozen mutation
            settings.embedding_model = "mutated"
