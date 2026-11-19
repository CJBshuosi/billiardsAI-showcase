import pytest

from cueorbit_replay.settings import Settings, guarded_url


@pytest.mark.parametrize(
    "dsn",
    [
        "sqlite:///test",
        "postgresql+asyncpg://a@localhost/product",
        "postgresql+asyncpg://a@localhost/cueorbit_replay_m1bad",
        "postgresql+asyncpg://a@localhost/cueorbit_replay_m1?options=-csearch_path%3Dpublic",
    ],
)
def test_reject_unsafe_database(dsn):
    with pytest.raises(ValueError):
        guarded_url(dsn)


def test_settings_redacts_credentials():
    s = Settings(
        database_url="postgresql+asyncpg://a:secret@localhost/cueorbit_replay_m1_test",
        api_key="key-secret",
    )
    assert "secret" not in repr(s)


def test_missing_dotenv_has_empty_defaults_without_parent_search(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=parent-placeholder\n")
    nested = tmp_path / "project"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env(env_file=nested / ".env", environ={})
    assert settings.database_url == ""
    assert settings.test_database_url == ""
    assert settings.api_key == ""
    assert not settings.live_ready


def test_dotenv_reads_only_supported_settings_without_global_mutation(
    tmp_path, monkeypatch
):
    import os

    for name in (
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
    ):
        monkeypatch.setenv(name, "unchanged-process-placeholder")
    monkeypatch.delenv("CUEORBIT_DOTENV_IGNORED_TEST", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "DATABASE_URL=file-db\n"
        "TEST_DATABASE_URL=file-test-db\n"
        "DEEPSEEK_API_KEY='placeholder with spaces'\n"
        "DEEPSEEK_MODEL=file-model\n"
        "CUEORBIT_DOTENV_IGNORED_TEST=ignored\n"
    )
    settings = Settings.from_env(env_file=path, environ={})
    assert settings.database_url == "file-db"
    assert settings.test_database_url == "file-test-db"
    assert settings.api_key == "placeholder with spaces"
    assert settings.model_name == "file-model"
    assert settings.live_ready
    assert "CUEORBIT_DOTENV_IGNORED_TEST" not in os.environ
    assert all(
        os.environ[name] == "unchanged-process-placeholder"
        for name in (
            "DATABASE_URL",
            "TEST_DATABASE_URL",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_MODEL",
        )
    )
    assert not hasattr(settings, "CUEORBIT_DOTENV_IGNORED_TEST")


@pytest.mark.parametrize("process_value", ["process-placeholder", ""])
def test_process_values_override_file_including_explicit_empty(tmp_path, process_value):
    path = tmp_path / ".env"
    path.write_text(
        "DATABASE_URL=file-db\nTEST_DATABASE_URL=file-test-db\nDEEPSEEK_API_KEY=file-placeholder\nDEEPSEEK_MODEL=file-model\n"
    )
    settings = Settings.from_env(
        env_file=path,
        environ={
            name: process_value
            for name in (
                "DATABASE_URL",
                "TEST_DATABASE_URL",
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_MODEL",
            )
        },
    )
    assert settings.database_url == process_value
    assert settings.test_database_url == process_value
    assert settings.api_key == process_value
    assert settings.model_name == process_value
    assert settings.live_ready is bool(process_value)


def test_default_reads_exact_project_path_independent_of_cwd(tmp_path, monkeypatch):
    from cueorbit_replay import settings as module

    root = tmp_path / "synthetic-project"
    cwd = tmp_path / "unrelated"
    root.mkdir()
    cwd.mkdir()
    (root / ".env").write_text("DEEPSEEK_MODEL=project-model\n")
    (cwd / ".env").write_text("DEEPSEEK_MODEL=wrong-cwd-model\n")
    monkeypatch.setattr(module, "PROJECT_ENV_FILE", root / ".env")
    monkeypatch.chdir(cwd)
    settings = Settings.from_env(environ={})
    assert settings.model_name == "project-model"


def test_project_path_is_anchored_to_this_source_checkout():
    from pathlib import Path

    from cueorbit_replay.settings import PROJECT_ENV_FILE

    assert PROJECT_ENV_FILE == Path(__file__).resolve().parents[2] / ".env"


def test_disable_file_and_constructor_do_not_read_actual_project_configuration(
    monkeypatch,
):
    from cueorbit_replay import settings as module

    def unexpected_read(*args, **kwargs):
        raise AssertionError("configuration file should not be read")

    monkeypatch.setattr(module, "dotenv_values", unexpected_read)
    assert Settings.from_env(env_file=None, environ={}).api_key == ""
    assert Settings(api_key="injected-placeholder").api_key == "injected-placeholder"


def test_file_empty_and_bare_values_do_not_enable_live_mode(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY\nDEEPSEEK_MODEL=\n")
    settings = Settings.from_env(env_file=path, environ={})
    assert settings.api_key == ""
    assert settings.model_name == ""
    assert not settings.live_ready


def test_configuration_repr_redacts_all_loaded_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "DATABASE_URL=hidden-db-marker\nTEST_DATABASE_URL=hidden-test-marker\nDEEPSEEK_API_KEY=hidden-key-marker\nDEEPSEEK_MODEL=hidden-model-marker\n"
    )
    representation = repr(Settings.from_env(env_file=path, environ={}))
    assert "hidden-" not in representation


def test_dotenv_values_are_literal_and_do_not_expand_other_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUEORBIT_UNRELATED_TEST", "must-not-be-expanded")
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY='literal-${CUEORBIT_UNRELATED_TEST}'\n")
    settings = Settings.from_env(env_file=path, environ={})
    assert settings.api_key == "literal-${CUEORBIT_UNRELATED_TEST}"


def test_default_process_environment_is_used_when_mapping_not_injected(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=file-placeholder\nDEEPSEEK_MODEL=file-model\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_MODEL", "process-model")
    settings = Settings.from_env(env_file=path)
    assert settings.api_key == ""
    assert settings.model_name == "process-model"
    assert not settings.live_ready
