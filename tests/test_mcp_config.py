import os

import pytest
from mcp import StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters

import mcp_config


def write(tmp_path, text):
    path = tmp_path / "mcp.toml"
    path.write_text(text)
    return path


def test_command_server_runs_over_stdio_with_its_args_and_env(tmp_path):
    servers = mcp_config.load(write(tmp_path, """
[servers.knowledge]
command = "shed"
args = ["serve", "--with", "knowledge"]
env = { MIND_DB = "rollupeurope" }
"""))
    server = servers["knowledge"]
    assert isinstance(server, StdioServerParameters)
    assert (server.command, server.args, server.env) == ("shed", ["serve", "--with", "knowledge"],
                                                         {"MIND_DB": "rollupeurope"})


def test_command_path_expands_home(tmp_path):
    servers = mcp_config.load(write(tmp_path, '[servers.a]\ncommand = "~/bin/tool"\n'))
    assert servers["a"].command == os.path.expanduser("~/bin/tool")


def test_url_server_sends_the_token_file_as_a_bearer_header(tmp_path):
    (tmp_path / "token").write_text("s3cret\n")
    servers = mcp_config.load(write(tmp_path, f"""
[servers.remote]
url = "http://127.0.0.1:8765/mcp"
token_file = "{tmp_path / 'token'}"
"""))
    server = servers["remote"]
    assert isinstance(server, StreamableHttpParameters)
    assert server.url == "http://127.0.0.1:8765/mcp"
    assert server.headers == {"Authorization": "Bearer s3cret"}


def test_url_server_without_token_sends_no_auth(tmp_path):
    servers = mcp_config.load(write(tmp_path, '[servers.open]\nurl = "http://127.0.0.1:9/mcp"\n'))
    assert servers["open"].headers is None


def test_every_server_in_the_file_is_returned(tmp_path):
    servers = mcp_config.load(write(tmp_path, """
[servers.one]
command = "a"
[servers.two]
url = "http://127.0.0.1:9/mcp"
"""))
    assert set(servers) == {"one", "two"}


@pytest.mark.parametrize("entry", [
    'command = "a"\nurl = "http://x/mcp"',        # both
    'args = ["x"]',                               # neither
    'command = "a"\ntoken_file = "t"',            # http key on a stdio server
    'url = "http://x/mcp"\nargs = ["x"]',         # stdio key on an http server
    'command = "a"\nargs = "not a list"',
    'command = "a"\nenv = { N = 1 }',
])
def test_malformed_server_is_rejected(tmp_path, entry):
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(write(tmp_path, f"[servers.bad]\n{entry}\n"))


def test_missing_or_empty_token_file_is_rejected(tmp_path):
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(write(tmp_path, f'[servers.r]\nurl = "http://x/mcp"\ntoken_file = "{tmp_path}/none"\n'))
    (tmp_path / "empty").write_text("\n")
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(write(tmp_path, f'[servers.r]\nurl = "http://x/mcp"\ntoken_file = "{tmp_path}/empty"\n'))


def test_unreadable_or_invalid_file_is_rejected(tmp_path):
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(tmp_path / "missing.toml")
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(write(tmp_path, "[servers.a\n"))
    with pytest.raises(mcp_config.ConfigError):
        mcp_config.load(write(tmp_path, '[server.a]\ncommand = "a"\n'))  # misspelled table
