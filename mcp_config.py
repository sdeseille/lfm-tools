"""MCP servers for lfm-server, read from a TOML file.

    [servers.knowledge]              # started by lfm-server, spoken to over stdio
    command = "shed"
    args = ["serve", "--with", "knowledge", "--read-only"]
    env = { MIND_DB = "rollupeurope" }

    [servers.rollupeurope]           # already running, spoken to over HTTP
    url = "http://127.0.0.1:8765/mcp"
    token_file = "~/.config/lfm/rollupeurope.token"

A server has either `command` or `url`. `env` adds to lfm-server's own
environment rather than replacing it. `token_file` holds a bearer token, kept
out of this file so the file can be shared.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client.session_group import ServerParameters, StreamableHttpParameters

STDIO_KEYS = {"command", "args", "env"}
HTTP_KEYS = {"url", "token_file"}


class ConfigError(ValueError):
    pass


def load(path: str | Path) -> dict[str, ServerParameters]:
    file = Path(path).expanduser()
    try:
        raw = tomllib.loads(file.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{file}: {exc}") from None
    unknown = raw.keys() - {"servers"}
    if unknown:
        raise ConfigError(f"{file}: unknown top-level keys {sorted(unknown)}")
    servers = raw.get("servers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"{file}: [servers] must be a table")
    return {name: _server(f"{file}: servers.{name}", entry) for name, entry in servers.items()}


def _server(where: str, entry: object) -> ServerParameters:
    if not isinstance(entry, dict):
        raise ConfigError(f"{where} must be a table")
    if ("command" in entry) == ("url" in entry):
        raise ConfigError(f"{where} needs exactly one of command or url")
    allowed = STDIO_KEYS if "command" in entry else HTTP_KEYS
    unknown = entry.keys() - allowed
    if unknown:
        raise ConfigError(f"{where}: {sorted(unknown)} not allowed here; use {sorted(allowed)}")

    if "command" in entry:
        args = entry.get("args", [])
        env = entry.get("env")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ConfigError(f"{where}.args must be a list of strings")
        if env is not None and not (isinstance(env, dict) and all(isinstance(v, str) for v in env.values())):
            raise ConfigError(f"{where}.env must be a table of strings")
        return StdioServerParameters(command=str(Path(entry["command"]).expanduser()), args=args, env=env)

    headers = None
    if "token_file" in entry:
        token_file = Path(entry["token_file"]).expanduser()
        try:
            token = token_file.read_text().strip()
        except OSError as exc:
            raise ConfigError(f"{where}.token_file: {exc}") from None
        if not token:
            raise ConfigError(f"{where}.token_file {token_file} is empty")
        headers = {"Authorization": f"Bearer {token}"}
    return StreamableHttpParameters(url=entry["url"], headers=headers)
