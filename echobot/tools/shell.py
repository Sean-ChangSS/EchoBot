from __future__ import annotations

import asyncio
import locale
import os
import re
import shlex
from pathlib import Path
from typing import Any

from .base import ToolOutput
from .filesystem import WorkspaceTool

# Allowlist of safe command prefixes the LLM agent may execute.
# Add entries here as needed. Everything else is blocked.
ALLOWED_COMMANDS: set[str] = {
    "ls", "cat", "head", "tail", "wc", "find", "grep", "echo",
    "pwd", "date", "whoami", "env", "printenv",
    "python", "python3", "pip", "pip3",
    "node", "npm", "npx",
    "git", "diff", "patch",
    "curl", "wget",
    "tar", "unzip", "gzip",
    "mkdir", "cp", "mv", "touch",
}

# Patterns that indicate shell injection / chaining attempts.
_DANGEROUS_PATTERNS: re.Pattern[str] = re.compile(
    r"[;|&`$]|\b(sudo|su|rm|chmod|chown|kill|pkill|shutdown|reboot|mkfs|dd|nc|ncat"
    r"|netcat|telnet|ssh|scp|sftp|wget\s+-O\s*/|curl\s+.*\|\s*sh|eval|exec)\b"
)


def _validate_command(command: str) -> None:
    """Raise ValueError if *command* is not on the allowlist or looks dangerous."""
    if _DANGEROUS_PATTERNS.search(command):
        raise ValueError(
            f"Command blocked: contains a disallowed operator or keyword. "
            f"Only simple, single commands from the allowlist are permitted."
        )

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if not tokens:
        raise ValueError("Empty command")

    base_cmd = Path(tokens[0]).name  # handle /usr/bin/python -> python
    if base_cmd not in ALLOWED_COMMANDS:
        raise ValueError(
            f"Command '{base_cmd}' is not in the allowed command list. "
            f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )


class CommandExecutionTool(WorkspaceTool):
    name = "run_shell_command"
    description = "Run a shell command in the workspace and return stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            },
            "workdir": {
                "type": "string",
                "description": "Relative working directory inside the workspace.",
                "default": ".",
            },
            "timeout": {
                "type": "number",
                "description": "Command timeout in seconds.",
                "default": 20,
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Maximum characters kept for stdout and stderr.",
                "default": 4000,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    async def run(self, arguments: dict[str, Any]) -> ToolOutput:
        command = str(arguments.get("command", "")).strip()
        if not command:
            raise ValueError("command is required")

        _validate_command(command)

        relative_workdir = str(arguments.get("workdir", ".")).strip() or "."
        timeout = _read_positive_float(arguments.get("timeout", 20), name="timeout")
        max_output_chars = _read_positive_int(
            arguments.get("max_output_chars", 4000),
            name="max_output_chars",
        )

        workdir = self._resolve_workspace_path(relative_workdir)
        if not workdir.exists():
            raise ValueError(f"Path does not exist: {relative_workdir}")
        if not workdir.is_dir():
            raise ValueError(f"Path is not a directory: {relative_workdir}")

        return await self._run_command(
            command,
            workdir,
            relative_workdir,
            timeout,
            max_output_chars,
        )

    async def _run_command(
        self,
        command: str,
        workdir: Path,
        relative_workdir: str,
        timeout: float,
        max_output_chars: int,
    ) -> dict[str, Any]:
        shell_command = _build_shell_command(command)
        process = await asyncio.create_subprocess_exec(
            *shell_command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout} seconds") from exc

        stdout_text = _decode_command_output(stdout_bytes)
        stderr_text = _decode_command_output(stderr_bytes)
        stdout, stdout_truncated = _truncate_text(stdout_text, max_output_chars)
        stderr, stderr_truncated = _truncate_text(stderr_text, max_output_chars)

        return {
            "command": command,
            "workdir": relative_workdir,
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }


def _read_positive_int(value: Any, *, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")

    return number


def _read_positive_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc

    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")

    return number


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    return text[:max_chars], True


def _decode_command_output(raw_bytes: bytes) -> str:
    if not raw_bytes:
        return ""

    preferred_encoding = locale.getpreferredencoding(False) or "utf-8"
    candidate_encodings = ["utf-8"]
    if preferred_encoding.lower() not in {"utf-8", "utf_8"}:
        candidate_encodings.append(preferred_encoding)

    for encoding in candidate_encodings:
        try:
            return raw_bytes.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue

    try:
        return raw_bytes.decode(preferred_encoding, errors="replace")
    except LookupError:
        return raw_bytes.decode("utf-8", errors="replace")


def _build_shell_command(command: str) -> list[str]:
    if os.name == "nt":
        return ["powershell.exe", "-NoProfile", "-Command", command]

    return ["/bin/sh", "-lc", command]
