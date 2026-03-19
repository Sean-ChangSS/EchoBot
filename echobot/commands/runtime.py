from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..runtime.settings import RuntimeSettingsStore
from .parsing import split_action_argument, split_command_parts

if TYPE_CHECKING:
    from ..orchestration import ConversationCoordinator


@dataclass(slots=True)
class RuntimeCommand:
    action: str
    key: str = ""
    value: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeSettingDefinition:
    name: str
    value_hint: str
    description: str


RUNTIME_SETTING_DEFINITIONS: dict[str, RuntimeSettingDefinition] = {
    "delegated_ack_enabled": RuntimeSettingDefinition(
        name="delegated_ack_enabled",
        value_hint="on|off",
        description="Show the task-start tip before background work",
    ),
}


def parse_runtime_command(text: str) -> RuntimeCommand | None:
    command_token, remainder = split_command_parts(text)
    if command_token != "/runtime":
        return None

    if not remainder:
        return RuntimeCommand(action="list")

    action, argument = split_action_argument(
        remainder,
    )

    if action in {"help", "list"}:
        return RuntimeCommand(action=action)
    if action == "get":
        key, _unused = split_action_argument(argument)
        return RuntimeCommand(action="get", key=key.lower())
    if action == "set":
        key, value = split_action_argument(argument)
        return RuntimeCommand(
            action="set",
            key=key.lower(),
            value=value,
        )

    return RuntimeCommand(action="help")


def format_runtime_help() -> str:
    return "\n".join(
        [
            "Runtime commands:",
            "/runtime list - List runtime settings and current values",
            "/runtime get <name> - Show one runtime setting",
            "/runtime set <name> <value> - Update one runtime setting",
            "",
            "Available runtime settings:",
            *[
                (
                    f"{definition.name} <{definition.value_hint}> - "
                    f"{definition.description}"
                )
                for definition in RUNTIME_SETTING_DEFINITIONS.values()
            ],
        ]
    )


async def execute_runtime_command(
    coordinator: "ConversationCoordinator",
    workspace: Path,
    command: RuntimeCommand,
) -> str:
    store = RuntimeSettingsStore(workspace / ".echobot" / "runtime_settings.json")

    if command.action == "help":
        return format_runtime_help()

    if command.action == "list":
        return _format_runtime_settings_list(coordinator)

    if command.action == "get":
        if not command.key:
            return "Usage: /runtime get <name>"
        if command.key not in RUNTIME_SETTING_DEFINITIONS:
            return _format_unknown_runtime_setting(command.key)
        return _format_runtime_setting_line(
            command.key,
            _read_runtime_setting(coordinator, command.key),
        )

    if command.action == "set":
        if not command.key or not command.value:
            return "Usage: /runtime set <name> <value>"
        if command.key not in RUNTIME_SETTING_DEFINITIONS:
            return _format_unknown_runtime_setting(command.key)
        try:
            parsed_value = _parse_runtime_setting_value(command.key, command.value)
        except ValueError as exc:
            return str(exc)
        try:
            await asyncio.to_thread(
                store.update_named_value,
                command.key,
                parsed_value,
            )
        except Exception as exc:
            return f"Failed to save runtime settings: {exc}"

        _apply_runtime_setting(coordinator, command.key, parsed_value)
        return (
            "Updated runtime setting: "
            + _format_runtime_setting_line(
                command.key,
                _read_runtime_setting(coordinator, command.key),
            )
        )

    return format_runtime_help()


def _format_runtime_settings_list(
    coordinator: "ConversationCoordinator",
) -> str:
    lines = ["Runtime settings:"]
    for name in RUNTIME_SETTING_DEFINITIONS:
        lines.append(
            _format_runtime_setting_line(
                name,
                _read_runtime_setting(coordinator, name),
            )
        )
    return "\n".join(lines)


def _format_runtime_setting_line(name: str, value: object) -> str:
    definition = RUNTIME_SETTING_DEFINITIONS[name]
    return f"{definition.name} = {_format_runtime_setting_value(name, value)}"


def _format_runtime_setting_value(name: str, value: object) -> str:
    if name == "delegated_ack_enabled":
        return "on" if bool(value) else "off"
    raise KeyError(name)


def _format_unknown_runtime_setting(name: str) -> str:
    known_names = ", ".join(RUNTIME_SETTING_DEFINITIONS)
    return f"Unknown runtime setting: {name}. Available settings: {known_names}"


def _parse_runtime_setting_value(name: str, raw_value: str) -> object:
    cleaned = str(raw_value or "").strip().lower()
    if name == "delegated_ack_enabled":
        if cleaned in {"on", "true", "enable", "enabled"}:
            return True
        if cleaned in {"off", "false", "disable", "disabled"}:
            return False
        raise ValueError(
            "Invalid value for delegated_ack_enabled. Use on or off."
        )
    raise KeyError(name)


def _read_runtime_setting(
    coordinator: "ConversationCoordinator",
    name: str,
) -> object:
    if name == "delegated_ack_enabled":
        return coordinator.delegated_ack_enabled
    raise KeyError(name)


def _apply_runtime_setting(
    coordinator: "ConversationCoordinator",
    name: str,
    value: object,
) -> None:
    if name == "delegated_ack_enabled":
        coordinator.set_delegated_ack_enabled(bool(value))
        return
    raise KeyError(name)
