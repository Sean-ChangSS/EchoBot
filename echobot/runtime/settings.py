from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeSettings:
    delegated_ack_enabled: bool | None = None
    extra_values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeSettings":
        extra_values = dict(data)
        raw_value = extra_values.pop("delegated_ack_enabled", None)
        if raw_value is None:
            delegated_ack_enabled = None
        elif isinstance(raw_value, bool):
            delegated_ack_enabled = raw_value
        else:
            raise ValueError("delegated_ack_enabled must be a boolean")

        return cls(
            delegated_ack_enabled=delegated_ack_enabled,
            extra_values=extra_values,
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra_values)
        data.update(
            {
                "delegated_ack_enabled": self.delegated_ack_enabled,
            }
        )
        return data

    def get_named_value(self, name: str) -> Any:
        if name == "delegated_ack_enabled":
            return self.delegated_ack_enabled
        raise KeyError(name)

    def set_named_value(self, name: str, value: Any) -> None:
        if name != "delegated_ack_enabled":
            raise KeyError(name)
        if value is not None and not isinstance(value, bool):
            raise ValueError("delegated_ack_enabled must be a boolean")
        self.delegated_ack_enabled = value


class RuntimeSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            return RuntimeSettings()

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Runtime settings file must contain a JSON object")
        return RuntimeSettings.from_dict(payload)

    def save(self, settings: RuntimeSettings) -> RuntimeSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return settings

    def update_named_value(self, name: str, value: Any) -> RuntimeSettings:
        settings = self.load()
        settings.set_named_value(name, value)
        return self.save(settings)
