"""Hyperparameter type definitions: Float, Int, Choice."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Float:
    name: str
    description: str
    bounds: tuple[float, float] | None = None
    initial: float | None = None

    def __post_init__(self):
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo >= hi:
                raise ValueError(
                    f"Float '{self.name}': bounds require lo < hi, got ({lo}, {hi})"
                )
        if self.initial is not None and self.bounds is not None:
            lo, hi = self.bounds
            if not (lo <= self.initial <= hi):
                raise ValueError(
                    f"Float '{self.name}': initial {self.initial} outside bounds {self.bounds}"
                )

    def validate(self, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Float '{self.name}': bool not accepted, got {value!r}")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Float '{self.name}': expected number, got {type(value).__name__}: {value!r}"
            )
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"Float '{self.name}': NaN not accepted")
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (lo <= value <= hi):
                raise ValueError(
                    f"Float '{self.name}': {value} outside bounds {self.bounds}"
                )
        return value

    def to_schema(self) -> dict:
        desc = self.description
        if self.bounds is not None:
            desc += f" Range: [{self.bounds[0]}, {self.bounds[1]}]."
        if self.initial is not None:
            desc += f" Suggested starting point: {self.initial}."
        return {"type": "number", "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (float): {self.description}"]
        if self.bounds is not None:
            parts.append(f"bounds={self.bounds}")
        if self.initial is not None:
            parts.append(f"initial={self.initial}")
        return ", ".join(parts)


@dataclass
class Int:
    name: str
    description: str
    bounds: tuple[int, int] | None = None
    initial: int | None = None

    def __post_init__(self):
        if self.bounds is not None:
            lo, hi = self.bounds
            if isinstance(lo, bool) or isinstance(hi, bool) or not (
                isinstance(lo, int) and isinstance(hi, int)
            ):
                raise ValueError(
                    f"Int '{self.name}': bounds must be integers, got ({lo!r}, {hi!r})"
                )
            if lo >= hi:
                raise ValueError(
                    f"Int '{self.name}': bounds require lo < hi, got ({lo}, {hi})"
                )
        if self.initial is not None:
            if isinstance(self.initial, bool) or not isinstance(self.initial, int):
                raise ValueError(
                    f"Int '{self.name}': initial must be int, got {self.initial!r}"
                )
            if self.bounds is not None:
                lo, hi = self.bounds
                if not (lo <= self.initial <= hi):
                    raise ValueError(
                        f"Int '{self.name}': initial {self.initial} outside bounds {self.bounds}"
                    )

    def validate(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Int '{self.name}': bool not accepted, got {value!r}")
        if isinstance(value, float):
            if value.is_integer():
                value = int(value)
            else:
                raise ValueError(
                    f"Int '{self.name}': float {value} is not integer-valued"
                )
        elif not isinstance(value, int):
            raise ValueError(
                f"Int '{self.name}': expected integer, got {type(value).__name__}: {value!r}"
            )
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (lo <= value <= hi):
                raise ValueError(
                    f"Int '{self.name}': {value} outside bounds {self.bounds}"
                )
        return value

    def to_schema(self) -> dict:
        desc = self.description
        if self.bounds is not None:
            desc += f" Range: [{self.bounds[0]}, {self.bounds[1]}]."
        if self.initial is not None:
            desc += f" Suggested starting point: {self.initial}."
        return {"type": "integer", "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (int): {self.description}"]
        if self.bounds is not None:
            parts.append(f"bounds={self.bounds}")
        if self.initial is not None:
            parts.append(f"initial={self.initial}")
        return ", ".join(parts)


@dataclass
class Choice:
    name: str
    description: str
    options: list
    initial: Any = None

    def __post_init__(self):
        if not self.options:
            raise ValueError(f"Choice '{self.name}': must have at least one option")
        if self.initial is not None and self.initial not in self.options:
            raise ValueError(
                f"Choice '{self.name}': initial {self.initial!r} not in options {self.options}"
            )

    def validate(self, value: Any) -> Any:
        if value not in self.options:
            raise ValueError(
                f"Choice '{self.name}': value {value!r} not in options {self.options}"
            )
        return value

    def to_schema(self) -> dict:
        desc = self.description + f" Options: {self.options}."
        if self.initial is not None:
            desc += f" Suggested starting point: {self.initial!r}."
        return {"enum": list(self.options), "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (choice): {self.description}", f"options={self.options}"]
        if self.initial is not None:
            parts.append(f"initial={self.initial!r}")
        return ", ".join(parts)


Param = Float | Int | Choice
