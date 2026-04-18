"""Hyperparameter type definitions: Float, Int, Choice."""

import math
from dataclasses import asdict, dataclass
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
        self._children: dict[Any, list["Param"]] = {}

    def when(self, value: Any, *children: "Param") -> "Float":
        """Declare children that are only active when this param equals ``value``.

        Args:
            value: The parent value that activates these children.
            *children: One or more ``Param`` instances that are required
                when this param's value equals ``value``.

        Returns:
            ``self``, for chaining.
        """
        self._children.setdefault(value, []).extend(children)
        return self

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
        for value, children in self._children.items():
            names = ", ".join(c.name for c in children)
            desc += f" When {value!r}: also include {names}."
        return {"type": "number", "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (float): {self.description}"]
        if self.bounds is not None:
            parts.append(f"bounds={self.bounds}")
        if self.initial is not None:
            parts.append(f"initial={self.initial}")
        base = ", ".join(parts)
        if not self._children:
            return base
        lines = [base]
        for value, children in self._children.items():
            for child in children:
                lines.append(f"  when {value!r}: {child.summary()}")
        return "\n".join(lines)


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
        self._children: dict[Any, list["Param"]] = {}

    def when(self, value: Any, *children: "Param") -> "Int":
        """Declare children that are only active when this param equals ``value``.

        Args:
            value: The parent value that activates these children.
            *children: One or more ``Param`` instances that are required
                when this param's value equals ``value``.

        Returns:
            ``self``, for chaining.
        """
        self._children.setdefault(value, []).extend(children)
        return self

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
        for value, children in self._children.items():
            names = ", ".join(c.name for c in children)
            desc += f" When {value!r}: also include {names}."
        return {"type": "integer", "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (int): {self.description}"]
        if self.bounds is not None:
            parts.append(f"bounds={self.bounds}")
        if self.initial is not None:
            parts.append(f"initial={self.initial}")
        base = ", ".join(parts)
        if not self._children:
            return base
        lines = [base]
        for value, children in self._children.items():
            for child in children:
                lines.append(f"  when {value!r}: {child.summary()}")
        return "\n".join(lines)


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
        self._children: dict[Any, list["Param"]] = {}

    def when(self, value: Any, *children: "Param") -> "Choice":
        """Declare children that are only active when this param equals ``value``.

        Args:
            value: The parent value that activates these children. Should
                be one of ``self.options``.
            *children: One or more ``Param`` instances that are required
                when this param's value equals ``value``.

        Returns:
            ``self``, for chaining.
        """
        self._children.setdefault(value, []).extend(children)
        return self

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
        for value, children in self._children.items():
            names = ", ".join(c.name for c in children)
            desc += f" When {value!r}: also include {names}."
        return {"enum": list(self.options), "description": desc}

    def summary(self) -> str:
        parts = [f"{self.name} (choice): {self.description}", f"options={self.options}"]
        if self.initial is not None:
            parts.append(f"initial={self.initial!r}")
        base = ", ".join(parts)
        if not self._children:
            return base
        lines = [base]
        for value, children in self._children.items():
            for child in children:
                lines.append(f"  when {value!r}: {child.summary()}")
        return "\n".join(lines)


Param = Float | Int | Choice


_PARAM_TYPES: dict[str, type] = {"Float": Float, "Int": Int, "Choice": Choice}


def param_to_dict(p: "Param") -> dict:
    """Serialize a Param to a JSON-safe dict, preserving its concrete type.

    Args:
        p: A ``Float``, ``Int``, or ``Choice`` instance.

    Returns:
        A dict containing all of ``p``'s dataclass fields plus a
        ``"__type__"`` key holding the concrete class name, and an optional
        ``"_children"`` key for conditional children. Suitable for
        ``json.dumps``.
    """
    d = asdict(p)
    d["__type__"] = type(p).__name__
    if p._children:
        d["_children"] = [
            [value, [param_to_dict(c) for c in children]]
            for value, children in p._children.items()
        ]
    return d


def param_from_dict(d: dict) -> "Param":
    """Reconstruct a Param from ``param_to_dict()`` output.

    Args:
        d: A dict produced by ``param_to_dict()``. Must contain a
            ``"__type__"`` key naming one of the known Param subclasses.

    Returns:
        The reconstructed ``Float``, ``Int``, or ``Choice`` instance,
        including any conditional children.

    Raises:
        KeyError: If ``"__type__"`` is missing or unknown.
    """
    d = dict(d)
    type_name = d.pop("__type__")
    children_data = d.pop("_children", [])
    cls = _PARAM_TYPES[type_name]
    if "bounds" in d and d["bounds"] is not None:
        d["bounds"] = tuple(d["bounds"])
    p = cls(**d)
    for value, child_dicts in children_data:
        p.when(value, *[param_from_dict(c) for c in child_dicts])
    return p
