"""Tests for llmtuna.space — Float, Int, Choice."""

import pytest

import llmtuna as lt


# ============================================================
# Float — construction
# ============================================================

@pytest.mark.parametrize("bounds", [(0.5, 0.5), (1.0, 0.5), (-1.0, -2.0)])
def test_float_construction_rejects_bad_bounds(bounds):
    with pytest.raises(ValueError, match="lo < hi"):
        lt.Float(name="lr", description="learning rate", bounds=bounds)


@pytest.mark.parametrize("initial", [0.5, 5.0])
def test_float_construction_rejects_initial_outside_bounds(initial):
    with pytest.raises(ValueError, match="outside bounds"):
        lt.Float(name="lr", description="learning rate", bounds=(1.0, 2.0), initial=initial)


def test_float_construction_accepts_no_bounds_no_initial():
    p = lt.Float(name="lr", description="learning rate")
    assert p.bounds is None and p.initial is None


def test_float_construction_accepts_bounds_only():
    p = lt.Float(name="lr", description="learning rate", bounds=(1e-6, 1.0))
    assert p.bounds == (1e-6, 1.0)


def test_float_construction_accepts_initial_only():
    p = lt.Float(name="lr", description="learning rate", initial=1e-3)
    assert p.initial == 1e-3


# ============================================================
# Float — validate
# ============================================================

def test_float_validate_accepts_float_in_bounds():
    p = lt.Float(name="lr", description="...", bounds=(0.0, 1.0))
    assert p.validate(value=0.5) == 0.5


def test_float_validate_coerces_int_to_float():
    p = lt.Float(name="lr", description="...", bounds=(0, 10))
    result = p.validate(value=5)
    assert result == 5.0
    assert isinstance(result, float)


def test_float_validate_accepts_exact_bounds():
    p = lt.Float(name="lr", description="...", bounds=(1.0, 2.0))
    assert p.validate(value=1.0) == 1.0
    assert p.validate(value=2.0) == 2.0


def test_float_validate_accepts_any_number_with_no_bounds():
    p = lt.Float(name="lr", description="...")
    assert p.validate(value=-1e9) == -1e9
    assert p.validate(value=1e9) == 1e9


@pytest.mark.parametrize("bad_value", [True, False])
def test_float_validate_rejects_bool(bad_value):
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="bool not accepted"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", ["0.5", None, [0.5], {"v": 0.5}])
def test_float_validate_rejects_non_numbers(bad_value):
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="expected number"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", [0.5, 2.5])
def test_float_validate_rejects_out_of_bounds(bad_value):
    p = lt.Float(name="lr", description="...", bounds=(1.0, 2.0))
    with pytest.raises(ValueError, match="outside bounds"):
        p.validate(value=bad_value)


def test_float_validate_rejects_nan():
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="NaN"):
        p.validate(value=float("nan"))


# ============================================================
# Float — to_schema
# ============================================================

def test_float_to_schema_returns_number_type():
    p = lt.Float(name="lr", description="learning rate")
    assert p.to_schema()["type"] == "number"


def test_float_to_schema_includes_description():
    p = lt.Float(name="lr", description="learning rate")
    assert "learning rate" in p.to_schema()["description"]


# ============================================================
# Int — construction
# ============================================================

@pytest.mark.parametrize("bounds", [(1.5, 10), (1, 10.5), (1.0, 10.0)])
def test_int_construction_rejects_float_bounds(bounds):
    with pytest.raises(ValueError, match="must be integers"):
        lt.Int(name="depth", description="...", bounds=bounds)


@pytest.mark.parametrize("bounds", [(True, 10), (1, False)])
def test_int_construction_rejects_bool_bounds(bounds):
    with pytest.raises(ValueError, match="must be integers"):
        lt.Int(name="depth", description="...", bounds=bounds)


@pytest.mark.parametrize("initial", [True, False, 1.5, "5"])
def test_int_construction_rejects_non_int_initial(initial):
    with pytest.raises(ValueError, match="initial must be int"):
        lt.Int(name="depth", description="...", initial=initial)


def test_int_construction_rejects_initial_outside_bounds():
    with pytest.raises(ValueError, match="outside bounds"):
        lt.Int(name="depth", description="...", bounds=(1, 10), initial=20)


# ============================================================
# Int — validate
# ============================================================

def test_int_validate_accepts_int():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    assert p.validate(value=5) == 5


def test_int_validate_coerces_integer_valued_float():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    result = p.validate(value=5.0)
    assert result == 5
    assert isinstance(result, int)


def test_int_validate_rejects_non_integer_float():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    with pytest.raises(ValueError, match="not integer-valued"):
        p.validate(value=5.5)


@pytest.mark.parametrize("bad_value", [True, False])
def test_int_validate_rejects_bool(bad_value):
    p = lt.Int(name="depth", description="...")
    with pytest.raises(ValueError, match="bool not accepted"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", ["5", None, [5]])
def test_int_validate_rejects_non_numbers(bad_value):
    p = lt.Int(name="depth", description="...")
    with pytest.raises(ValueError, match="expected integer"):
        p.validate(value=bad_value)


def test_int_validate_bounds_inclusive():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    assert p.validate(value=1) == 1
    assert p.validate(value=10) == 10


@pytest.mark.parametrize("bad_value", [0, 11])
def test_int_validate_rejects_out_of_bounds(bad_value):
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    with pytest.raises(ValueError, match="outside bounds"):
        p.validate(value=bad_value)


# ============================================================
# Int — to_schema
# ============================================================

def test_int_to_schema_returns_integer_type():
    p = lt.Int(name="depth", description="...")
    assert p.to_schema()["type"] == "integer"


# ============================================================
# Choice — construction
# ============================================================

def test_choice_construction_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one option"):
        lt.Choice(name="activation", description="...", options=[])


def test_choice_construction_rejects_initial_not_in_options():
    with pytest.raises(ValueError, match="not in options"):
        lt.Choice(
            name="activation",
            description="...",
            options=["relu", "gelu"],
            initial="silu",
        )


def test_choice_construction_accepts_mixed_type_options():
    p = lt.Choice(name="misc", description="...", options=[1, "two", 3.0])
    assert p.options == [1, "two", 3.0]


# ============================================================
# Choice — validate
# ============================================================

def test_choice_validate_accepts_value_in_options():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu", "silu"])
    assert p.validate(value="gelu") == "gelu"


def test_choice_validate_rejects_value_not_in_options():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu"])
    with pytest.raises(ValueError, match="not in options"):
        p.validate(value="tanh")


def test_choice_validate_strict_type_equality():
    """String '5' must not match int 5 — type-strict membership."""
    p = lt.Choice(name="misc", description="...", options=["5", "10"])
    with pytest.raises(ValueError, match="not in options"):
        p.validate(value=5)


# ============================================================
# Choice — to_schema
# ============================================================

def test_choice_to_schema_has_enum_no_type():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu", "silu"])
    schema = p.to_schema()
    assert schema["enum"] == ["relu", "gelu", "silu"]
    assert "type" not in schema


# ============================================================
# Conditional children — when()
# ============================================================

def test_when_returns_self_for_chaining():
    opt = lt.Choice(name="optimizer", description="...", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="lr", bounds=(1e-5, 1e-1))
    assert opt.when("adamw", lr) is opt


def test_when_accumulates_multiple_calls_same_value():
    opt = lt.Choice(name="optimizer", description="...", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="lr")
    beta = lt.Float(name="beta", description="beta")
    opt.when("adamw", lr)
    opt.when("adamw", beta)
    assert len(opt._children["adamw"]) == 2


def test_when_accepts_multiple_children_at_once():
    opt = lt.Choice(name="optimizer", description="...", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="lr")
    beta = lt.Float(name="beta", description="beta")
    opt.when("adamw", lr, beta)
    assert len(opt._children["adamw"]) == 2


def test_when_different_values_have_separate_children():
    opt = lt.Choice(name="optimizer", description="...", options=["adamw", "muon"])
    lr_adamw = lt.Float(name="lr_adamw", description="...")
    lr_muon = lt.Float(name="lr_muon", description="...")
    opt.when("adamw", lr_adamw)
    opt.when("muon", lr_muon)
    assert "adamw" in opt._children
    assert "muon" in opt._children
    assert opt._children["adamw"][0].name == "lr_adamw"
    assert opt._children["muon"][0].name == "lr_muon"


def test_when_on_float_param():
    dropout = lt.Float(name="dropout", description="...", bounds=(0.0, 1.0))
    seed = lt.Int(name="dropout_seed", description="...")
    dropout.when(0.5, seed)
    assert 0.5 in dropout._children


def test_when_on_int_param():
    heads = lt.Int(name="heads", description="...", bounds=(1, 16))
    head_dim = lt.Int(name="head_dim", description="...")
    heads.when(8, head_dim)
    assert 8 in heads._children


def test_no_children_by_default():
    p = lt.Float(name="lr", description="...")
    assert p._children == {}


# ============================================================
# Conditional children — to_schema / summary
# ============================================================

def test_to_schema_includes_conditional_hint():
    opt = lt.Choice(name="optimizer", description="Optimizer", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="learning rate")
    opt.when("adamw", lr)
    desc = opt.to_schema()["description"]
    assert "adamw" in desc
    assert "lr" in desc


def test_to_schema_without_children_unchanged():
    p = lt.Choice(name="act", description="activation", options=["relu", "gelu"])
    desc = p.to_schema()["description"]
    assert "when" not in desc.lower()


def test_summary_includes_indented_children():
    opt = lt.Choice(name="optimizer", description="Optimizer", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="learning rate", bounds=(1e-5, 1e-1))
    opt.when("adamw", lr)
    result = opt.summary()
    assert "when 'adamw'" in result
    assert "lr" in result


def test_summary_without_children_is_single_line():
    p = lt.Choice(name="act", description="activation", options=["relu"])
    assert "\n" not in p.summary()


# ============================================================
# Conditional children — serialization round-trip
# ============================================================

def test_param_to_dict_includes_children():
    from llmtuna.space import param_to_dict
    opt = lt.Choice(name="optimizer", description="...", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="lr", bounds=(1e-5, 1e-1))
    opt.when("adamw", lr)
    d = param_to_dict(opt)
    assert "_children" in d
    assert len(d["_children"]) == 1
    value, child_dicts = d["_children"][0]
    assert value == "adamw"
    assert child_dicts[0]["name"] == "lr"


def test_param_to_dict_no_children_key_when_empty():
    from llmtuna.space import param_to_dict
    p = lt.Float(name="lr", description="lr")
    d = param_to_dict(p)
    assert "_children" not in d


def test_param_round_trip_with_children():
    from llmtuna.space import param_to_dict, param_from_dict
    opt = lt.Choice(name="optimizer", description="Optimizer", options=["adamw", "muon"])
    lr = lt.Float(name="lr", description="lr", bounds=(1e-5, 1e-1))
    beta = lt.Float(name="beta1", description="beta1", bounds=(0.8, 0.99))
    opt.when("adamw", lr, beta)
    opt.when("muon", lr)

    restored = param_from_dict(param_to_dict(opt))
    assert isinstance(restored, lt.Choice)
    assert "adamw" in restored._children
    assert "muon" in restored._children
    assert len(restored._children["adamw"]) == 2
    assert len(restored._children["muon"]) == 1
    assert restored._children["adamw"][0].name == "lr"
    assert isinstance(restored._children["adamw"][0], lt.Float)
    assert restored._children["adamw"][0].bounds == (1e-5, 1e-1)


def test_param_round_trip_no_children():
    from llmtuna.space import param_to_dict, param_from_dict
    p = lt.Float(name="lr", description="lr", bounds=(1e-5, 1.0))
    restored = param_from_dict(param_to_dict(p))
    assert restored._children == {}
    assert restored.bounds == (1e-5, 1.0)
