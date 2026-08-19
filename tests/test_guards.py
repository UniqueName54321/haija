from haija.guards import GuardError, evaluate_guard, evaluate_guards


def _ctx(state, **extra):
    ctx = {"actor": "Alpha", "mark": "X", "params": {}, "state": state, "turn": 1, "now": 0}
    ctx.update(extra)
    return ctx


def test_guard_eq_neq():
    s = {"board": [" ", "X", " "], "score": 5}
    assert evaluate_guard({"op": "eq", "path": "board.1", "value": "X"}, _ctx(s))[0]
    assert not evaluate_guard({"op": "eq", "path": "board.1", "value": "O"}, _ctx(s))[0]
    assert evaluate_guard({"op": "neq", "path": "board.0", "value": "X"}, _ctx(s))[0]


def test_guard_numeric_compare():
    s = {"score": 5}
    assert evaluate_guard({"op": "gt", "path": "score", "value": 3}, _ctx(s))[0]
    assert evaluate_guard({"op": "gte", "path": "score", "value": 5}, _ctx(s))[0]
    assert evaluate_guard({"op": "lt", "path": "score", "value": 10}, _ctx(s))[0]
    assert evaluate_guard({"op": "lte", "path": "score", "value": 5}, _ctx(s))[0]
    # template values render as strings but still compare numerically
    ctx2 = _ctx(s, params={"n": 4})
    assert evaluate_guard({"op": "gt", "path": "score", "value": "{{params.n}}"}, ctx2)[0]


def test_guard_in_and_not_in():
    s = {"card": "b"}
    assert evaluate_guard({"op": "in", "path": "card", "value": ["a", "b"]}, _ctx(s))[0]
    assert evaluate_guard({"op": "not_in", "path": "card", "value": ["x", "y"]}, _ctx(s))[0]


def test_guard_exists():
    s = {"hand": ["a"], "hp": None}
    assert evaluate_guard({"op": "exists", "path": "hand"}, _ctx(s))[0]
    assert evaluate_guard({"op": "exists", "path": "hp"}, _ctx(s))[0]  # key present even if None
    assert not evaluate_guard({"op": "exists", "path": "missing"}, _ctx(s))[0]
    assert evaluate_guard({"op": "not_exists", "path": "missing"}, _ctx(s))[0]
    assert not evaluate_guard({"op": "not_exists", "path": "hand"}, _ctx(s))[0]


def test_guard_exists_supports_value_addressed_list_path():
    state = {"hands": {"Alpha": ["Wild", "Y0"]}}
    ctx = _ctx(state, params={"card": "Y0"})
    guard = {"op": "exists", "path": "hands.{{actor}}.{{params.card}}"}
    assert evaluate_guard(guard, ctx)[0]
    ctx["params"]["card"] = "R9"
    assert not evaluate_guard(guard, ctx)[0]


def test_guard_not():
    s = {"board": [" ", "X"]}
    assert evaluate_guard({"not": {"op": "eq", "path": "board.0", "value": "X"}}, _ctx(s))[0]


def test_evaluate_guards_and():
    s = {"a": 1, "b": 2}
    ctx = _ctx(s)
    ok = [{"op": "gt", "path": "a", "value": 0}, {"op": "lt", "path": "b", "value": 3}]
    assert evaluate_guards(ok, ctx)[0]
    bad = [{"op": "gt", "path": "a", "value": 0}, {"op": "lt", "path": "b", "value": 2}]
    assert not evaluate_guards(bad, ctx)[0]
    assert evaluate_guards(None, ctx)[0]  # no guards = pass
    assert evaluate_guards({"op": "exists", "path": "a"}, ctx)[0]  # single dict


def test_guard_unknown_op_raises():
    try:
        evaluate_guard({"op": "bogus", "path": "x"}, _ctx({}))
    except GuardError:
        return
    raise AssertionError("expected GuardError for unknown op")
