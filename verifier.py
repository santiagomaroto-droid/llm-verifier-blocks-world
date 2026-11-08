"""Verificador simbólico de planes (mundo de bloques).

El LLM propone un plan como texto ("pick(b); place(b); ...") y este módulo
comprueba paso a paso precondiciones y efectos. Si algo falla, devuelve el
motivo, que luego se le pasa al LLM para que replanifique.
"""
import re
from dataclasses import dataclass
from typing import Optional

ARITY = {"pick": 1, "place": 1, "stack": 2}


def initial_state(stacks):
    """stacks: lista de pilas, de abajo arriba. [["a","b"],["c"]] -> b sobre a, c suelto."""
    facts = {("handempty",)}
    for stack in stacks:
        for i, block in enumerate(stack):
            facts.add(("ontable", block) if i == 0 else ("on", block, stack[i - 1]))
            if i == len(stack) - 1:
                facts.add(("clear", block))
    return frozenset(facts)


def describe(state):
    """Estado en texto, pensado para meterlo en el prompt del LLM."""
    fmt = lambda f: f[0] if len(f) == 1 else f"{f[0]}({','.join(f[1:])})"
    return ", ".join(sorted(fmt(f) for f in state))


def apply(state, action, *args):
    """Devuelve (nuevo_estado, None) o (None, motivo_del_fallo)."""
    if action not in ARITY or len(args) != ARITY[action]:
        return None, f"acción desconocida o con argumentos incorrectos: {action}{args}"
    blocks = {b for f in state for b in f[1:]}
    for b in args:
        if b not in blocks:
            return None, f"{action}: el bloque '{b}' no existe"
    s, name = set(state), f"{action}({','.join(args)})"

    if action == "pick":
        (x,) = args
        below = next((f[2] for f in s if f[0] == "on" and f[1] == x), None)
        if ("handempty",) not in s:
            return None, f"{name}: la mano no está libre"
        if ("clear", x) not in s:
            return None, f"{name}: {x} tiene algo encima o ya está en la mano"
        s -= {("handempty",), ("clear", x)}
        if below:
            s.discard(("on", x, below))
            s.add(("clear", below))
        else:
            s.discard(("ontable", x))
        s.add(("holding", x))

    elif action == "place":  # dejar en la mesa
        (x,) = args
        if ("holding", x) not in s:
            return None, f"{name}: no estás sosteniendo {x}"
        s.discard(("holding", x))
        s |= {("ontable", x), ("clear", x), ("handempty",)}

    elif action == "stack":  # apilar x sobre y
        x, y = args
        if ("holding", x) not in s:
            return None, f"{name}: no estás sosteniendo {x}"
        if x == y or ("clear", y) not in s:
            return None, f"{name}: {y} no está libre"
        s -= {("holding", x), ("clear", y)}
        s |= {("on", x, y), ("clear", x), ("handempty",)}

    return frozenset(s), None


def parse_plan(text):
    """'pick(b); place(b)' -> [('pick','b'), ('place','b')]. Acepta ; o saltos de línea."""
    steps = []
    for chunk in re.split(r"[;\n]+", text):
        m = re.fullmatch(r"\s*(\w+)\(([^)]*)\)\s*", chunk)
        if m:
            args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
            steps.append((m.group(1), *args))
        elif chunk.strip():
            steps.append(("?", chunk.strip()))  # basura del LLM: falla al verificar
    return steps


@dataclass
class Result:
    ok: bool
    final: frozenset
    failed_step: Optional[int] = None
    reason: Optional[str] = None


def verify_plan(state, plan, goal=frozenset()):
    for i, (action, *args) in enumerate(plan):
        new, err = apply(state, action, *args)
        if err:
            return Result(False, state, i, err)
        state = new
    if not set(goal) <= state:
        missing = describe(frozenset(set(goal) - state))
        return Result(False, state, None, f"el plan termina sin cumplir el objetivo: falta {missing}")
    return Result(True, state)


if __name__ == "__main__":
    start = initial_state([["a", "b"], ["c"]])  # b sobre a; c suelto
    goal = {("on", "a", "c")}
    print("Estado inicial:", describe(start))

    good = parse_plan("pick(b); place(b); pick(a); stack(a,c)")
    print("Plan bueno ->", verify_plan(start, good, goal))

    bad = parse_plan("pick(a); stack(a,c)")
    print("Plan malo  ->", verify_plan(start, bad, goal))
