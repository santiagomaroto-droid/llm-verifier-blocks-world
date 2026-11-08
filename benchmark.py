"""Benchmark: ¿ayuda el verificador? 30 tareas de mundo de bloques.

    python benchmark.py --mock    # prueba el pipeline sin API (resultados NO reales)
    python benchmark.py           # LLM real: Claude (si hay ANTHROPIC_API_KEY) u Ollama local

Cada tarea tiene un primer intento compartido. Si falla, se prueban dos ramas (hasta 3 reintentos):
  - con feedback: se le dice al LLM qué paso falló y por qué
  - sin feedback: se repite la misma pregunta (control)
Tres columnas: 1 intento / 4 intentos sin feedback / 4 intentos con feedback.
"""
import argparse
import csv
import itertools
import random
from collections import deque

from verifier import initial_state, describe, apply, ARITY
from planner import compare, get_llm


def applicable(state):
    blocks = sorted({b for f in state for b in f[1:]})
    for action, n in ARITY.items():
        for args in itertools.product(blocks, repeat=n):
            new, err = apply(state, action, *args)
            if not err:
                yield (action, *args), new


def random_stacks(rng, names):
    names = names[:]
    rng.shuffle(names)
    stacks, i = [], 0
    while i < len(names):
        k = rng.randint(1, 3)
        stacks.append(names[i:i + k])
        i += k
    return stacks


def instruction_text(goal):
    parts = [f"el bloque {x} esté encima del bloque {y}" for _, x, y in sorted(goal)]
    return "Consigue que " + " y que ".join(parts) + "."


def make_tasks(seed=0, per_size=10):
    """Tareas resolubles por construcción: el objetivo sale de un estado alcanzable."""
    rng, tasks = random.Random(seed), []
    for n in (3, 4, 5):
        names = list("abcde"[:n])
        count = 0
        while count < per_size:
            start = s = initial_state(random_stacks(rng, names))
            for _ in range(rng.randint(4, 12)):
                s = rng.choice(list(applicable(s)))[1]
            ons = sorted(f for f in s if f[0] == "on")
            if not ons:
                continue
            goal = frozenset(rng.sample(ons, rng.randint(1, min(3, len(ons)))))
            if goal <= start:
                continue
            tasks.append({"n": n, "start": start, "goal": goal,
                          "instruction": instruction_text(goal)})
            count += 1
    return tasks


def oracle_plan(start, goal):
    queue, seen = deque([(start, [])]), {start}
    while queue:
        s, plan = queue.popleft()
        if goal <= s:
            return plan
        for step, new in applicable(s):
            if new not in seen:
                seen.add(new)
                queue.append((new, plan + [step]))


def mock_llm_for(task, rng):
    """LLM simulado: falla ~60% de las veces que se le pregunta lo mismo (se salta un paso)
    y acierta siempre que recibe feedback."""
    good = oracle_plan(task["start"], task["goal"])
    fmt = lambda p: "\n".join(f"{a}({','.join(r)})" for a, *r in p)

    def llm(messages):
        if len(messages) == 1 and len(good) > 1 and rng.random() < 0.6:
            i = rng.randrange(len(good))
            return fmt(good[:i] + good[i + 1:])
        return fmt(good)
    return llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    rng, rows = random.Random(1), []
    for i, t in enumerate(make_tasks()):
        llm = mock_llm_for(t, rng) if args.mock else get_llm()
        r = compare(t["start"], t["goal"], t["instruction"], llm, max_retries=3)
        rows.append({"id": i, "bloques": t["n"],
                     "un_intento": int(r["first_ok"]),
                     "cuatro_sin_feedback": int(r["rs_ok"]),
                     "cuatro_con_feedback": int(r["fb_ok"]),
                     "intentos_con_feedback": r["fb_attempts"],
                     "instruccion": t["instruction"],
                     "estado": describe(t["start"]),
                     "primer_plan": r["first_plan"].replace("\n", " | "),
                     "primer_error": r["first_error"] or "",
                     "error_final_con_feedback": r.get("fb_error") or ""})
        mark = lambda v: "OK" if v else "x "
        print(f"[{i + 1:02d}/30] 1 intento={mark(r['first_ok'])}  4 sin feedback={mark(r['rs_ok'])}"
              f"  4 con feedback={mark(r['fb_ok'])}  {t['instruction']}")

    print("\nBloques | 1 intento | 4 intentos sin feedback | 4 intentos con feedback")
    for n in (3, 4, 5, None):
        sub = [r for r in rows if n is None or r["bloques"] == n]
        pct = lambda key: 100 * sum(r[key] for r in sub) / len(sub)
        print(f"{'TOTAL' if n is None else n:>7} | {pct('un_intento'):8.0f}% | "
              f"{pct('cuatro_sin_feedback'):22.0f}% | {pct('cuatro_con_feedback'):22.0f}%")

    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nGuardado en results.csv")

    fails = [r for r in rows if r["primer_error"]][:3]
    print("\nEjemplos de primer intento fallido:")
    for r in fails:
        print(f"- {r['instruccion']}\n  estado: {r['estado']}\n  plan: {r['primer_plan']}\n  error: {r['primer_error']}")


if __name__ == "__main__":
    main()
