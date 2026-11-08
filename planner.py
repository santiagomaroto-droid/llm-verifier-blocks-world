"""Bucle: LLM propone plan -> verificador lo comprueba -> si falla, replanifica.

Uso:
    python planner.py --mock     # LLM simulado, sin instalar nada
    python planner.py            # Claude si hay ANTHROPIC_API_KEY; si no, Ollama local
"""
import json
import os
import re
import sys
import urllib.request

from verifier import initial_state, describe, verify_plan

MODEL = os.getenv("PLANNER_MODEL", "claude-haiku-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Tres ejemplos de longitud y forma distintas, para que el modelo no copie una plantilla.
# Se generan con describe() para que el formato sea idéntico al de las tareas reales.
_EJEMPLOS = [
    ([["x"], ["y"]], "Consigue que el bloque y esté encima del bloque x.",
     "pick(y)\nstack(y,x)"),
    ([["x", "y"], ["z"]], "Consigue que el bloque x esté encima del bloque z.",
     "pick(y)\nplace(y)\npick(x)\nstack(x,z)"),
    ([["x", "y", "z"]], "Consigue que el bloque z esté encima del bloque x.",
     "pick(z)\nplace(z)\npick(y)\nplace(y)\npick(z)\nstack(z,x)"),
]
_EJEMPLOS_TXT = "\n\n".join(
    f"Estado: {describe(initial_state(stacks))}\nObjetivo: {goal}\nPlan:\n{plan}"
    for stacks, goal, plan in _EJEMPLOS)

SYSTEM = f"""Eres un planificador para un brazo robótico en un mundo de bloques.
Acciones disponibles:
- pick(x): coge el bloque x (x debe estar libre y tu mano vacía)
- place(x): deja en la mesa el bloque que sostienes
- stack(x,y): apila sobre y el bloque x que sostienes (y debe estar libre)
Reglas:
- Un bloque con otro encima no está libre: primero hay que quitar lo que tiene encima (pick y place).
- Solo puedes sostener un bloque a la vez.
- Termina el plan en cuanto se cumpla el objetivo. No añadas pasos después.
- El número de pasos depende del estado: razona sobre el estado actual, no copies la forma de los ejemplos.
Ejemplos:
{_EJEMPLOS_TXT}

Responde SOLO con el plan, un paso por línea, sin explicaciones."""


def claude_llm(messages):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(model=MODEL, max_tokens=400, system=SYSTEM, messages=messages)
    return r.content[0].text


def ollama_llm(messages):
    """Modelo local vía Ollama (http://localhost:11434). No necesita pip ni clave."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}] + messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def get_llm():
    """Claude si hay clave de API; si no, un modelo local con Ollama."""
    return claude_llm if os.getenv("ANTHROPIC_API_KEY") else ollama_llm


def mock_llm(messages):
    """LLM simulado: se equivoca a la primera y acierta al recibir feedback."""
    if len(messages) == 1:
        return "pick(a)\nstack(a,c)"
    return "pick(b)\nplace(b)\npick(a)\nstack(a,c)"


def parse_llm_output(text):
    """Extrae los pasos tipo pick(b) del texto, tolerando numeración, markdown o charla."""
    return [(name.lower(), *[a.strip().lower() for a in raw.split(",") if a.strip()])
            for name, raw in re.findall(r"\b(\w+)\(([^()]*)\)", text)]


def _prompt(state, instruction):
    return f"Estado: {describe(state)}\nObjetivo: {instruction}"


def _feedback(res):
    where = f"El paso {res.failed_step + 1} falla" if res.failed_step is not None else "El plan falla"
    return (f"{where}: {res.reason}.\n"
            f"Estado en ese punto: {describe(res.final)}.\n"
            "Da un plan nuevo y completo desde el estado inicial.")


def _attempt(state, goal, llm, messages):
    text = llm(messages)
    return text, verify_plan(state, parse_llm_output(text), goal)


def solve(state, goal, instruction, llm, max_retries=3):
    """max_retries=0 equivale a 'sin verificador' (se puntúa solo el primer intento)."""
    messages = [{"role": "user", "content": _prompt(state, instruction)}]
    first_plan = first_error = None
    for attempt in range(max_retries + 1):
        text, res = _attempt(state, goal, llm, messages)
        if attempt == 0 and not res.ok:
            first_plan, first_error = text, res.reason
        if res.ok:
            return {"ok": True, "attempts": attempt + 1, "plan": text,
                    "first_plan": first_plan, "first_error": first_error}
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": _feedback(res)}]
    return {"ok": False, "attempts": max_retries + 1, "plan": text, "last_error": res.reason,
            "first_plan": first_plan, "first_error": first_error}


def compare(state, goal, instruction, llm, max_retries=3):
    """Un primer intento compartido y, si falla, dos ramas de reintento:
    - fb: con feedback del verificador (le decimos por qué falló)
    - rs: sin feedback (control: se repite la misma pregunta; ¿ayuda el feedback o solo reintentar?)
    """
    first = [{"role": "user", "content": _prompt(state, instruction)}]
    text, res = _attempt(state, goal, llm, first)
    out = {"first_ok": res.ok, "first_plan": text, "first_error": res.reason}
    if res.ok:
        out.update({"fb_ok": True, "fb_attempts": 1, "rs_ok": True, "rs_attempts": 1})
        return out
    for name, use_feedback in (("fb", True), ("rs", False)):
        messages = list(first)
        if use_feedback:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": _feedback(res)}]
        ok, attempts, last = False, 1, res
        for attempt in range(2, max_retries + 2):
            attempts = attempt
            t, r = _attempt(state, goal, llm, messages)
            if r.ok:
                ok = True
                break
            last = r
            if use_feedback:
                messages += [{"role": "assistant", "content": t},
                             {"role": "user", "content": _feedback(r)}]
        out[name + "_ok"], out[name + "_attempts"] = ok, attempts
        out[name + "_error"] = None if ok else last.reason
    return out


if __name__ == "__main__":
    start = initial_state([["a", "b"], ["c"]])
    goal = {("on", "a", "c")}
    instruction = "Pon el bloque a encima del bloque c."

    if "--mock" in sys.argv:
        print("(LLM simulado)")
        llm = mock_llm
    else:
        llm = get_llm()
        print(f"(usando {'Claude' if llm is claude_llm else 'Ollama: ' + OLLAMA_MODEL})")

    print("Sin verificador:", solve(start, goal, instruction, llm, max_retries=0))
    print("Con verificador:", solve(start, goal, instruction, llm, max_retries=3))
