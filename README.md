# Verifier-in-the-loop planning with a small local LLM (blocks world)

A small LLM running locally (qwen2.5:7b via Ollama) proposes a plan for a blocks-world task. A symbolic verifier checks every step against its preconditions and effects. If the plan fails, the verifier's error message (failed step, reason, state at that point) is sent back to the model for another try.

Everything runs on a laptop, no GPU and no paid API. This is a small, honest study, not a new method.

## Results

30 tasks (10 each with 3, 4 and 5 blocks), one run, success rate:

| Blocks | 1 attempt | 4 attempts, no feedback | 4 attempts, with verifier feedback |
|-------:|----------:|------------------------:|-----------------------------------:|
| 3      | 30%       | 30%                     | 50%                                |
| 4      | 70%       | 70%                     | 70%                                |
| 5      | 30%       | 30%                     | 50%                                |
| **All**| **43%**   | **43%**                 | **57%**                            |

- Retrying without feedback did not help. Verifier feedback fixed 4 of 30 tasks and broke none.
- Most failures are state-tracking errors: the model tries to pick up a block that has another block on top of it.
- Prompt design mattered more than expected. With a single worked example in the prompt, the model copied that example's 4-step shape (20% first-attempt success). Three examples of different lengths raised it to 43%.

## Limitations

- One run of 30 tasks. A difference of 4 tasks is suggestive, not statistically conclusive.
- The no-feedback control samples at temperature 0.3, so retries are nearly identical. A fairer control would use a higher temperature.
- Symbolic blocks world only: no perception, no physics, no robot.
- The idea of pairing an LLM with an external verifier is not new (e.g. LLM-Modulo-style loops, SayCan, Text2Motion). This is a small reproducible exercise.
- In `demo.html`, the model's attempts are real failures from the run above, but the "valid plans" are computed by a search, not produced by the model.

## Files

- `verifier.py`: state, actions (`pick`, `place`, `stack`) and step-by-step plan checking.
- `planner.py`: LLM backends (Ollama or Claude API), prompt, and the propose, verify, retry loop.
- `benchmark.py`: task generator (solvable by construction, fixed seed) and the comparison above.
- `demo.html`: open in a browser to replay three real failures and the corrected plans.

## Run it

```
ollama pull qwen2.5:7b
python verifier.py
python benchmark.py --mock   # pipeline check with a simulated LLM (results are NOT real)
python benchmark.py          # real run, 30 to 60 minutes on CPU
```

Python 3 and Ollama are the only requirements. Set `ANTHROPIC_API_KEY` to use Claude instead (`pip install anthropic`).

Author: Santiago

demo.html
