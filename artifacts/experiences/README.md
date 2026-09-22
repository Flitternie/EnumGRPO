# Released SWAN experience pools

These frozen pools are the exact inputs used for the reported three-run SWAN
evaluations:

- `swan_enumgrpo.json`: full EnumGRPO pool (32 experiences).
- `swan_vanilla_grpo.json`: pool learned without plan enumeration
  (27 experiences).
- `swan_reflexion.json`: pool learned by independent trajectory reflection,
  without grouped comparison (76 experiences).

Each JSON object records the final learning step and an `experiences` mapping
from stable IDs to the complete experience text. Evaluation launchers consume
these JSON files directly; no separately generated prompt artifact is needed.
