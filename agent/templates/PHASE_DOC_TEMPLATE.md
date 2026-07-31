# <phase_name>.md — Template
Copy this for each phase/stage (data_prep.md, training_info.md, model.md, eval.md...)
One file per phase. Everything below stays in this single file.

## Name
<Short name of this phase, e.g. "Data Preparation">

## Background
Why this phase exists. What problem it addresses. Any relevant context
from the business/technical understanding steps.

## Goals / Purpose
- What "done" looks like for this phase, concretely
- What this phase explicitly does NOT try to solve

## Input / Output
- **Input:** <files, data sources, upstream artifacts, formats>
- **Output:** <files, artifacts, expected shape/schema>

## How to do it (general plan)
1. Step one
2. Step two
3. ...

## Pipeline
<Concrete sequence of scripts/commands, e.g.:>
```
src/data/load_raw.py → src/data/clean.py → src/data/feature_eng.py → data/processed/train.parquet
```

## Detailed experiment plan
- Specific parameters, thresholds, edge cases to handle
- Known gotchas / things that broke before
- Links to relevant code: `src/data/clean.py#L20` (relative link + anchor)
- External reference: <https://...> (absolute URL)

## Links
- Related phase docs: [overview.md](overview.md), [eval.md](eval.md)
- Progress tracking: [progress/<phase>_status.md](progress/<phase>_status.md)
