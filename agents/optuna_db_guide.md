# Optuna Database (RDB) — Loading, Analysis, and Export Guide

## 1. Loading a Study from SQLite Storage

```python
import optuna

study = optuna.load_study(
    study_name='<study_name>',
    storage='sqlite:///<path_to_db>.db'
)
```

**Key notes:**
- `study_name` must match the name passed to `create_study()`
- The connection string format is `sqlite:///absolute/or/relative/path.db`
- `load_study` does **not** re-run optimization — it reads persisted trials

---

## 2. Basic Study Inspection

```python
# ── Trial counts ──
len(study.trials)                                       # total
sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
sum(1 for t in study.trials if t.state == TrialState.PRUNED)

# ── Best trial ──
best = study.best_trial
best.number             # trial index
best.value              # objective value
best.params             # dict of all HPs
best.user_attrs         # custom attributes (e.g. val_acc)
best.duration           # timedelta
```

### DataFrames

```python
df = study.trials_dataframe()
# Columns pattern:
#   number, value, datetime_start, datetime_complete, duration,
#   params_<hp_name>, user_attrs_<key>, state
```

---

## 3. Filtering Trials

```python
from optuna.trial import TrialState

complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
pruned   = [t for t in study.trials if t.state == TrialState.PRUNED]

# By parameter value
good = [t for t in complete if t.params.get('dropout', 0) >= 0.2]

# By duration (e.g. exclude hardware anomalies)
normal = [t for t in complete if t.duration.total_seconds() < 600]
```

**Common pattern:** Trials that took 10×+ longer than peers (same config family) likely suffered a hardware hiccup — exclude from analysis.

---

## 4. Hyperparameter Importance (FANOVA)

```python
from optuna.importance import get_param_importances

# Default: all trials, objective value as target
imp = get_param_importances(study)

# On validation accuracy instead (if stored in user_attrs)
imp_val = get_param_importances(
    study,
    target=lambda t: t.user_attrs.get('val_acc', 0) / 100.0
)

# Only complete trials, excluding pruned
imp_clean = get_param_importances(study, target=None)
```

**Important:** FANOVA has internal randomness — re-running may shift values slightly. Stable ranking typically requires ≥5 complete trials. The target metric choice (objective value vs. val_acc) can drastically reorder importance (e.g. batch_norm may jump from 0.01 to 0.28 when targeting val_acc instead of macro PR-AUC).

---

## 5. Pattern Analysis

### 5a. Top-N best trials

```python
sorted_trials = sorted(complete, key=lambda t: -t.value)
for t in sorted_trials[:5]:
    dur_min = t.duration.total_seconds() / 60
    print(f'#{t.number:2d}  val={t.value:.4f}  dur={dur_min:.1f}min')
    for k, v in t.params.items():
        print(f'    {k}={v}')
```

### 5b. Parameter distribution by state

```python
for hp in ['params_optimizer', 'params_scheduler', 'params_activation']:
    for state_name, group in [('COMPLETE', df_complete), ('PRUNED', df_pruned)]:
        counts = group[hp].value_counts()
        print(f'{state_name} {hp}:')
        for val, cnt in counts.items():
            print(f'  {val}: {cnt}')
```

### 5c. Dead parameter zones (100% pruned)

Cross-tabulate `state` against each categorical HP to find configurations that never survive pruning:

```python
for hp in col for col in df.columns if col.startswith('params_'):
    ct = pd.crosstab(df[hp], df['state'])
    total = ct.sum(axis=1)
    pruned_frac = ct.get('PRUNED', 0) / total
    dead = pruned_frac[pruned_frac == 1.0]
    if not dead.empty:
        print(f'{hp}: {list(dead.index)}')
```

### 5d. Learning rate distribution

```python
for state_name, group in [('COMPLETE', complete), ('PRUNED', pruned)]:
    lrs = [t.params['lr'] for t in group if 'lr' in t.params]
    if lrs:
        print(f'{state_name} lr: min={min(lrs):.6f}  med={sorted(lrs)[len(lrs)//2]:.6f}  max={max(lrs):.6f}')
```

---

## 6. Intermediate Value Analysis (Learning Curves per Trial)

```python
# Extract per-epoch values from a trial's intermediate_values dict
trial = study.trials[14]
for epoch, val in sorted(trial.intermediate_values.items()):
    print(f'  Epoch {epoch}: {val:.4f}')
```

**Use cases:**
- Compare convergence speed across similar configs
- Detect whether pruning happened too early for certain HPs (e.g. batch_size=256 might need more warmup steps before the pruner kicks in)
- Validate that the best trial didn't just get lucky on the last epoch

---

## 7. Visualization (Exportable)

```python
from optuna.visualization import (
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_intermediate_values,
)

# Save as HTML (interactive)
fig = plot_param_importances(study)
fig.write_html(os.path.join(OUT_DIR, 'param_importance.html'))

fig = plot_parallel_coordinate(study)
fig.write_html(os.path.join(OUT_DIR, 'parallel_coord.html'))

# Save as static image (requires kaleido or orca)
fig.write_image(os.path.join(OUT_DIR, 'param_importance.png'))
```

---

## 8. Export to /agents Directory

```python
import json, os, csv

AGENTS_DIR = '/path/to/agents'

# ── Summary text report ──
def write_summary_text(study, path):
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == TrialState.PRUNED]
    best = study.best_trial

    lines = [
        f'Study: {study.study_name}',
        f'Trials: {len(study.trials)} total, {len(complete)} complete, {len(pruned)} pruned',
        f'',
        f'Best trial: #{best.number}',
        f'  Value: {best.value:.6f}',
        f'  Duration: {best.duration}',
        f'  Params:',
    ]
    for k, v in best.params.items():
        lines.append(f'    {k}: {v}')
    lines.append('')

    # Importance
    from optuna.importance import get_param_importances
    imp = get_param_importances(study)
    lines.append('Hyperparameter importance:')
    for name, val in sorted(imp.items(), key=lambda x: -x[1]):
        lines.append(f'  {name}: {val:.4f}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

write_summary_text(study, os.path.join(AGENTS_DIR, 'optuna_summary.txt'))

# ── JSON report ──
def write_summary_json(study, path):
    best = study.best_trial
    report = {
        'study_name': study.study_name,
        'n_total': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        'best_trial': {
            'number': best.number,
            'value': best.value,
            'params': best.params,
            'user_attrs': best.user_attrs,
            'duration_seconds': best.duration.total_seconds(),
        },
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

write_summary_json(study, os.path.join(AGENTS_DIR, 'optuna_summary.json'))

# ── Full trial CSV ──
df = study.trials_dataframe()
df.to_csv(os.path.join(AGENTS_DIR, 'all_trials.csv'), index=False)

# ── Importance CSV ──
imp = get_param_importances(study)
import_df = pd.DataFrame(
    sorted(imp.items(), key=lambda x: -x[1]),
    columns=['hyperparameter', 'importance']
)
import_df.to_csv(os.path.join(AGENTS_DIR, 'hyperparameter_importance.csv'), index=False)
```

---

## 9. Complete Workflow Script

```python
"""
optuna_db_report.py — Load Optuna DB, analyze, export results to /agents.
Usage: python optuna_db_report.py <path_to_db> <study_name>
"""
import sys, json, os, pandas as pd
import optuna
from optuna.importance import get_param_importances
from optuna.trial import TrialState

DB_PATH   = sys.argv[1]
STUDY_NAME = sys.argv[2]
AGENTS_DIR = sys.argv[3] if len(sys.argv) > 3 else 'agents'

os.makedirs(AGENTS_DIR, exist_ok=True)

study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///{DB_PATH}')
complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
pruned   = [t for t in study.trials if t.state == TrialState.PRUNED]

# ── 1. Best trial ──
best = study.best_trial
best_report = {
    'number': best.number, 'value': best.value,
    'duration_seconds': best.duration.total_seconds(),
    'params': best.params, 'user_attrs': best.user_attrs,
}
json.dump(best_report, open(os.path.join(AGENTS_DIR, 'best_trial.json'), 'w'), indent=2)

# ── 2. All trials CSV ──
study.trials_dataframe().to_csv(os.path.join(AGENTS_DIR, 'all_trials.csv'), index=False)

# ── 3. Top-N text summary ──
top5 = sorted(complete, key=lambda t: -t.value)[:5]
lines = [f'Study: {STUDY_NAME}', f'Total: {len(study.trials)}  Complete: {len(complete)}  Pruned: {len(pruned)}', '']
lines.append(f'Best trial: #{best.number}  value={best.value:.6f}')
lines.append('')
for t in top5:
    dur_m = t.duration.total_seconds() / 60
    lines.append(f'  #{t.number:2d}  value={t.value:.6f}  val_acc={t.user_attrs.get("val_acc", "?"):.1f}%  dur={dur_m:.1f}min')
    for k, v in t.params.items():
        lines.append(f'      {k}={v}')
    lines.append('')

lines.append('Hyperparameter importance (FANOVA):')
for name, val in sorted(get_param_importances(study).items(), key=lambda x: -x[1]):
    lines.append(f'  {name}: {val:.4f}')

lines.append('')
lines.append('Dead zones (100% pruned):')
for hp in ['params_optimizer', 'params_scheduler', 'params_batch_norm', 'params_activation']:
    ct = pd.crosstab(study.trials_dataframe()[hp], study.trials_dataframe()['state'])
    total = ct.sum(axis=1)
    pruned_frac = ct.get('PRUNED', 0) / total
    dead = pruned_frac[pruned_frac == 1.0]
    if not dead.empty:
        lines.append(f'  {hp}: {list(dead.index)}')

with open(os.path.join(AGENTS_DIR, 'report.txt'), 'w') as f:
    f.write('\n'.join(lines))

print(f'Report saved to {AGENTS_DIR}/')
```

---

## 10. Quick Reference: Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| FANOVA importance shifts between runs | Different rankings each call | Run multiple times and average; ensure ≥5 complete trials |
| Dashboard vs. script disagree on importance | Different HP rankings | Check whether the dashboard targets `val_acc` vs. objective value |
| `load_study` fails | `StudyNotFound` | Verify `study_name` and DB path; create study first with `create_study` |
| `params_momentum` is NaN for most trials | FANOVA warnings | Conditional HPs cause NaN — filter or impute; FANOVA handles NaNs by design |
| Trial #17 took 81× longer than peers | Outlier skewing analysis | Filter by duration `t.duration.total_seconds() < threshold` |
| Pruned trials have no `value` | FANOVA raises on missing values | Check `t.value` — if `None`, the trial never reported an intermediate value |
| Plot export fails | `write_image` requires kaleido | Install `kaleido` or use `write_html` for interactive plots |
