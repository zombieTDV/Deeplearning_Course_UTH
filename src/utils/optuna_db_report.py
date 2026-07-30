"""
optuna_db_report.py — Load Optuna DB, analyze, export results. 
Usage: python optuna_db_report.py <path_to_db> <study_name> [output_dir]
"""
import sys, json, os, pandas as pd
import optuna
from optuna.importance import get_param_importances
from optuna.trial import TrialState

DB_PATH = sys.argv[1]
STUDY_NAME = sys.argv[2]
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else 'agents'
os.makedirs(OUT_DIR, exist_ok=True)

study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///{DB_PATH}')
complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
best = study.best_trial

# best trial JSON
json.dump(
    {
        'number': best.number, 'value': best.value,
        'duration_seconds': best.duration.total_seconds(),
        'params': best.params, 'user_attrs': best.user_attrs,
    },
    open(os.path.join(OUT_DIR, 'best_trial.json'), 'w'), indent=2,
)

# all trials CSV
df = study.trials_dataframe()
df.to_csv(os.path.join(OUT_DIR, 'all_trials.csv'), index=False)

# text report
lines = [
    f'Study: {STUDY_NAME}',
    f'Total: {len(study.trials)}  Complete: {len(complete)}  Pruned: {len(pruned)}',
    '', f'Best trial: #{best.number}  value={best.value:.6f}', '',
]
for t in sorted(complete, key=lambda t: -t.value)[:5]:
    dur_m = t.duration.total_seconds() / 60
    val_acc = t.user_attrs.get('val_acc', '?')
    lines.append(f'  #{t.number:2d}  value={t.value:.6f}  val_acc={val_acc}  dur={dur_m:.1f}min')
    for k, v in t.params.items():
        lines.append(f'      {k}={v}')
    lines.append('')

lines.append('Hyperparameter importance (FANOVA):')
for name, val in sorted(get_param_importances(study).items(), key=lambda x: -x[1]):
    lines.append(f'  {name}: {val:.4f}')

lines.append('')
lines.append('Dead zones (100% pruned):')
for hp in [c for c in df.columns if c.startswith('params_')]:
    ct = pd.crosstab(df[hp], df['state'])
    pruned_frac = ct.get('PRUNED', 0) / ct.sum(axis=1)
    dead = pruned_frac[pruned_frac == 1.0]
    if not dead.empty:
        lines.append(f'  {hp}: {list(dead.index)}')

with open(os.path.join(OUT_DIR, 'report.txt'), 'w') as f:
    f.write('\n'.join(lines))

print(f'Report saved to {OUT_DIR}/')
