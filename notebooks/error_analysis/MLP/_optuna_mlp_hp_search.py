"""Optuna hyperparameter search for MLP on FashionMNIST.

Usage:
    python notebooks/error_analysis/MLP/_optuna_mlp_hp_search.py          # full search (50 trials)
    python notebooks/error_analysis/MLP/_optuna_mlp_hp_search.py --quick   # 10 trials for testing
"""

import os, sys, json, argparse, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')))
from src.train_utils import train_one_epoch
from src.eval_utils import get_all_probas_and_labels, compute_pr_auc_scores

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available()
          else 'cpu')
print(f'Device: {DEVICE}')

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
OUT_DIR = os.path.join(PROJ_ROOT, 'outputs', 'error_analysis', 'MLP', 'optuna_search')
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 50
N_VALID = 5000          # samples held out from training set for validation
EPOCHS_PER_TRIAL = 15   # training epochs per trial
BATCH_SIZE = 256        # fixed for faster throughput
FIXED_TEST_BATCH = 512

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ── Data (loaded once at module level) ──────────────────────────────────────
# NOTE: all 50+ trials share the same fixed 5 000-sample validation split.
# TPE may overfit to this particular split's quirks.  The test-set evaluation
# in evaluate_best() is the real check — watch for val/test divergence.
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
_train_full = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=False, transform=_tf)
_test_full = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=False, transform=_tf)

_indices = np.arange(len(_train_full))
np.random.seed(42)
np.random.shuffle(_indices)
_train_subset = Subset(_train_full, _indices[N_VALID:])
_val_subset = Subset(_train_full, _indices[:N_VALID])

TRAIN_LOADER = DataLoader(_train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
VAL_LOADER = DataLoader(_val_subset, batch_size=FIXED_TEST_BATCH, shuffle=False,
                        num_workers=0, pin_memory=True)
TEST_LOADER = DataLoader(_test_full, batch_size=FIXED_TEST_BATCH, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f'Train: {len(TRAIN_LOADER)} batches  Val: {len(VAL_LOADER)} batches  Test: {len(TEST_LOADER)} batches')


# ── Dynamic MLP ─────────────────────────────────────────────────────────────
ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': lambda: nn.LeakyReLU(0.1),
    'ELU': lambda: nn.ELU(alpha=1.0),
    'GELU': nn.GELU,
}


def build_mlp(trial):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    units = []
    for i in range(n_layers):
        low, high = (128, 1024) if n_layers <= 3 else (64, 512)
        units.append(trial.suggest_int(f'units_{i}', low, high, log=True))

    act_name = trial.suggest_categorical('activation', list(ACTIVATIONS.keys()))
    act_cls = ACTIVATIONS[act_name]

    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)

    layers = [nn.Flatten()]
    in_dim = 784
    for i, out_dim in enumerate(units):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 10))

    return nn.Sequential(*layers)


# ── Objective ───────────────────────────────────────────────────────────────
def objective(trial):
    torch.manual_seed(42)
    if DEVICE.startswith('cuda'):
        torch.cuda.manual_seed_all(42)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optim_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    scheduler_on = trial.suggest_categorical('scheduler', ['none', 'cosine', 'step'])

    model = build_mlp(trial).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())

    if optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                              momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    if scheduler_on == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL)
    elif scheduler_on == 'step':
        step_size = trial.suggest_int('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 0.1, 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    train_loader, val_loader, _ = TRAIN_LOADER, VAL_LOADER, TEST_LOADER

    for epoch in range(EPOCHS_PER_TRIAL):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        if scheduler:
            scheduler.step()

        # Validation PR-AUC at end of each epoch for pruning
        probas, labels = get_all_probas_and_labels(model, val_loader, DEVICE, 10)
        pr_scores = compute_pr_auc_scores(probas, labels, model_name='_trial')
        val_pr = pr_scores['macro']

        trial.report(val_pr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return the same metric used for pruning: macro PR-AUC after the final epoch.
    # Validation accuracy is logged in study.trials_dataframe() via the user_attrs below.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = 100 * correct / total
    trial.set_user_attr('val_acc', val_acc)

    return val_pr


# ── Study ───────────────────────────────────────────────────────────────────
def run_study(n_trials=N_TRIALS):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=3, interval_steps=1
        ),
        study_name='mlp_fashionmnist',
        storage=None,
    )
    study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)
    return study


# ── Best trial evaluation ──────────────────────────────────────────────────
def evaluate_best(study):
    best = study.best_trial
    print(f'\n{"="*70}')
    print(f'Best trial: #{best.number}')
    print(f'Val macro PR-AUC: {best.value:.6f}')
    print(f'Params:')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Rebuild model with best params
    class FixedTrial:
        def __init__(self, params):
            self._params = params
        def suggest_int(self, name, low, high, log=False):
            return self._params[name]
        def suggest_float(self, name, low, high, log=False, step=None):
            return self._params[name]
        def suggest_categorical(self, name, choices):
            return self._params[name]

    trial_stub = FixedTrial(best.params)
    model = build_mlp(trial_stub).to(DEVICE)

    optim_name = best.params['optimizer']
    lr = best.params['lr']
    wd = best.params['weight_decay']
    if optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                              momentum=best.params['momentum'])

    scheduler = None
    if best.params['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    elif best.params['scheduler'] == 'step':
        # Scale step_size by 2: search ran 15 epochs, final train is 30.
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=best.params['step_size'] * 2,
            gamma=best.params['gamma'])

    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = TRAIN_LOADER, VAL_LOADER, TEST_LOADER

    # Train full 30 epochs with best-checkpoint early stopping
    print(f'\nTraining best config for 30 epochs...')
    train_losses = []
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(30):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(loss)
        if scheduler:
            scheduler.step()

        # Validation accuracy per epoch for early stopping
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc_epoch = 100 * correct / total

        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_checkpoint.pth'))

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == 29:
            print(f'  Epoch [{epoch+1}/30] Loss: {loss:.4f}  ValAcc: {val_acc_epoch:.2f}%')

    print(f'  Best checkpoint at epoch {best_epoch+1}: val_acc={best_val_acc:.2f}%')
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'best_checkpoint.pth')))
    model.to(DEVICE)

    # Test evaluation from best checkpoint
    probas, labels = get_all_probas_and_labels(model, test_loader, DEVICE, 10)
    pr_scores = compute_pr_auc_scores(probas, labels, model_name='optuna_best')
    from src.eval_utils import evaluate_detailed
    test_acc, _, _ = evaluate_detailed(model, test_loader, DEVICE, CLASS_NAMES, model_name='optuna_best')

    # Save best params and final-model weights
    with open(os.path.join(OUT_DIR, 'best_params.json'), 'w') as f:
        json.dump({
            'number': best.number,
            'val_macro_pr_auc': best.value,
            'val_accuracy': best.user_attrs.get('val_acc', None),
            'test_accuracy': test_acc,
            'best_val_epoch': best_epoch,
            'best_val_acc_at_ckpt': best_val_acc,
            'params': best.params,
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, 'train_losses_best.txt'), 'w') as f:
        for loss in train_losses:
            f.write(f'{loss}\n')

    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model_weights.pth'))

    print(f'\nAll results saved to {OUT_DIR}/')


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run 10 trials for testing')
    parser.add_argument('--trials', type=int, default=None, help='Override trial count')
    args = parser.parse_args()

    n_trials = args.trials or (10 if args.quick else N_TRIALS)
    print(f'Starting Optuna search: {n_trials} trials, {EPOCHS_PER_TRIAL} epochs each')

    study = run_study(n_trials)

    # ── Report ──
    pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
    complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    print(f'\nStudy completed: {complete} complete, {pruned} pruned')

    df = study.trials_dataframe()
    df.to_csv(os.path.join(OUT_DIR, 'all_trials.csv'), index=False)

    evaluate_best(study)
