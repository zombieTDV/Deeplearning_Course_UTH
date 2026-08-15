import nbformat

notebook_path = "notebooks/02_ex2_finetune.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# 1. Update Markdown Header (Cell 0)
new_header = """# Practice 3: Exercise 2 — Finetuned Transformer & PEFT LoRA Sentiment Analysis on IMDB

## Overview & Purpose
This notebook provides a **100% self-contained, interactive end-to-end workflow** for Practice 3 Exercise 2:
- **Two-Stage Data-Centric AI Architecture:**
  - **Stage 1 (Independent Audit Model):** Independent `distilbert-base-uncased` trained with **5-Fold Cross-Validation Out-Of-Fold (OOF)** to produce 100% unbiased, non-memorized soft probability matrix for Cleanlab Confident Learning.
  - **Stage 2 (Final Model):** `distilbert-base-uncased` + **PEFT LoRA ($r=32, \\alpha=64$)** fine-tuned on the denoised dataset with **Head + Tail Truncation ($128+384=512$)** targeting **93.0% – 94.0%+ Test Accuracy**.
- **Interactive Fine-Tuning Execution Suite (`EXP-00` to `EXP-07` & `EXP-LORA-DENOISED`):** Select presets and train directly inside the notebook.
- **Head + Tail Truncation ($128+384=512$):** Solves verdict truncation on long reviews ($>512$ tokens) while preserving 100% architectural compatibility.
- **Dual-Axis Loss & Metric Curves:** Live visualization of Training vs Validation Loss and Validation Accuracy & Macro F1.
- **Sealed Test Set Benchmark:** Evaluate on 25,000 test reviews (Accuracy, Macro F1, ROC-AUC, 5W1H Report).
- **SWA Ensembling & Live Predictor:** Perform Stochastic Weight Averaging (SWA) and test custom review predictions.
- **Monochromatic Comparison & Error Audit:** Clean head-to-head comparison table against Zero-Shot baseline and top misclassification root cause audit.

---

## 📑 Roadmap & Presets Table
| Step | Description | What it does | Preset / Artifact Path |
|:---:|:---|:---|:---|
| **0** | Environment & Imports | Initialize PyTorch, CUDA GPU device, and `%autoreload 2` | `notebooks/02_ex2_finetune.ipynb` |
| **1** | Visual Dataset EDA | Plot 50/50 label balance pie chart & untruncated token length distribution | `src/data/eda_imdb.py` |
| **1.1** | HTML Noise & Context Reclaim | Verify 58.7% `<br />` artifacts, text diff, and token economy gain | `src/data/prepare_imdb.py` |
| **1.2** | 5-Fold OOF Cleanlab Audit | Independent 5-Fold Out-Of-Fold audit across 22,500 train samples (Zero Memorization) | `src/data/cleanlab_denoiser.py` |
| **1.3** | Head + Tail Truncation | Visual comparison of 128 Head + 384 Tail retention on long reviews (>512 tokens) | `src/utils/truncation_viz.py` |
| **2** | Final LoRA Training | Train PEFT LoRA ($r=32, \\alpha=64$) on clean denoised split | `src/training/trainer.py` |
| **3** | Dual-Panel Metric Curves | Plot Training vs Validation Loss and Validation Accuracy & F1 | `src/eval/plotter.py` |
| **4** | Direct Test Set Evaluation | Run evaluation CLI on 25,000 test reviews & generate 5W1H report | `src/eval/evaluator.py` |
| **5** | SWA & Interactive Inference | Perform SWA parameter averaging & test custom review sentiment | `src/models/predictor.py` |
| **6** | Baseline Comparison & Audit | Render clean comparison table vs Zero-Shot & analyze misclassifications | `src/eval/error_auditor.py` |
"""

nb.cells[0].source = new_header

# 2. Update Cell 4 (Step 1.2)
cell_4_code = """import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 1.2: Data-Centric AI — Independent 5-Fold Out-Of-Fold (OOF) Audit via Cleanlab
# Stage 1: Train independent Audit Model via 5-Fold Cross-Validation on original 22,500 samples
#          Guarantees 100% Zero-Memorization and uncorrupted out-of-sample probability estimation.
# Stage 2: Cleanlab audits soft probabilities and exports clean dataset (data/processed/imdb_denoised_512).
from src import IMDBCleanlabAuditor

auditor = IMDBCleanlabAuditor(project_root=PROJECT_ROOT)

# Run Confident Learning Audit with 5-Fold Out-Of-Fold Cross-Validation
audit_results = auditor.run_audit_and_visualize(
    top_k=5,
    max_samples=None,
    use_cache=True,
    use_oof=True,     # Golden Standard: 5-Fold Out-Of-Fold Cross-Validation
    n_splits=5,
    epochs=2,
)

# Export pristine denoised dataset (train: 22,388 samples) for Final LoRA model training
auditor.export_denoised_dataset(audit_results["issue_indices"])
"""

nb.cells[4].source = cell_4_code

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully updated 02_ex2_finetune.ipynb!")
