import nbformat

notebook_path = "notebooks/02_ex2_finetune.ipynb"

cells = []

# Cell 0: Markdown Header with 100% Clickable Markdown Links
cell_0 = nbformat.v4.new_markdown_cell(source="""# Practice 3: Exercise 2 — Finetuned Transformer & PEFT LoRA Sentiment Analysis on IMDB

## Overview & Purpose
This notebook provides a **100% self-contained, interactive end-to-end workflow** for Practice 3 Exercise 2:
- **Two-Stage Data-Centric AI Architecture:**
  - **Stage 1 (Independent Audit Model):** Independent `distilbert-base-uncased` trained with **5-Fold Cross-Validation Out-Of-Fold (OOF)** to produce a 100% unbiased, non-memorized soft probability matrix for Cleanlab Confident Learning.
  - **Stage 2 (Final Model):** `distilbert-base-uncased` + **PEFT LoRA ($r=32, \\alpha=64$)** fine-tuned on the pristine denoised dataset with **Head + Tail Truncation ($128+384=512$)** targeting **93.0% – 94.0%+ Test Accuracy**.
- **Interactive Fine-Tuning Execution Suite (`EXP-00` to `EXP-07` & `EXP-LORA-DENOISED`):** Select presets and train directly inside the notebook.
- **Head + Tail Truncation ($128+384=512$):** Solves verdict truncation on long reviews ($>512$ tokens) while preserving 100% architectural compatibility.
- **Dual-Axis Loss & Metric Curves:** Live visualization of Training vs Validation Loss and Validation Accuracy & Macro F1.
- **Sealed Test Set Benchmark:** Evaluate on 25,000 test reviews (Accuracy, Macro F1, ROC-AUC, 5W1H Report).
- **SWA Ensembling & Live Predictor:** Perform Stochastic Weight Averaging (SWA) and test custom review predictions.
- **Monochromatic Comparison & Error Audit:** Clean head-to-head comparison table against Zero-Shot baseline and top misclassification root cause audit.

---

## 📊 Pipeline Visual Architecture Diagram

```mermaid
flowchart TD
    A["Step 0: Setup & Imports<br/>(PyTorch, CUDA GPU, Parent-Aware PROJECT_ROOT)"] --> B["Step 1: Visual Dataset EDA<br/>(Split Counts, Label Balance, Untruncated Token Length)"]
    B --> B1["Step 1.1: HTML Noise Verification<br/>(Stripping &lt;br /&gt; Tags &amp; Token Economy)"]
    B1 --> B2["Step 1.2: Head + Tail Truncation & Dataset Prep<br/>(128 Head + 384 Tail = 512 Tokens, prepare_imdb)"]
    B2 --> B3["Step 1.3: 5-Fold OOF Audit Model via Cleanlab<br/>(Independent 5-Fold OOF Cross-Validation, Zero Data Leakage)"]
    B3 --> C["Step 2: Final LoRA Training Execution<br/>(Select EXP-LORA-DENOISED, PEFT LoRA r=32, lr=3.0e-4)"]
    C --> D["Step 3: Loss & Metric Curves<br/>(Dual Y-Axes: Train/Val Loss &amp; Val Accuracy/Macro F1)"]
    D --> E["Step 4: Sealed Test Evaluation<br/>(25,000 Test Reviews, 5W1H Summary, Confusion Matrix, ROC)"]
    E --> F["Step 5: SWA & Interactive Predictor<br/>(Averaged Checkpoint, Custom Sentiment Inference)"]
    F --> G["Step 6: Head-to-Head Baseline Comparison<br/>(Monochromatic Table vs Zero-Shot &amp; Misclassification Audit)"]
```

---

## 📑 Roadmap & Presets Table (Clickable Links)
| Step | Description | What it does | Clickable Source / Artifact Path |
|:---:|:---|:---|:---|
| **0** | Environment & Imports | Initialize PyTorch, CUDA GPU device, and `%autoreload 2` | [`notebooks/02_ex2_finetune.ipynb`](02_ex2_finetune.ipynb) |
| **1** | Visual Dataset EDA | Plot 50/50 label balance pie chart & untruncated token length distribution | [`src/data/eda_imdb.py`](../src/data/eda_imdb.py) |
| **1.1** | HTML Noise & Context Reclaim | Verify 58.7% `<br />` artifacts, text diff, and token economy gain | [`src/data/prepare_imdb.py`](../src/data/prepare_imdb.py) |
| **1.2** | Head + Tail Truncation | Visual comparison of 128 Head + 384 Tail retention & tokenize 512 base split | [`src/utils/truncation_viz.py`](../src/utils/truncation_viz.py) |
| **1.3** | 5-Fold OOF Cleanlab Audit | Independent 5-Fold Out-Of-Fold audit across 22,500 train samples (Zero Memorization) | [`src/data/cleanlab_denoiser.py`](../src/data/cleanlab_denoiser.py) |
| **2** | Final LoRA Training | Train PEFT LoRA ($r=32, \\alpha=64$) on clean denoised split | [`src/training/trainer.py`](../src/training/trainer.py) |
| **3** | Dual-Panel Metric Curves | Plot Training vs Validation Loss and Validation Accuracy & F1 | [`src/eval/plotter.py`](../src/eval/plotter.py) |
| **4** | Direct Test Set Evaluation | Run evaluation CLI on 25,000 test reviews & generate 5W1H report | [`src/eval/evaluator.py`](../src/eval/evaluator.py) |
| **5** | SWA & Interactive Inference | Perform SWA parameter averaging & test custom review sentiment | [`src/models/predictor.py`](../src/models/predictor.py) |
| **6** | Baseline Comparison & Audit | Render clean comparison table vs Zero-Shot & analyze misclassifications | [`src/eval/error_auditor.py`](../src/eval/error_auditor.py) |

---

## 🔗 Key Configs & Reports (Clickable Links)
- **LoRA Denoised Config:** [`configs/config_imdb_sentiment_denoised.yaml`](../configs/config_imdb_sentiment_denoised.yaml)
- **EX-13 Head-Tail Report:** [`agents/experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md`](../agents/experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md)
- **EX-08 Cleanlab Report:** [`agents/experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md`](../agents/experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md)
- **EX-07 HTML Cleaning Report:** [`agents/experiments/EX7_HTML_DATA_CLEANING_REPORT.md`](../agents/experiments/EX7_HTML_DATA_CLEANING_REPORT.md)
- **Master Experiments Index:** [`agents/experiments/README.md`](../agents/experiments/README.md)
""")
cells.append(cell_0)

# Cell 1: Environment Setup
cell_1 = nbformat.v4.new_code_cell(source="""%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
print(f"Project Root:   {PROJECT_ROOT}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
""")
cells.append(cell_1)

# Cell 2: Step 1 - EDA
cell_2 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 1: Visual Dataset Exploratory Data Analysis (EDA)
# Source: src/data/eda_imdb.py
from src import IMDBDatasetEDA

eda = IMDBDatasetEDA(max_length=512)
eda.plot_dataset_overview(sample_size=1000)
eda.print_sample_reviews()
""")
cells.append(cell_2)

# Cell 3: Step 1.1 - HTML Cleaning
cell_3 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 1.1: HTML Noise Verification & Clean Data Preprocessing Demonstration
# Source: src/data/eda_imdb.py & src/data/prepare_imdb.py
from src import IMDBDatasetEDA

eda = IMDBDatasetEDA(max_length=512)
eda.plot_html_noise_overview()
""")
cells.append(cell_3)

# Cell 4: Step 1.2 - Head + Tail Truncation Strategy & Dataset Prep
cell_4 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 1.2: Head + Tail Truncation Strategy & Dataset Preparation (EX-13)
# Source: src/utils/truncation_viz.py & src/data/prepare_imdb.py
# 1. Visualize Head (128 tokens: premise) + Tail (384 tokens: verdict & score) preservation
from src.utils.truncation_viz import visualize_head_tail_truncation
from src.data.prepare_imdb import prepare_imdb

truncation_stats = visualize_head_tail_truncation(sample_index=0)

# 2. Execute Head + Tail Tokenization & Split (Train: 22,500, Val: 2,500, Test: 25,000)
tokenized_ds, tokenizer, meta = prepare_imdb(
    dataset_id="stanfordnlp/imdb",
    model_name="distilbert-base-uncased",
    max_length=512,
    processed_dir="data/processed/imdb_tokenized_512",
    force=False,  # Set True to re-tokenize from scratch
)

print(f"\\n✅ Head + Tail Tokenized Dataset Ready: {meta['splits']}")
""")
cells.append(cell_4)

# Cell 5: Step 1.3 - 5-Fold OOF Cleanlab Audit
cell_5 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 1.3: Data-Centric AI — Independent 5-Fold Out-Of-Fold (OOF) Audit via Cleanlab
# Source: src/data/cleanlab_denoiser.py
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
""")
cells.append(cell_5)

# Cell 6: Step 2 - Final LoRA Training Execution
cell_6 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 2: Interactive Fine-Tuning Execution Suite (Final LoRA Model)
# Source: src/training/trainer.py & configs/config_imdb_sentiment_denoised.yaml
from src import IMDBTrainer

# Presets available:
# - "EXP-LORA-DENOISED": Train LoRA on pristine Cleanlab denoised dataset (Recommended)
# - "EXP-LORA": Train LoRA on standard tokenized dataset
# - "EXP-06": Full Fine-Tuning 512 tokens
trainer = IMDBTrainer(preset_name="EXP-LORA-DENOISED", project_root=PROJECT_ROOT)
trainer.run_training()
""")
cells.append(cell_6)

# Cell 7: Step 3 - Loss & Metric Curves
cell_7 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 3: Dual-Panel Metric Curves (Loss & Accuracy/F1)
# Source: src/eval/plotter.py
from src import IMDBPlotter

plotter = IMDBPlotter(project_root=PROJECT_ROOT)
plotter.plot_training_curves()
""")
cells.append(cell_7)

# Cell 8: Step 4 - Test Benchmark Evaluation
cell_8 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 4: Direct Test-Set Evaluation & Finetuned ROC/Confusion Matrix Artifacts
# Source: src/eval/evaluator.py & src/eval/plotter.py
from src import IMDBEvaluator, IMDBPlotter

evaluator = IMDBEvaluator(project_root=PROJECT_ROOT)
evaluator.evaluate_latest_checkpoint()

plotter = IMDBPlotter(project_root=PROJECT_ROOT)
plotter.display_roc_curve(finetuned=True)
""")
cells.append(cell_8)

# Cell 9: Step 5 - SWA Predictor & Custom Inference
cell_9 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 5: Interactive Custom Review Predictor & SWA Ensembling
# Source: src/models/predictor.py
from src import SentimentPredictor

predictor = SentimentPredictor.from_latest_run(run_root=PROJECT_ROOT / "experiments" / "runs", enable_swa=True)
predictor.predict("This film is an absolute masterpiece of modern cinema!")
predictor.predict("Boring, poorly acted, and a complete waste of two hours.")
""")
cells.append(cell_9)

# Cell 10: Step 6 - Head-to-Head Table & Error Audit
cell_10 = nbformat.v4.new_code_cell(source="""import sys
from pathlib import Path
cwd = Path(".").resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 6: Monochromatic Head-to-Head Comparison Table & Misclassification Audit
# Source: src/eval/evaluator.py & src/eval/error_auditor.py
from src import IMDBEvaluator, ErrorAuditor

evaluator = IMDBEvaluator(project_root=PROJECT_ROOT)
evaluator.render_comparison_table()

print("\\n" + "=" * 65)
print(" MISCLASSIFICATION ROOT CAUSE AUDIT")
print("=" * 65)
ErrorAuditor.audit_top_misclassifications(run_root=PROJECT_ROOT / "experiments" / "runs")
""")
cells.append(cell_10)

# Build and write notebook
nb = nbformat.v4.new_notebook(cells=cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully synchronized {len(cells)} cells with direct clickable links to {notebook_path}!")
