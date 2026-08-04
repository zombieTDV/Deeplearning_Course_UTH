# Class-Logit Bias Sweep — Progress & Tracking Status

## Status: COMPLETED ✅

### Checklist
- [x] Formulate logit bias tuning strategy for confused classes (`cat` vs `dog`)
- [x] Create standalone Jupyter notebook: [`notebooks/practice_2_logit_bias_sweep.ipynb`](file:///home/bush/Desktop/Deeplearning_Course_UTH/notebooks/practice_2_logit_bias_sweep.ipynb)
- [x] Implement logit extraction helper for Validation & Test sets
- [x] Perform 2D Grid Search over $(\beta_{\text{cat}}, \beta_{\text{dog}})$ on `val_loader` (5,000 samples)
- [x] Select optimal bias $(\beta_{\text{cat}}^*, \beta_{\text{dog}}^*)$ maximizing Validation Accuracy
- [x] Benchmark baseline vs. bias-tuned model on `test_loader` (10,000 samples)
- [x] Generate Validation Heatmap & Before/After Confusion Matrix artifacts
- [x] Save metrics to `experiments/results/logit_bias_sweep_results.json`

### Key Results Summary

| Model | Baseline Val Acc | Tuned Val Acc | Optimal $(\beta_{\text{cat}}^*, \beta_{\text{dog}}^*)$ | Baseline Test Acc | Tuned Test Acc | Test Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 95.60% | **95.86%** | $(-0.50, +0.10)$ | 95.64% | **95.69%** | **+0.05%** |
| **DenseNet121** | 96.16% | **96.24%** | $(-0.10, +0.20)$ | 96.09% | 96.06% | -0.03% |
| **Soft-Voting Ensemble** | 97.14% | **97.28%** | $(-0.10, +0.10)$ | 96.76% | 96.73% | -0.03% |

