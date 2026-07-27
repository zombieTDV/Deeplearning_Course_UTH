# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 88.79 | 93.16 | +4.37 |
| Macro ROC-AUC | 0.9911 | 0.9962 | +0.0051 |
| Macro PR-AUC (AP) | 0.9409 | 0.9734 | +0.0325 |
| Parameters | 275,005 | 140,778 | -134,227 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9885 | 0.9939 |
| Trouser | 0.9995 | 0.9998 |
| Pullover | 0.9834 | 0.9944 |
| Dress | 0.9940 | 0.9973 |
| Coat | 0.9865 | 0.9950 |
| Sandal | 0.9989 | 0.9999 |
| Shirt | 0.9636 | 0.9830 |
| Sneaker | 0.9986 | 0.9996 |
| Bag | 0.9996 | 0.9999 |
| Ankle boot | 0.9987 | 0.9993 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9166 | 0.9570 |
| Trouser | 0.9973 | 0.9988 |
| Pullover | 0.8847 | 0.9637 |
| Dress | 0.9552 | 0.9773 |
| Coat | 0.9012 | 0.9606 |
| Sandal | 0.9925 | 0.9990 |
| Shirt | 0.7866 | 0.8859 |
| Sneaker | 0.9873 | 0.9966 |
| Bag | 0.9965 | 0.9990 |
| Ankle boot | 0.9909 | 0.9959 |
