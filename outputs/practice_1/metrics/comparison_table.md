# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 89.23 | 92.94 | +3.71 |
| Macro ROC-AUC | 0.9918 | 0.9962 | +0.0044 |
| Macro PR-AUC (AP) | 0.9446 | 0.9734 | +0.0288 |
| Parameters | 275,005 | 140,778 | -134,227 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9895 | 0.9936 |
| Trouser | 0.9992 | 0.9999 |
| Pullover | 0.9847 | 0.9946 |
| Dress | 0.9944 | 0.9977 |
| Coat | 0.9870 | 0.9954 |
| Sandal | 0.9993 | 0.9999 |
| Shirt | 0.9677 | 0.9819 |
| Sneaker | 0.9985 | 0.9995 |
| Bag | 0.9993 | 0.9999 |
| Ankle boot | 0.9984 | 0.9995 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9180 | 0.9534 |
| Trouser | 0.9958 | 0.9993 |
| Pullover | 0.8930 | 0.9633 |
| Dress | 0.9590 | 0.9796 |
| Coat | 0.8995 | 0.9646 |
| Sandal | 0.9941 | 0.9994 |
| Shirt | 0.8134 | 0.8833 |
| Sneaker | 0.9868 | 0.9958 |
| Bag | 0.9953 | 0.9994 |
| Ankle boot | 0.9910 | 0.9961 |
