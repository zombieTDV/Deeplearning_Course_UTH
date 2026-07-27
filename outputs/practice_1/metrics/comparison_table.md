# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 89.48 | 93.35 | +3.87 |
| Macro ROC-AUC | 0.9914 | 0.9961 | +0.0047 |
| Macro PR-AUC (AP) | 0.9454 | 0.9731 | +0.0277 |
| Parameters | 275,005 | 140,778 | -134,227 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9883 | 0.9924 |
| Trouser | 0.9991 | 0.9996 |
| Pullover | 0.9839 | 0.9945 |
| Dress | 0.9940 | 0.9976 |
| Coat | 0.9876 | 0.9957 |
| Sandal | 0.9991 | 0.9999 |
| Shirt | 0.9666 | 0.9824 |
| Sneaker | 0.9986 | 0.9996 |
| Bag | 0.9987 | 0.9999 |
| Ankle boot | 0.9983 | 0.9992 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9217 | 0.9531 |
| Trouser | 0.9961 | 0.9980 |
| Pullover | 0.9013 | 0.9625 |
| Dress | 0.9557 | 0.9769 |
| Coat | 0.9087 | 0.9668 |
| Sandal | 0.9925 | 0.9991 |
| Shirt | 0.8049 | 0.8835 |
| Sneaker | 0.9886 | 0.9967 |
| Bag | 0.9937 | 0.9990 |
| Ankle boot | 0.9912 | 0.9955 |
