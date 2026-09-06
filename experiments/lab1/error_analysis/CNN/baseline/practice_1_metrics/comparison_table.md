# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 88.15 | 92.02 | +3.87 |
| Macro ROC-AUC | 0.9909 | 0.9953 | +0.0044 |
| Macro PR-AUC (AP) | 0.9394 | 0.9677 | +0.0283 |
| Parameters | 235,146 | 421,642 | +186,496 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9889 | 0.9928 |
| Trouser | 0.9992 | 0.9997 |
| Pullover | 0.9837 | 0.9925 |
| Dress | 0.9939 | 0.9970 |
| Coat | 0.9857 | 0.9934 |
| Sandal | 0.9989 | 0.9997 |
| Shirt | 0.9637 | 0.9789 |
| Sneaker | 0.9978 | 0.9994 |
| Bag | 0.9991 | 0.9997 |
| Ankle boot | 0.9981 | 0.9995 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9154 | 0.9474 |
| Trouser | 0.9953 | 0.9982 |
| Pullover | 0.8906 | 0.9518 |
| Dress | 0.9548 | 0.9737 |
| Coat | 0.8901 | 0.9496 |
| Sandal | 0.9910 | 0.9978 |
| Shirt | 0.7928 | 0.8689 |
| Sneaker | 0.9811 | 0.9948 |
| Bag | 0.9939 | 0.9983 |
| Ankle boot | 0.9893 | 0.9962 |
