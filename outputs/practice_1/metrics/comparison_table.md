# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 87.77 | 93.22 | +5.45 |
| Macro ROC-AUC | 0.9906 | 0.9961 | +0.0056 |
| Macro PR-AUC (AP) | 0.9364 | 0.9736 | +0.0372 |
| Parameters | 235,146 | 140,778 | -94,368 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9870 | 0.9935 |
| Trouser | 0.9992 | 0.9998 |
| Pullover | 0.9836 | 0.9945 |
| Dress | 0.9932 | 0.9975 |
| Coat | 0.9848 | 0.9956 |
| Sandal | 0.9990 | 0.9998 |
| Shirt | 0.9634 | 0.9818 |
| Sneaker | 0.9982 | 0.9995 |
| Bag | 0.9988 | 0.9999 |
| Ankle boot | 0.9983 | 0.9993 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9060 | 0.9534 |
| Trouser | 0.9956 | 0.9976 |
| Pullover | 0.8861 | 0.9683 |
| Dress | 0.9511 | 0.9776 |
| Coat | 0.8861 | 0.9639 |
| Sandal | 0.9920 | 0.9980 |
| Shirt | 0.7786 | 0.8859 |
| Sneaker | 0.9835 | 0.9964 |
| Bag | 0.9933 | 0.9993 |
| Ankle boot | 0.9913 | 0.9953 |
