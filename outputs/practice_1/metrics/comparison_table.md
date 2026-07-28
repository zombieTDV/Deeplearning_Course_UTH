# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 88.45 | 93.30 | +4.85 |
| Macro ROC-AUC | 0.9913 | 0.9966 | +0.0053 |
| Macro PR-AUC (AP) | 0.9431 | 0.9761 | +0.0330 |
| Parameters | 275,005 | 140,778 | -134,227 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9893 | 0.9948 |
| Trouser | 0.9992 | 1.0000 |
| Pullover | 0.9865 | 0.9953 |
| Dress | 0.9941 | 0.9974 |
| Coat | 0.9861 | 0.9955 |
| Sandal | 0.9988 | 0.9999 |
| Shirt | 0.9647 | 0.9842 |
| Sneaker | 0.9980 | 0.9996 |
| Bag | 0.9981 | 0.9998 |
| Ankle boot | 0.9982 | 0.9996 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9274 | 0.9603 |
| Trouser | 0.9960 | 0.9997 |
| Pullover | 0.9020 | 0.9656 |
| Dress | 0.9562 | 0.9783 |
| Coat | 0.8885 | 0.9673 |
| Sandal | 0.9915 | 0.9990 |
| Shirt | 0.8065 | 0.8999 |
| Sneaker | 0.9837 | 0.9961 |
| Bag | 0.9917 | 0.9989 |
| Ankle boot | 0.9875 | 0.9962 |
