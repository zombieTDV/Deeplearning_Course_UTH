# Model Comparison Table

| Metric | MLP | CNN | Diff |
|--------|----:|----:|-----:|
| Test Accuracy (%) | 89.62 | 93.04 | +3.42 |
| Macro ROC-AUC | 0.9930 | 0.9965 | +0.0035 |
| Macro PR-AUC (AP) | 0.9525 | 0.9752 | +0.0227 |
| Parameters | 275,005 | 140,778 | -134,227 |

## Per-class ROC-AUC

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9909 | 0.9949 |
| Trouser | 0.9997 | 1.0000 |
| Pullover | 0.9884 | 0.9951 |
| Dress | 0.9956 | 0.9975 |
| Coat | 0.9890 | 0.9953 |
| Sandal | 0.9992 | 0.9999 |
| Shirt | 0.9714 | 0.9835 |
| Sneaker | 0.9985 | 0.9995 |
| Bag | 0.9989 | 0.9999 |
| Ankle boot | 0.9986 | 0.9996 |

## Per-class PR-AUC (AP)

| Class | MLP | CNN |
|-------|----:|----:|
| T-shirt/top | 0.9350 | 0.9612 |
| Trouser | 0.9981 | 0.9998 |
| Pullover | 0.9186 | 0.9644 |
| Dress | 0.9646 | 0.9781 |
| Coat | 0.9112 | 0.9647 |
| Sandal | 0.9939 | 0.9991 |
| Shirt | 0.8312 | 0.8941 |
| Sneaker | 0.9870 | 0.9955 |
| Bag | 0.9936 | 0.9995 |
| Ankle boot | 0.9919 | 0.9962 |
