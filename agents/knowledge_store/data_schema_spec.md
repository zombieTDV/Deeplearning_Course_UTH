# Data Schema Specification (Knowledge Store Template)

## Dataset Specifications
* **Dataset Name**: `<DATASET_NAME>` (e.g., ImageNet, CIFAR-10, Custom tabular/image data)
* **Input Data Shape**: `<INPUT_SHAPE>` (e.g., `(C, H, W)` or `(num_features,)`)
* **Data Type & Range**: Float32 (Normalized `[min_val, max_val]`)
* **Number of Classes**: `<NUM_CLASSES>` (or Target Output Dimension)

## Class Label Mapping
```text
0: <CLASS_0_LABEL>
1: <CLASS_1_LABEL>
...
N: <CLASS_N_LABEL>
```

## Data Split Ratios
* **Train Set**: `<TRAIN_RATIO>%`
* **Validation Set**: `<VAL_RATIO>%`
* **Test Set**: `<TEST_RATIO>%`
