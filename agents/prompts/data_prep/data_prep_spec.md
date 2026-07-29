Name:
Data Preparation Pipeline Template

Description:
Load raw dataset from `data/raw/`, apply preprocessing transformations, and export batched DataLoaders.

Purpose:
Prepare cleaned and normalized data batches for model training and validation.

Input:
Raw dataset in `data/raw/<DATASET_NAME>`

Output:
Train & Validation `DataLoader` instances producing batched tensors `(B, <INPUT_SHAPE>)` and targets `(B,)`.

How to do:
1. Load dataset from `data/raw/`
2. Apply normalization and augmentation transformations
3. Wrap in PyTorch `DataLoader` with `batch_size` from config
4. Verify batch tensor shapes in Smoke Test

Examples:
Input:  Raw dataset files
Output: PyTorch DataLoader producing input tensors `(B, C, H, W)` and labels `(B,)`
