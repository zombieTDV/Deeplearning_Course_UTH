Name:
Evaluation Pipeline Template

Description:
Evaluate trained model performance on validation/test sets, computing evaluation metrics and exporting reports.

Purpose:
Assess model accuracy, loss, and generalization performance.

Input:
Trained model checkpoint and test DataLoader

Output:
Evaluation metric report and summary plots saved to `evaluation/`.

How to do:
1. Set model to `eval()` mode with `torch.no_grad()`
2. Run forward pass across test batches
3. Compute evaluation metrics (Accuracy, F1, Loss, confusion matrix)
4. Export results to `evaluation/` directory

Examples:
Input:  Test dataset
Output: Performance summary report and evaluation plots
