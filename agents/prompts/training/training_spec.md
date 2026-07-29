Name:
Training Pipeline Template

Description:
Implement training loop with loss calculation, gradient backpropagation, optimizer update steps, and checkpoint saving.

Purpose:
Optimize model parameters over training epochs and save model weights.

Input:
Model instance, DataLoader, Loss function, Optimizer

Output:
Trained model weights saved to `models/backup/` and training metrics log.

How to do:
1. Set model to `train()` mode
2. Iterate over DataLoader batches
3. Compute loss, run `loss.backward()`, and call `optimizer.step()`
4. Log epoch loss and save checkpoints to `models/backup/<model_name>_epoch.pt`

Examples:
Input:  `<EPOCHS>` training run
Output: Model checkpoint `models/backup/<model_name>_best.pt`
