Name:
Model Architecture Pipeline Template

Description:
Define deep neural network architecture class (`nn.Module`) for target task.

Purpose:
Construct forward propagation graph mapping input feature tensors to output predictions/logits.

Input:
Normalized input tensor `(B, <INPUT_SHAPE>)`

Output:
Output predictions / raw logits tensor `(B, <NUM_CLASSES>)`

How to do:
1. Define class `<MODEL_CLASS_NAME>(nn.Module)` in `models/<model_name>.py`
2. Implement feature extraction / hidden layers in `__init__`
3. Implement `forward(self, x)` method with tensor shape annotations
4. Return raw logits / outputs

Examples:
Input:  Tensor shape `(B, <INPUT_SHAPE>)`
Output: Logits tensor shape `(B, <NUM_CLASSES>)`
