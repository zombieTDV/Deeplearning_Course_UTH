# 🐛 BUG-01: PyTorch DataLoader Multiprocessing `BrokenPipeError` in Python 3.14 & Linux Jupyter Environments

- **Bug ID**: `BUG-01`
- **Category**: Runtime Execution / Multiprocessing / DataLoader
- **Date Identified**: 2026-08-04
- **Severity**: High (Blocked validation & training loop execution in Jupyter notebooks)
- **Status**: Resolved ✅
- **Target Component**: [`src/data/dataloader.py`](../../src/data/dataloader.py), [`scratch/build_notebook.py`](../../scratch/build_notebook.py), [`notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb)

---

## 1. Symptom & Error Traceback

When executing validation or training loops inside Jupyter Notebook cells (`validate()` or `train_one_epoch()`), the execution crashed during DataLoader iteration with a `BrokenPipeError`:

```text
---------------------------------------------------------------------------
BrokenPipeError                           Traceback (most recent call last)
Cell In[5], line 82
     80         writer = SummaryWriter(log_dir=os.path.join(TB_LOG_DIR, run_name))
---> 82         res = train_model(model, loader, val_loader, criterion, optimizer,
     83                           device, num_epochs=NUM_EPOCHS, run_name=run_name,
     84                           scheduler=scheduler, writer=writer, save_dir=CKPT_DIR,
     85                           early_stopping=True, patience=3, min_delta=1e-4)

File ~/Desktop/Deeplearning_Course_UTH/src/training/train_model.py:181, in train_model(...)
--> 181 val_loss, val_acc = validate(model, val_loader, criterion, device)

File ~/Desktop/Deeplearning_Course_UTH/src/training/train_model.py:85, in validate(...)
---> 85 for images, labels in loader:

File ~/Desktop/Deeplearning_Course_UTH/.venv/lib64/python3.14/site-packages/torch/utils/data/dataloader.py:505, in DataLoader.__iter__(self)
--> 505     return self._get_iterator()

File ~/Desktop/Deeplearning_Course_UTH/.venv/lib64/python3.14/site-packages/torch/utils/data/dataloader.py:438, in DataLoader._get_iterator(self)
--> 438     return _MultiProcessingDataLoaderIter(self)

File ~/Desktop/Deeplearning_Course_UTH/.venv/lib64/python3.14/site-packages/torch/utils/data/dataloader.py:1176, in _MultiProcessingDataLoaderIter.__init__(self, loader)
-> 1176     w.start()

File /usr/lib64/python3.14/multiprocessing/process.py:121, in BaseProcess.start(self)
--> 121 self._popen = self._Popen(self)

File /usr/lib64/python3.14/multiprocessing/context.py:306, in ForkServerProcess._Popen(process_obj)
--> 306     return Popen(process_obj)

File /usr/lib64/python3.14/multiprocessing/popen_forkserver.py:58, in Popen._launch(self, process_obj)
     57 with open(w, 'wb', closefd=True) as f:
---> 58     f.write(buf.getbuffer())

BrokenPipeError: [Errno 32] Broken pipe
```

---

## 2. Root Cause Analysis

1. **Multiprocessing Forkserver Incompatibility**:
   In Python 3.14 on Linux systems, the default multiprocessing context inside a Jupyter Kernel process is set to `forkserver`.
2. **IPC Pipe Disconnection**:
   When PyTorch's `_MultiProcessingDataLoaderIter` attempts to spawn background worker processes (`num_workers > 0`, e.g., `num_workers=2`), writing worker metadata to the forkserver socket pipe (`f.write(buf.getbuffer())`) fails due to premature socket pipe closure (`Errno 32 Broken pipe`).
3. **Jupyter Environment Constraint**:
   Spawning child worker processes inside Jupyter interactive kernel loops on Linux frequently encounters IPC pipe deadlocks or broken pipes when handling complex tensor dataloaders.

---

## 3. Solution & Remediation Strategy

### 3.1 Single-Process Main Thread Loading (`num_workers = 0`)
Setting `num_workers = 0` forces PyTorch DataLoader to fetch data on the main process thread, completely bypassing worker process spawning, IPC pipe communication, and forkserver overhead.

### 3.2 Code Changes Applied

1. **Updated [`src/data/dataloader.py`](../../src/data/dataloader.py)**:
   - Changed default `num_workers` parameter to `0`.
   - Added automatic fallback safety: if `num_workers == 0`, set `persistent_workers = False`.

   ```python
   def get_cifar10_loaders(
       batch_size: int = 64,
       num_workers: int = 0,
       train_transform: object | None = None,
       eval_transform: object | None = None,
       pin_memory: bool = True,
       persistent_workers: bool = False,
   ) -> tuple[DataLoader, DataLoader, DataLoader]:
       if num_workers == 0:
           persistent_workers = False
   ```

2. **Updated [`scratch/build_notebook.py`](../../scratch/build_notebook.py)**:
   - Updated Cell 5 constant: `NUM_WORKERS = 0`.

3. **Regenerated Notebook Deliverable**:
   - Re-executed `python scratch/build_notebook.py` to update [`notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb).

---

## 4. Verification Evidence

Executed verification test script simulating DataLoader iteration with `num_workers=0`:

```bash
.venv/bin/python -c "
from src.data.dataloader import get_cifar10_loaders
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=0)
images, labels = next(iter(val_loader))
print('SUCCESS! Val batch loaded smoothly with shape:', images.shape)
"
```

**Output**:
```text
SUCCESS! Val batch loaded smoothly with shape: torch.Size([64, 3, 224, 224])
```

---

## 5. Related Links & References
- **Master Bug Index**: [`agents/bugs/README.md`](README.md)
- **DataLoader Implementation**: [`src/data/dataloader.py`](../../src/data/dataloader.py)
- **Notebook Generator Script**: [`scratch/build_notebook.py`](../../scratch/build_notebook.py)
- **Notebook Deliverable**: [`notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb)
