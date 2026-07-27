# Agents Log

## Session 1 — 2026-07-25

**Model:** DeepSeek V4 Flash  
**Tools:** CLI (Kilo)  
**Prompt origin:** Thực hiện Practice 1: PyTorch FashionMNIST Classification theo agents/purposre.md  
**Prompt enhance:** Tạo notebook practice_1.ipynb với cấu trúc step-by-step, để trống code quan trọng cho học viên tự implement (model architecture, training loop, evaluation).

**Files modified:**
- `requirements.txt` — added torch, torchvision, jupyter, nbformat
- `notebooks/practice_1.ipynb` — created from scratch

**Output:**
- `notebooks/practice_1.ipynb` — 10 steps, markdown instructions + skeleton code
- `outputs/images/sample_fashionmnist.png` — sample visualization
- `outputs/plots/` — empty (chờ học viên train model)
- `outputs/model/` — empty (chờ học viên save model)

**Accept:** Toàn bộ cấu trúc notebook, requirements, agents rules.  
**Deny:** Không điền sẵn code quan trọng (model, training, evaluation) — để học viên tự code.  
**Reason:** Học viên muốn tự code phần quan trọng để học PyTorch.

**Test:** `jupyter nbconvert --execute` thành công — dataset tải được, DataLoader hoạt động, visualization in ra được.
