# dblock/ — Vendored DiffusionBlocks Sources (Sakana AI, ICLR 2026)

- **Motivation/Background**: LAB2 evaluation of the DiffusionBlocks method
  (arXiv:2506.14202) needs the official model implementation; the upstream
  repo (https://github.com/SakanaAI/DiffusionBlocks) targets transformers
  4.52.4 and lightning, which we do not install project-wide.
- **Purpose**: Provide a self-contained, transformers-5.x-compatible copy of
  the official ViT + DiffusionBlocks sources, used by
  [src/experiments/exp_08_diffusionblocks.py](../exp_08_diffusionblocks.py).
- **Overview Pipeline**: Copied from upstream `main` (single-commit snapshot),
  patched for transformers >= 5.x, then exercised by the experiment harness.
- **Detailed Plan**: File inventory; patch log; license & attribution.
- **References**: transformers, torch, scipy, numpy.

- **Created**: 2026-09-06T13:14:10+07:00
- **Last Updated**: 2026-09-06T13:14:10+07:00

---

## File inventory

| File | Origin | Changes |
| --- | --- | --- |
| `vit.py` | upstream `vit.py` (Apache-2.0) | patched, see below |
| `dblock_modules.py` | upstream `dblock_modules.py` (Apache-2.0) | unmodified |

## Patch log (`vit.py`, transformers >= 5.x compatibility only)

1. **`ViTIntermediate`** — transformers 5.x renamed it to `ViTMLP` and merged
   the two MLP halves (no dropout). A 4.x-equivalent `ViTIntermediate` class
   (dense + GELU) is vendored so the official `ViTDiTLayer` split
   (intermediate → ViTOutput) behaves identically to the paper's code.
2. **`get_head_mask` / `_convert_head_mask_to_5d`** — removed from
   `PreTrainedModel` in transformers 5.x; vendored on `ViTPreTrainedModel`.
3. **`ViTPatchEmbeddings.forward`** — v5 no longer accepts the
   `interpolate_pos_encoding` kwarg; the vendored `ViTDiTEmbeddings` call site
   drops it (the official code raises `NotImplementedError` for interpolation
   anyway).

No training/inference math was modified. The experiment harness
(`exp_08_diffusionblocks.py`) ports the official `model.py` training step and
diffusion inference verbatim (get_sigmas / get_weights / estimate_target_layer
/ denoise / diffusion_step).

## License & attribution

Upstream: Sakana AI DiffusionBlocks, Apache-2.0
(https://github.com/SakanaAI/DiffusionBlocks). The vendored files retain their
original copyright headers. The base ViT code is © Google AI / Ross Wightman /
The HuggingFace Inc. team, copied from
https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/vit/modeling_vit.py.
