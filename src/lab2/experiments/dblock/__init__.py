"""Vendored DiffusionBlocks sources (Sakana AI, ICLR 2026, Apache-2.0).

- `vit.py`: official ViT/DiT implementation from
  https://github.com/SakanaAI/DiffusionBlocks (transformers 4.52.4 base),
  minimally patched for transformers >= 5.x compatibility:
  * vendored 4.x-style ``ViTIntermediate`` (v5 renamed it to ``ViTMLP``)
  * vendored ``get_head_mask`` / ``_convert_head_mask_to_5d`` (removed in v5)
  * dropped ``interpolate_pos_encoding`` kwarg from ``ViTPatchEmbeddings``
  The official numerics are unchanged.
- `dblock_modules.py`: sigma schedule helpers, unmodified from upstream.

See `dblock/README.md` for attribution and the patch log.
"""
