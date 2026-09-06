# LAB2 & LAB3 Experiment Directory Provenance Audit Report

> **Branch:** `consolidate/unified-main`  
> **Audit Date:** 2026-09-06  
> **Mode:** READ-ONLY — no files modified, staged, committed, or moved.

---

## 1. Tracked File Counts — `experiments/` Directories

| Directory | Tracked Files (excl. `.gitkeep`) | `.gitkeep` entries | Total tracked |
|:---|:---:|:---:|:---:|
| `experiments/lab1/` | **211** | 0 | 211 |
| `experiments/lab2/` | **94** | 0 | 94 |
| `experiments/lab3/` | **26** | 3 | **29** |

> [!NOTE]
> `.gitkeep` files are placeholder sentinels for empty directories (`checkpoints/`, `plots/`, `runs/`). They are not experiment artifacts. The previous audit's "29 committed files" for LAB3 includes these 3 `.gitkeep` entries; the actual experiment artifact count is **26**.

---

## 2. LAB2 Baseline Experiment Files (`backup/lab2-pre-consolidation`)

### 2.1 Experiment artifact paths in baseline

The LAB2 baseline (`backup/lab2-pre-consolidation`, tagged at commit `be2f0ac`) contained experiment artifacts under **two flat paths** (no `lab2/` sub-namespace):

| Baseline path | File count |
|:---|:---:|
| `experiments/plots/` | **62** |
| `experiments/results/` | **32** |
| `experiments/lab2/` | **0** (path did not exist) |
| `src/experiments/` | **25** (Python experiment scripts) |
| `agents/experiments/` | **15** (markdown planning docs) |

**Total baseline experiment artifacts:** 94 output files (`plots/` + `results/`) + 25 scripts + 15 doc markdown = **134 tracked entries** across experiment-related paths.

### 2.2 Location in consolidated branch (HEAD)

The consolidation remapped these paths into the `lab2/`-namespaced layout:

| Baseline path | Consolidated path | Count |
|:---|:---|:---:|
| `experiments/plots/` | `experiments/lab2/plots/` | 62 |
| `experiments/results/` | `experiments/lab2/results/` | 32 |
| `src/experiments/` | `src/lab2/experiments/` | 25 |
| `agents/experiments/*.md` | `docs/lab2/experiments/` | 15 |

### 2.3 Which commit introduced the LAB2 experiment files?

The LAB2 experiment files were **NOT** introduced by the named LAB2 consolidation commit `3ee9fb6`.

- **`67b1178`** (`chore: checkpoint interrupted phase 2 consolidation`) — This is the commit on `backup/phase2-interrupted` that added all 216 LAB2 files including `experiments/lab2/plots/` (62 files), `experiments/lab2/results/` (32 files), `src/lab2/experiments/` (25 files). This branch was later fast-forward merged into `consolidate/unified-main`.
- **`3ee9fb6`** (`fix(lab2): resolve test imports, root paths, and track consolidation handoff`) — This commit added `CONSOLIDATION_HANDOFF.md` and fixed test import paths only. It touched **zero** experiment files.

> [!IMPORTANT]
> `3ee9fb6` is the *fix* commit that landed on `consolidate/unified-main` after the merge. The actual experiment artifact ingestion occurred in `67b1178`.

---

## 3. LAB2 Consolidation Commit `3ee9fb6` — Experiment File Check

```
git show --name-status 3ee9fb6
```

Files modified in `3ee9fb6`:
- `A  CONSOLIDATION_HANDOFF.md`
- `M  src/lab2/data/config.py`
- `M  tests/lab2/conftest.py`
- `M  tests/lab2/test_build_model.py`
- `M  tests/lab2/test_config.py`
- `M  tests/lab2/test_loaders.py`

**Result:** Zero `experiments/` or `experiments/lab2/` files appear in commit `3ee9fb6`. This commit was a targeted fix commit, not the migration commit.

---

## 4. LAB3 Confirmation

### 4.1 Commit `a145b4a` tree

```
git ls-tree -r --name-only a145b4a -- experiments/lab3
```

Returns **29 entries** exactly, comprising:

| Sub-path | Files |
|:---|:---:|
| `experiments/lab3/checkpoints/.gitkeep` | 1 |
| `experiments/lab3/plots/.gitkeep` | 1 |
| `experiments/lab3/plots/*.png` | 11 |
| `experiments/lab3/results/README.md` | 1 |
| `experiments/lab3/results/*.json` | 8 |
| `experiments/lab3/results/*.txt` | 1 |
| `experiments/lab3/results/*.png` | 1 |
| `experiments/lab3/results/*.npy` | 4 |
| `experiments/lab3/runs/.gitkeep` | 1 |

**Total = 29** — matches the prior audit exactly.  
**Non-gitkeep artifacts = 26** (the `.gitkeep` sentinels are expected placeholder files for empty runtime directories).

The 29 LAB3 experiment files are confirmed tracked at HEAD and were introduced in a single atomic commit `a145b4a`.

---

## 5. Orphaned LAB2 Experiment Artifacts

### 5.1 File-by-file cross-reference

Comparing `experiments/plots/` (62 files) and `experiments/results/` (32 files) from the baseline against `experiments/lab2/plots/` and `experiments/lab2/results/` at HEAD:

| Metric | Count |
|:---|:---:|
| Baseline `experiments/plots/` files | 62 |
| HEAD `experiments/lab2/plots/` files | **62** |
| Baseline `experiments/results/` files | 32 |
| HEAD `experiments/lab2/results/` files | **32** |
| Files missing from HEAD | **0** |
| Orphaned files at unexpected paths | **0** |

All 94 baseline experiment output artifacts are present at HEAD under the correctly namespaced `experiments/lab2/` prefix. No files were left at the old flat path (`experiments/plots/`, `experiments/results/`) — these paths do not exist in `consolidate/unified-main`.

---

## 6. Consolidation Plan Evidence — Were Empty Directories Permitted?

Source: [`CONSOLIDATION_HANDOFF.md`](file:///C:/document/Study%20documents/Deeplearning_Course/CONSOLIDATION_HANDOFF.md)

### 6.1 Explicit LAB2 migration directive (line 25)

> **LAB2** | Phase 4, Step 4 | **COMMITTED (Interrupted)** | `67b1178` on `backup/phase2-interrupted` | 216 files: `src/lab2/`, `notebooks/lab2/`, `tests/lab2/`, `docs/lab2/`, `configs/lab2_data.yaml`, `data/lab2/processed/`, **`experiments/lab2/`**. Working tree clean.

This explicitly names `experiments/lab2/` as part of the 216-file LAB2 migration package — it was in scope and was committed.

### 6.2 Explicit LAB3 migration directive (line 100-101)

> ```bash
> git checkout LAB3_HuggingFace -- experiments/
> # Move plots and results into experiments/lab3/ (preserve experiments/lab1 and experiments/lab2)
> ```

This confirms `experiments/lab3/` was a required migration target.

### 6.3 No evidence permitting empty experiment directories

The document contains **no statement** that:
- LAB2 experiment artifacts could be omitted
- Generated artifacts could be discarded
- Empty experiment directories were acceptable
- Only source code needed migration

The only `.gitkeep` empty-directory pattern present in the repository is used for LAB3 runtime directories (`checkpoints/`, `runs/`) that are intentionally empty until experiments execute — not for artifact directories.

---

## 7. Final Classification

### LAB2 — `experiments/lab2/`

| Criterion | Finding |
|:---|:---|
| Baseline had tracked experiment artifacts? | **YES** — 94 output files (62 plots + 32 results) |
| Artifacts were supposed to migrate? | **YES** — explicitly named in CONSOLIDATION_HANDOFF.md §1 |
| Artifacts present in HEAD? | **YES** — 94 files, 1:1 filename match, zero missing |
| Migration commit? | `67b1178` (LAB2 phase checkpoint) |
| `.gitkeep`-only placeholders? | None — directory is fully populated |

**Classification: ✅ COMPLETE**

All 94 baseline LAB2 experiment artifacts were migrated into `experiments/lab2/` with correct namespacing. The consolidation commit `3ee9fb6` did not introduce them (it was a fix commit), but they were introduced by `67b1178` which was merged via fast-forward into `consolidate/unified-main` before `3ee9fb6`.

---

### LAB3 — `experiments/lab3/`

| Criterion | Finding |
|:---|:---|
| Baseline had tracked experiment artifacts? | **YES** — confirmed present in `backup/lab3-pre-consolidation` |
| Artifacts were supposed to migrate? | **YES** — explicitly named in CONSOLIDATION_HANDOFF.md §4 |
| Artifacts present in HEAD? | **YES** — 29 tracked entries (26 artifacts + 3 `.gitkeep`) |
| Migration commit? | `a145b4a` (single atomic LAB3 consolidation commit) |
| `.gitkeep`-only placeholders? | 3 entries for runtime dirs — expected and intentional |

**Classification: ✅ COMPLETE**

All 26 LAB3 experiment artifacts are tracked. The 3 `.gitkeep` files are intentional sentinels for runtime directories (`checkpoints/`, `plots/`, `runs/`) that scripts populate at runtime — this is consistent with LAB3's design and the directory structure specification.

---

## 8. Summary Report (Audit Items 1–12)

| # | Item | Finding |
|:---|:---|:---|
| 1 | `experiments/lab1` tracked file count | **211** |
| 2 | `experiments/lab2` tracked file count | **94** (excl. `.gitkeep`) |
| 3 | `experiments/lab3` tracked file count | **26** artifacts + 3 `.gitkeep` = **29** total |
| 4 | LAB2 baseline experiment files | 94 output artifacts (`experiments/plots/` × 62 + `experiments/results/` × 32) + 25 scripts in `src/experiments/` |
| 5 | LAB2 consolidated experiment files | 94 at `experiments/lab2/plots/` × 62 + `experiments/lab2/results/` × 32 |
| 6 | Missing/orphaned LAB2 files | **None** — 62/62 plots and 32/32 results accounted for |
| 7 | LAB3 confirmation | **Confirmed** — 29 entries at commit `a145b4a`, identical to HEAD |
| 8 | Evidence from consolidation plan | `CONSOLIDATION_HANDOFF.md` explicitly names `experiments/lab2/` and `experiments/lab3/` as required migration targets |
| 9 | Empty directories explicitly permitted? | **No** — plan contains zero language permitting omission of experiment artifacts |
| 10 | Final classification for LAB2 | **✅ COMPLETE** |
| 11 | Final classification for LAB3 | **✅ COMPLETE** |
| 12 | BLOCKER for main integration? | **No** — experiment provenance is fully established for both LAB2 and LAB3 |

---

> [!NOTE]
> **Key clarification on `3ee9fb6` vs `67b1178`:** The commit explicitly identified as the "LAB2 consolidation commit" (`3ee9fb6`) is a *post-merge fix commit* that corrects test imports and adds the handoff document. The actual LAB2 experiment migration occurred in `67b1178`, which was fast-forward merged into `consolidate/unified-main` before that fix was applied. The experiment files are correctly present in HEAD regardless of which commit label is used as the reference.
