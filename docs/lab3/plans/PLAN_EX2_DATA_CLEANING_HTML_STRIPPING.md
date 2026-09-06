# TECHNICAL PLAN: IMDB Data Cleaning & HTML Artifact Stripping (Approach 1)

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


**Document ID:** `PLAN-EX2-DATA-CLEANING-HTML-STRIPPING`  
**Date:** 2026-08-15  
**Author:** bush-le + Antigravity AI Pair Programmer  
**Scope:** Phase 3 Feature Engineering & Data Preprocessing Pipeline  
**Target Goal:** Eliminate HTML line break artifacts (`<br />`, `<br/>`, `<br>`) across 50,000 IMDB reviews to reclaim 15–35 context tokens per review, improve tokenization efficiency, and prevent review conclusion truncation under max_length 512 constraint.  
**Status:** Implemented & Verified in Pipeline  
**Report Link:** [`agents/experiments/EX7_HTML_DATA_CLEANING_REPORT.md`](../experiments/EX7_HTML_DATA_CLEANING_REPORT.md)  

---

## 📌 1. Background & Problem Statement

### 1.1 Root Cause of HTML Noise in IMDB Dataset
The canonical `stanfordnlp/imdb` dataset was constructed via raw web-scraping from IMDb user review boards (Maas et al., 2011). As a byproduct of HTML parsing, raw text reviews retain unconverted HTML break tags:
- `<br />` (standard self-closing tag)
- `<br/>` (unspaced self-closing tag)
- `<br>` (unclosed line break)
- Consecutive duplicates: `<br /><br />` (used for paragraph separation)

### 1.2 Impact on Transformer Tokenization
When processed by WordPiece / BPE tokenizers (e.g. `distilbert-base-uncased`):
1. A single `<br />` is fragmented into **4 distinct sub-word tokens**: `['<', 'br', '/', '>']`.
2. A paragraph break `<br /><br />` consumes **8 tokens** of the sequence budget.
3. In 58.66% of IMDB reviews, HTML tags consume **15 to 35 tokens**, pushing critical sentiment conclusion paragraphs outside the sequence window (`max_length = 512`).

---

## 🎯 2. Implementation Specifications

### 2.1 Function Specification (`src/data/prepare_imdb.py`)

```python
import re

def clean_text(text: str) -> str:
    """Strip HTML line breaks (<br />, <br>) and collapse redundant whitespace.
    
    Args:
        text (str): Raw review text from dataset.
        
    Returns:
        str: Sanitized text with HTML break tags replaced by spaces and collapsed whitespace.
    """
    if not isinstance(text, str):
        return text
    # Replace all variations of HTML line breaks with a single space
    cleaned = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    # Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
```

### 2.2 Integration into Tokenization Pipeline
- **File:** [`src/data/prepare_imdb.py`](../../src/data/prepare_imdb.py)
- In `tokenize_split(examples, tokenizer, max_length)`: Apply `clean_text` to `examples["text"]` prior to calling `tokenizer(...)`.
- **Export:** Expose `clean_text` in [`src/__init__.py`](../../src/__init__.py) for interactive inspection and testing.

---

## 🧪 3. Verification & Execution Checklist

| Task Item | Target File / Artifact | Success Criteria | Status |
|:---|:---|:---|:---:|
| **1. Text Cleaner Function** | `src/data/prepare_imdb.py` | Regex handles all `<br>`, `<br/>`, `<br />` variants | ✅ Done |
| **2. Pipeline Integration** | `src/data/prepare_imdb.py` | `tokenize_split` applies `clean_text` before tokenizer | ✅ Done |
| **3. Lazy Package Export** | `src/__init__.py` | `from src import clean_text` works smoothly | ✅ Done |
| **4. Unit Tests Suite** | `tests/test_data_cleaning.py` | 100% pass on whitespace & HTML break cases | ✅ Done |
| **5. Notebook Inspection Cell** | `notebooks/02_ex2_finetune.ipynb` | Cell Step 1.1 displays statistics & visual donut/bar chart | ✅ Done |
| **6. Cache Re-Tokenization** | `data/processed/imdb_tokenized_512` | 50,000 samples tokenized and saved to disk | ✅ Done |

