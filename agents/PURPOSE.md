# PURPOSE.md — Project Brief & Requirements

- **Motivation/Background**: This file is the source of truth for *why* the
  project exists. Every phase doc, experiment, and progress entry derives from
  it, so a vague or wrong PURPOSE.md propagates into everything downstream.
- **Purpose**: Document the original brief: the problem, success criteria,
  scope boundaries, constraints, and audience.
- **Overview Pipeline**: Write/import the brief here → the agent runs a
  clarifying interview → you review and lock the final version → the roadmap
  is generated from it (see
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) Step 2–3).
- **Detailed Plan**: §1 brief; §2 clarifying answers; §3 locked objective.
- **References**: `HOW_TO_SETUP_AI_AGENT.md`, `OVERVIEW.md`,
  `templates/PROJECT_ROADMAP_TEMPLATE.md`.

---

## 1. Original Brief


Practice 3 - Get started with Hugging Face

Exercise 1: Sentiment Analysis with Hugging Face

1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

Exercise 2: Finetuning a Pretrained Model for Binary Text Classification

In this exercise, you will:

1. Install the necessary Hugging Face libraries (transformers, datasets, evaluate).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

## 2. Clarifying Answers

- **Problem/motivation**: Graded coursework for the deep learning course. The
  project delivers hands-on Hugging Face skills — loading pretrained models
  from the Hub, tokenization, and the Trainer-based finetuning workflow — that
  are missing without it.
- **Success criteria**: All steps of both exercises run end-to-end. Exercise
  2's finetuned model is evaluated on the IMDB test split; accuracy is reported
  with full 5W1H context (no hard threshold). Run artifacts (config, history,
  checkpoints, logs) auto-persist per
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md).
- **Scope boundaries**: Ex 1 uses a standard HF sentiment-analysis pipeline on
  an English sentence; Ex 2 uses IMDB + `distilbert-base-uncased` only. Out of
  scope: other datasets/models, non-binary classification, deployment/serving,
  hyperparameter sweeps, non-English samples.
- **Constraints**: Multi-GPU machine environment (1 machine with 8GB VRAM, 1 machine with 4GB VRAM) — target ≤3.5GB usage (strict ceiling 4GB for compatibility, ceiling 8GB on higher machines). English-only sample sentences.
- **Audience/context**: Coursework submission — scripts under `src/`, a
  demo/analysis notebook, and repo-compliant run artifacts under
  `experiments/runs/`.

## 3. Locked Objective

Complete Practice 3 as graded coursework: (1) run sentiment analysis with a
pretrained Hugging Face pipeline model on a sample English sentence, and (2)
finetune `distilbert-base-uncased` on the IMDB binary sentiment dataset using
the HF `Trainer`, then evaluate and report test accuracy with full 5W1H
context. All training runs from scripts under `src/` with artifacts
(checkpoints, config, history, logs) auto-persisted per repo rules, targeting
≤3.5GB VRAM (strictly fitting within 4GB VRAM machines).

---

## References

- [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) — the setup workflow
- [OVERVIEW.md](OVERVIEW.md) — project plan derived from this brief
