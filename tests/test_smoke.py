"""Smoke tests: verify the project skeleton imports and the layout is intact.

These pass out of the box right after bootstrapping — they are the first
line of defense that a fresh checkout is wired correctly.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_src_packages_importable():
    import src.data  # noqa: F401
    import src.eval  # noqa: F401
    import src.experiments  # noqa: F401
    import src.models  # noqa: F401
    import src.training  # noqa: F401
    import src.utils  # noqa: F401


def test_core_directories_exist():
    for rel in (
        "agents/rules",
        "agents/templates",
        "agents/phases",
        "agents/progress",
        "agents/experiments",
        "agents/bugs",
        "agents/references",
        "configs",
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "experiments/runs",
        "experiments/results",
        "experiments/plots",
        "experiments/checkpoints",
        "tests",
    ):
        assert (PROJECT_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_agent_knowledge_base_files_exist():
    for rel in (
        "agents/README.md",
        "agents/PURPOSE.md",
        "agents/OVERVIEW.md",
        "agents/HOW_TO_SETUP_AI_AGENT.md",
        "agents/rules/FOLDER_STRUCTURE.md",
        "agents/rules/LOGGING_CHECKPOINT_RULES.md",
        "agents/rules/RESULTS_REPORTING.md",
        "agents/rules/CODEBASE_AUDIT.md",
        "agents/rules/NAMING_CONVENTION.md",
        "agents/templates/SMOKE_TEST_CHECKLIST.md",
        "agents/templates/PROJECT_ROADMAP_TEMPLATE.md",
    ):
        assert (PROJECT_ROOT / rel).is_file(), f"missing file: {rel}"
