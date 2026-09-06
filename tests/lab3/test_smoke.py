"""Smoke tests: verify the project skeleton imports and the layout is intact.

These pass out of the box right after bootstrapping — they are the first
line of defense that a fresh checkout is wired correctly.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_src_packages_importable():
    import src.lab3.data  # noqa: F401
    import src.lab3.eval  # noqa: F401
    import src.lab3.experiments  # noqa: F401
    import src.lab3.models  # noqa: F401
    import src.lab3.training  # noqa: F401
    import src.lab3.utils  # noqa: F401


def test_core_directories_exist():
    for rel in (
        # Shared repo-level agent knowledge-base dirs
        "agents/rules",
        "agents/templates",
        "configs",
        # LAB3 data root dirs
        "data/lab3/processed",
        "notebooks",
        # LAB3-namespaced experiment dirs
        "experiments/lab3/runs",
        "experiments/lab3/results",
        "experiments/lab3/plots",
        "experiments/lab3/checkpoints",
        "tests",
        # LAB3-specific agent content migrated to docs/lab3/
        "docs/lab3/phases",
        "docs/lab3/progress",
        "docs/lab3/experiments",
        "docs/lab3/bugs",
        "docs/lab3/plans",
    ):
        assert (PROJECT_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_agent_knowledge_base_files_exist():
    for rel in (
        # Shared repo-level rule/template files
        "agents/README.md",
        "agents/rules/FOLDER_STRUCTURE.md",
        "agents/rules/LOGGING_CHECKPOINT_RULES.md",
        "agents/rules/RESULTS_REPORTING.md",
        "agents/rules/CODEBASE_AUDIT.md",
        "agents/rules/NAMING_CONVENTION.md",
        "agents/templates/SMOKE_TEST_CHECKLIST.md",
        "agents/templates/PROJECT_ROADMAP_TEMPLATE.md",
        # LAB3-specific agent docs migrated to docs/lab3/
        "docs/lab3/PURPOSE.md",
        "docs/lab3/OVERVIEW.md",
        "docs/lab3/README.md",
    ):
        assert (PROJECT_ROOT / rel).is_file(), f"missing file: {rel}"
