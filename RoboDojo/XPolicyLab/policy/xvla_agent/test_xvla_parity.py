"""Guard the no-review path against drifting from the X_VLA baseline."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "X_VLA"


def _without_codex_hooks(source: str) -> str:
    replacements = (
        ("\nfrom codex_review import CodexReviewer", ""),
        (
            "\n        self.codex_reviewer = (\n"
            "            CodexReviewer(self.model_cfg)\n"
            "            if self.model_cfg.get(\"codex_enabled\", False)\n"
            "            else None\n"
            "        )",
            "",
        ),
        (
            "\n        if self.codex_reviewer is not None and len(env_list) != 1:\n"
            "            raise ValueError(\n"
            "                \"xvla_agent Codex review supports one active rollout\"\n"
            "            )",
            "",
        ),
        (
            "\n        if self.codex_reviewer is not None:\n"
            "            observation = self._raw_by_env[resolved_env_idx]\n"
            "            try:\n"
            "                return self.codex_reviewer.review(observation, actions)\n"
            "            except Exception as exc:\n"
            "                return self.codex_reviewer.hold_after_error(observation, exc)",
            "",
        ),
        (
            "\n        if self.codex_reviewer is not None:\n"
            "            self.codex_reviewer.reset()",
            "",
        ),
    )
    for extension, baseline in replacements:
        assert extension in source, f"missing expected Codex extension: {extension!r}"
        source = source.replace(extension, baseline, 1)
    return source


def test_model_is_xvla_plus_codex_hooks() -> None:
    baseline = (BASE / "model.py").read_text(encoding="utf-8")
    agent = (HERE / "model.py").read_text(encoding="utf-8")
    assert _without_codex_hooks(agent) == baseline


def test_shared_runtime_files_match() -> None:
    for name in ("deploy.py", "save_images.py", "self_lock_guard.py"):
        assert (HERE / name).read_text(encoding="utf-8").rstrip() == (
            BASE / name
        ).read_text(encoding="utf-8").rstrip(), name


def test_no_review_defaults_match_xvla() -> None:
    baseline = (BASE / "deploy.yml").read_text(encoding="utf-8")
    agent = (HERE / "deploy.yml").read_text(encoding="utf-8")
    core, marker, codex = agent.partition("\n# Sparse Codex review.")
    assert marker and "codex_enabled: false" in codex
    assert core.replace(
        "policy_name: xvla_agent", "policy_name: X_VLA", 1
    ).rstrip() == baseline.rstrip()


if __name__ == "__main__":
    test_model_is_xvla_plus_codex_hooks()
    test_shared_runtime_files_match()
    test_no_review_defaults_match_xvla()
    print("xvla_agent no-review parity tests passed")
