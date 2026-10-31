"""`sextant guide` prints the English guide, and every topic the CLI points users to exists."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from sextant import cli

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "user"
PACKAGE = ROOT / "src" / "sextant"


@pytest.mark.parametrize("topic", cli.GUIDE_TOPICS)
def test_guide_prints_each_topic(topic: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["guide", topic]) == 0
    captured = capsys.readouterr()
    assert captured.out == (GUIDE / f"{topic}.md").read_text()
    assert captured.out.startswith("# ") and not captured.err


def test_guide_without_a_topic_prints_the_overview(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["guide"]) == 0
    assert capsys.readouterr().out == (GUIDE / "README.md").read_text()


def test_every_topic_has_a_page_and_every_page_is_a_topic() -> None:
    assert sorted(p.stem for p in GUIDE.glob("*.md")) == sorted(cli.GUIDE_TOPICS)


def test_every_guide_command_in_cli_messages_names_a_served_topic() -> None:
    named = set()
    for path in PACKAGE.rglob("*.py"):
        named.update(re.findall(r"sextant guide ([A-Za-z][\w-]*)", path.read_text()))
    # The runtime-unavailable hint and the legacy read-only hint must stay covered.
    assert {"runtime", "data-and-recovery"} <= named
    assert named <= set(cli.GUIDE_TOPICS), sorted(named - set(cli.GUIDE_TOPICS))


def test_every_guide_page_points_only_to_served_topics() -> None:
    for path in GUIDE.glob("*.md"):
        named = set(re.findall(r"sextant guide ([A-Za-z][\w-]*)", path.read_text()))
        assert named <= set(cli.GUIDE_TOPICS), (path.name, sorted(named - set(cli.GUIDE_TOPICS)))


def test_the_guide_is_written_in_english() -> None:
    for path in GUIDE.glob("*.md"):
        assert not re.search(r"[\u3400-\u9fff\uf900-\ufaff]", path.read_text()), path.name
