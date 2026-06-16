"""
Append one Trust/ICB utility feedback entry with required linkage fields.

Usage example:
    epiforcasts-feedback \
      --organisation-type Trust \
      --organisation-name "Example NHS Trust" \
      --session-type review \
      --usefulness 4 --timeliness 4 --clarity 4 --confidence 4 \
      --question "Should escalation staffing start earlier next week?" \
      --signal "risk trajectory by ICB" \
      --action "start pre-escalation rota" \
      --interpretation-risks "none" \
      --change "add uncertainty explainer text in dashboard" \
      --owner "Ops Analytics Lead" \
      --target-date "2026-06-15" \
      --change-link "docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md"
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import click


LOG_PATH = Path("docs/90-changelog/logs/TRUST_ICB_UTILITY_FEEDBACK_LOG.md")


def _git_short_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


_SCORE = click.IntRange(1, 5)


@click.command(name="feedback")
@click.option("--organisation-type", type=click.Choice(["Trust", "ICB", "Joint"]), required=True)
@click.option("--organisation-name", required=True)
@click.option("--session-type", type=click.Choice(["huddle", "review", "workshop"]), required=True)
@click.option("--data-period", default="not specified", show_default=True)
@click.option("--question", required=True)
@click.option("--signal", required=True)
@click.option("--action", required=True)
@click.option("--usefulness", type=_SCORE, required=True)
@click.option("--timeliness", type=_SCORE, required=True)
@click.option("--clarity",    type=_SCORE, required=True)
@click.option("--confidence", type=_SCORE, required=True)
@click.option("--interpretation-risks", required=True)
@click.option("--change", required=True)
@click.option("--owner", required=True)
@click.option("--target-date", required=True)
@click.option("--change-link", required=True)
def main(
    organisation_type: str,
    organisation_name: str,
    session_type: str,
    data_period: str,
    question: str,
    signal: str,
    action: str,
    usefulness: int,
    timeliness: int,
    clarity: int,
    confidence: int,
    interpretation_risks: str,
    change: str,
    owner: str,
    target_date: str,
    change_link: str,
) -> None:
    """Record one Trust/ICB utility feedback session."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing feedback log: {LOG_PATH}")

    now = dt.datetime.now(dt.timezone.utc)
    commit_hash = _git_short_hash()

    entry = f"""\n\n## {now.date()} ({organisation_type} utility session)\n\n- Date/time (UTC): {now.isoformat().replace('+00:00', 'Z')}\n- Organisation type: {organisation_type}\n- Organisation name: {organisation_name}\n- Session type: {session_type}\n- Commit hash and app version: {commit_hash}\n- Data period reviewed: {data_period}\n\n### Decision-support questions\n\n- Which local operational question was being answered? {question}\n- Which signal/view was used? {signal}\n- What action was considered? {action}\n- Was uncertainty understood and usable? yes/no (capture in session notes)\n\n### Utility outcome\n\n- Usefulness score (1-5): {usefulness}\n- Timeliness score (1-5): {timeliness}\n- Clarity score (1-5): {clarity}\n- Confidence in interpretation (1-5): {confidence}\n- Any interpretation risks observed: {interpretation_risks}\n\n### Follow-up\n\n- What should change in UI/model/docs? {change}\n- Owner: {owner}\n- Target date: {target_date}\n- Link to resulting commit/changelog item: {change_link}\n"""

    content = LOG_PATH.read_text(encoding="utf-8")
    if "No Trust/ICB utility feedback sessions logged yet." in content:
        content = content.replace("No Trust/ICB utility feedback sessions logged yet.", "At least one session has now been logged. See entries below.")
        LOG_PATH.write_text(content, encoding="utf-8")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)

    print(f"[OK] Feedback entry appended to {LOG_PATH}")


if __name__ == "__main__":
    main()
