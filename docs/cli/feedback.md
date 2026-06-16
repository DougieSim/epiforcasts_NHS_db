# `epiforcasts feedback`

**Standalone script:** `epiforcasts-feedback`

Record **one Trust/ICB utility feedback session** with the required linkage
fields, appending a structured entry to the feedback log
(`docs/90-changelog/logs/TRUST_ICB_UTILITY_FEEDBACK_LOG.md`). All fields except
`--data-period` are required.

## Usage

```bash
uv run epiforcasts feedback \
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
```

## Options

| Option | Type | Required | Allowed / default | Description |
| --- | --- | --- | --- | --- |
| `--organisation-type` | choice | yes | `Trust` \| `ICB` \| `Joint` | Type of organisation. |
| `--organisation-name` | text | yes | — | Organisation name. |
| `--session-type` | choice | yes | `huddle` \| `review` \| `workshop` | Session format. |
| `--data-period` | text | no | `not specified` | Data period reviewed. |
| `--question` | text | yes | — | Operational question being answered. |
| `--signal` | text | yes | — | Signal/view used. |
| `--action` | text | yes | — | Action considered. |
| `--usefulness` | int | yes | `1`–`5` | Usefulness score. |
| `--timeliness` | int | yes | `1`–`5` | Timeliness score. |
| `--clarity` | int | yes | `1`–`5` | Clarity score. |
| `--confidence` | int | yes | `1`–`5` | Confidence in interpretation. |
| `--interpretation-risks` | text | yes | — | Interpretation risks observed. |
| `--change` | text | yes | — | What should change in UI/model/docs. |
| `--owner` | text | yes | — | Owner of the follow-up. |
| `--target-date` | text | yes | — | Target date for the change. |
| `--change-link` | text | yes | — | Link to the resulting commit/changelog item. |
| `--help` | flag | — | — | Show help and exit. |

## Behaviour

- Errors if the feedback log file is missing.
- Score options enforce the `1–5` integer range; click rejects out-of-range values.

## Related

- [`evidence`](evidence.md) — evidence cycles complement utility feedback for governance.
