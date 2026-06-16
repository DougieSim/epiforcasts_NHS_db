# epiforecasts NHS Pressure Demo

Probabilistic, uncertainty-aware, synthetic-data demonstration for NHS-style system pressure analytics.

This repository is designed for safe experimentation, transparent communication, and collaborative engineering practice. It does not contain real patient-identifiable data.

## 1. Purpose

This project demonstrates how a Bayesian workflow can support conversation about system pressure in a way that:

1. shows uncertainty explicitly;
2. separates offline model inference from online user experience;
3. keeps governance and plain-language communication central.

## Architecture and evidence at a glance

This repository is structured as an evidence-to-decision support prototype.
Its central architecture choice is deliberate separation between:

1. offline probabilistic inference (compute-heavy);
2. online dashboard serving from cached artifacts (interaction-safe).

Why this matters:

1. uncertainty is communicated explicitly rather than hidden behind single-value outputs;
2. results are reproducible and auditable through persisted artifacts;
3. user experience is reliable because dashboards do not run MCMC at interaction time;
4. governance boundaries are explicit, reducing risk of over-interpretation.

Canonical architecture and rationale: [docs/20-architecture/README.md](docs/20-architecture/README.md)
Canonical technical evidence summary: [docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md](docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md)
Canonical governance controls: [docs/50-governance/GOVERNANCE_OVERVIEW.md](docs/50-governance/GOVERNANCE_OVERVIEW.md)

## 2. What is in scope

In scope:

1. synthetic weekly panel generation;
2. Bayesian model fitting using PyMC;
3. cached posterior serving to Streamlit applications;
4. documentation for technical and non-technical audiences;
5. prototype governance framing.

Out of scope:

1. production clinical decision support;
2. live NHS integration;
3. validated operational thresholds;
4. patient-level inference.

## 3. Quick start

This project uses [uv](https://docs.astral.sh/uv/). Install the environment and
run a development flow:

```bash
uv sync
uv run epiforcasts dev
```

Run inference once and then launch the full dashboard:

```bash
uv run epiforcasts daemon --once
uv run epiforcasts dashboard
```

If port `8501` is already in use, dashboard startup auto-falls back to the next free port (`8502`..`8510`) instead of failing.

Run a one-command local health check:

```bash
uv run epiforcasts health
```

All commands are subcommands of the unified `epiforcasts` CLI — run
`uv run epiforcasts --help` for the full list. Each also has a standalone
`epiforcasts-*` console script (e.g. `uv run epiforcasts-inference --fast`).

## Environment reliability notes

`uv sync` creates a reproducible `.venv` from `uv.lock`. Prefix commands with
`uv run` (or activate `.venv`) so they use that environment.

If you see PyTensor compiler warnings (`g++ not detected`), inference still runs
but may be much slower. Unlike the old Pixi setup, uv does not bundle a C++
compiler — install one at the system level (MSVC Build Tools on Windows,
`build-essential` on Linux, Xcode CLT on macOS). See
[docs/40-operations/PYTENSOR_COMPILER.md](docs/40-operations/PYTENSOR_COMPILER.md).

Dashboard startup reliability:

1. `uv run epiforcasts dashboard` (and `--fast`) use a safe launcher with automatic port fallback.
2. If fallback occurs, the selected port is printed in terminal output.

Evidence and confidence hardening automation:

1. Run one independent evidence cycle and auto-append a timestamped entry:

```bash
uv run epiforcasts evidence --run-inference-fast
```

2. Record at least one Trust/ICB utility session with explicit change linkage:

```bash
uv run epiforcasts feedback --organisation-type Trust --organisation-name "Example NHS Trust" --session-type review --question "Operational question" --signal "Signal used" --action "Action considered" --usefulness 4 --timeliness 4 --clarity 4 --confidence 4 --interpretation-risks "None" --change "UI/model/docs change" --owner "Owner" --target-date "2026-06-15" --change-link "docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md"
```

## Artifact policy

Generated inference artifacts are local runtime outputs and are not committed by default:

1. `posteriors.nc`
2. `posteriors_metadata.nc`
3. `.cache/`
4. `.artifacts/` (fallback outputs when primary files are locked)

Regenerate these via:

```bash
uv run epiforcasts daemon --once
uv run epiforcasts health
```

If `posteriors.nc` is locked by a running dashboard, inference now retries and then writes a timestamped fallback artifact instead of crashing.

For full operational guidance, read [docs/40-operations/RUNBOOK.md](docs/40-operations/RUNBOOK.md).
For a minimal first run, read [docs/40-operations/FIRST_RUN_DUMMIES.md](docs/40-operations/FIRST_RUN_DUMMIES.md).

## 4. Repository structure

Top-level hygiene policy:

1. Keep first-level folders intentional and stable (`docs/`, `EpiNow2/`, and canonical entry-point files only).
2. Route generated runtime byproducts to hidden working folders (`.cache/`, `.artifacts/`) rather than root.
3. Avoid adding ad-hoc top-level files unless they are durable project entry points.

Core Python components:

1. [generate_synthetic_data.py](generate_synthetic_data.py)
2. [bayesian_pressure_model.py](bayesian_pressure_model.py)
3. [run_inference.py](run_inference.py)
4. [inference_daemon.py](inference_daemon.py)
5. [cache_manager.py](cache_manager.py)
6. [app.py](app.py)
7. [app_fast.py](app_fast.py)

Primary documentation home:

1. [docs/README.md](docs/README.md)
2. [docs/10-product/LAYPERSON_GUIDE.md](docs/10-product/LAYPERSON_GUIDE.md)
3. [docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md](docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md)
4. [docs/70-reference/glossary.md](docs/70-reference/glossary.md)

## 5. Documentation taxonomy migration plan

The documentation set is being normalised into a stable taxonomy:

1. `docs/00-overview`: orientation, scope, document map, migration notes;
2. `docs/10-product`: user and stakeholder explainers;
3. `docs/20-architecture`: software architecture and component contracts;
4. `docs/30-model`: statistical and methodological documentation;
5. `docs/40-operations`: runbooks, deployment and support;
6. `docs/50-governance`: safety, assurance, compliance artefacts;
7. `docs/60-contributing`: contribution, style, review standards;
8. `docs/70-reference`: glossary, references, assumptions register;
9. `docs/80-decisions`: architecture and model decision records;
10. `docs/90-changelog`: release and change history.

Migration principles:

1. each topic has one canonical home;
2. old pages become short stubs with links, then retire;
3. every page includes Audience, Purpose, Scope, Assumptions, Limitations, Owner, and References;
4. cross-links are mandatory for discoverability and handover;
5. unresolved terms are added to the glossary before merge.

Detailed plan: [docs/00-overview/DOCS_TAXONOMY_MIGRATION_PLAN.md](docs/00-overview/DOCS_TAXONOMY_MIGRATION_PLAN.md).
High-level structure options: [docs/00-overview/FOLDER_STRUCTURE_OPTIONS.md](docs/00-overview/FOLDER_STRUCTURE_OPTIONS.md).

## 6. Collaboration standards

Before submitting changes:

1. read [CONTRIBUTING.md](CONTRIBUTING.md);
2. apply [docs/60-contributing/style-guide.md](docs/60-contributing/style-guide.md);
3. use shared terminology in [docs/70-reference/glossary.md](docs/70-reference/glossary.md);
4. support claims with evidence from [docs/70-reference/references.md](docs/70-reference/references.md).

## 7. Audience pathways

1. Non-technical readers: [docs/10-product/LAYPERSON_GUIDE.md](docs/10-product/LAYPERSON_GUIDE.md)
2. Delivery and governance teams: [docs/50-governance/GOVERNANCE_OVERVIEW.md](docs/50-governance/GOVERNANCE_OVERVIEW.md)
3. Engineers and analysts: [docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md](docs/30-model/TECHNICAL_SUMMARY_ADVANCED.md)
4. Operators: [docs/40-operations/RUNBOOK.md](docs/40-operations/RUNBOOK.md)

## 8. Current status

This repository is a prototype and communication artefact for collaborative design and safe experimentation. It is not production-authorised clinical software.

Defensibility references:

1. architecture and rationale: [docs/20-architecture/README.md](docs/20-architecture/README.md)
2. assumptions register: [docs/70-reference/assumptions-register.md](docs/70-reference/assumptions-register.md)
3. evidence references: [docs/70-reference/references.md](docs/70-reference/references.md)

Release tracking: [CHANGELOG.md](CHANGELOG.md)

## 9. Speculative direction

This section is intentionally forward-looking and non-committal. It describes a possible growth path if partners choose to develop this prototype further.

### Why this matters in the UK context

Winter pressure in England is driven by interacting signals across acute demand, respiratory burden, discharge friction, workforce pressure, and primary/community access constraints. A useful next step is local early-signal visibility that is:

1. probabilistic rather than binary;
2. explainable to operational leaders;
3. comparable across Trust and ICB footprints.

### Audience and engagement model

Primary audiences for a growth phase:

1. Trust operational teams (site operations, UEC leads, discharge teams);
2. ICB analytics and performance teams;
3. regional and national partners who coordinate planning assumptions.

Proposed engagement cycle:

1. agree local questions first (for example: where pressure is rising fastest, where uncertainty is widest);
2. co-design signal views with Trust and ICB users;
3. run short evidence reviews on whether signals were timely and useful;
4. tune presentation language before tuning model complexity.

### Data expansion path (speculative)

Potential near-term data enrichment, subject to permissions and governance:

1. UKHSA syndromic and respiratory surveillance indicators;
2. NHS England operational and urgent/emergency pressure indicators;
3. ONS and other public contextual datasets relevant to seasonal demand;
4. weather and environmental context where operationally meaningful.

Design principle: add sources only when they improve decision usefulness and can be explained clearly in assumptions and limitations.

### Evidence posture for growth

Any extension should preserve evidence discipline:

1. each new signal mapped to an explicit assumption in [docs/70-reference/assumptions-register.md](docs/70-reference/assumptions-register.md);
2. each external source recorded with provenance in [docs/70-reference/references.md](docs/70-reference/references.md);
3. each major direction change logged in [docs/80-decisions/README.md](docs/80-decisions/README.md) and [docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md](docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md).

9.5 confidence guardrails:

1. sustained multi-day green evidence trend is required (not same-day runs only);
2. at least one Trust/ICB utility feedback loop must be logged and linked to a resulting change.
3. use `log_evidence_run.py` for repeatable daily evidence entries and `record_trust_feedback.py` for structured utility traceability.

### How this broaches NHS long-range planning

A mature version of this approach can support priorities commonly associated with NHS long-range planning (including the evolving 10-year planning direction):

1. earlier intervention through earlier risk visibility;
2. better coordination across place, Trust, and ICB boundaries;
3. transparent use of data and analytics for operational planning;
4. clearer communication of uncertainty to avoid false certainty in winter escalation decisions.

This remains a prototype pathway, not a policy product.

Policy-facing PoC framing for Trust and ICB winter use:

1. [docs/50-governance/POC_POLICY_ALIGNMENT_UK.md](docs/50-governance/POC_POLICY_ALIGNMENT_UK.md)
