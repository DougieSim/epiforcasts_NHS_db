# First Run Guide: Out-of-the-Box in 5 to 10 Minutes

## Audience
First-time users who want the easiest possible setup.

## Goal
Get from zero to a running dashboard with the fewest decisions.

This project uses [uv](https://docs.astral.sh/uv/). Install it first, then:

## Option 1: Easiest one-liner

Run this:

```bash
uv sync
uv run epiforcasts dev
```

What it does:

1. creates synthetic data;
2. runs one offline inference pass;
3. opens the fast dashboard.

Use this when:

1. you just want to see it working;
2. you are checking your environment quickly.

## Option 2: Slightly more control

Run these commands one by one:

```bash
uv sync
uv run epiforcasts generate
uv run epiforcasts daemon --once
uv run epiforcasts dashboard
```

What this gives you:

1. clearer understanding of each stage;
2. full dashboard rather than the minimal fast one.

## Option 3: Continuous background updates

Run in two terminals:

Terminal A:

```bash
uv run epiforcasts daemon
```

Terminal B:

```bash
uv run epiforcasts dashboard --fast
```

Use this when:

1. you are demoing repeatedly;
2. you want automatic refresh behaviour in the background.

## Quick checks

If something looks wrong, run:

```bash
uv run epiforcasts cache status
uv run epiforcasts cache check
```

## Most common first-run problems

1. Missing dependencies:
Run uv sync again.
2. Cache not ready:
Run uv run epiforcasts daemon --once.
3. Slow first inference:
This is expected; later runs are faster with cache.

## Which option should I pick?

1. New and curious: Option 1.
2. Want to learn the flow: Option 2.
3. Preparing repeat demos: Option 3.

## Related links

1. full runbook: RUNBOOK.md
2. compiler notes: PYTENSOR_COMPILER.md
3. lay-person guide: ../10-product/LAYPERSON_GUIDE.md
4. docs home: ../README.md
