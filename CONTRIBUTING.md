# Contributing

Thanks for your interest in contributing. This project uses Conventional Commit
prefixes and git hooks to keep history consistent.

## Development Setup

```bash
uv sync --extra dev
```

Useful commands:

```bash
uv run invoke lint
uv run invoke format-check
uv run invoke test
uv run invoke ci
```

## Commit Messages

Commit messages should start with one of the following prefixes (optional scope
allowed): `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `perf:`,
`build:`, `ci:`, `style:`, `revert:`.

Examples:

```text
feat: add swiss pairing strategy
fix: handle missing judge output
docs(readme): update quick start
```

## Git Hooks (Conventional Commits)

Install the commit-msg hook (optional but recommended; hooks are not versioned):

```bash
uv run invoke install-hooks
```

Verify it's active:

```bash
ls .git/hooks/commit-msg
```

CI does not enforce commit message format; the hook is local and optional.

## Pre-commit (Ruff + Ruff Format)

Install and run pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
