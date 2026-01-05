import os
import shutil
from pathlib import Path

from invoke import task
from invoke.exceptions import Exit


@task
def install_hooks(_):
    repo_root = Path(__file__).resolve().parent
    hook_source = repo_root / "scripts" / "git-hooks" / "commit-msg"
    if not hook_source.exists():
        raise Exit(f"Missing hook template: {hook_source}")

    hook_dest = repo_root / ".git" / "hooks" / "commit-msg"
    hook_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hook_source, hook_dest)
    os.chmod(hook_dest, 0o700)
    print(f"Installed commit-msg hook to {hook_dest}")


@task
def lint(c):
    c.run("ruff check .")


@task
def format_check(c):
    c.run("ruff format --check .")


@task
def test(c):
    c.run("pytest")


@task
def ci(c):
    lint(c)
    format_check(c)
    test(c)
