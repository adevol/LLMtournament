from pathlib import Path
import os
import shutil

from invoke import task
from invoke.exceptions import Exit


@task
def install_hooks(c):
    repo_root = Path(__file__).resolve().parent
    hook_source = repo_root / "scripts" / "git-hooks" / "commit-msg"
    if not hook_source.exists():
        raise Exit(f"Missing hook template: {hook_source}")

    hook_dest = repo_root / ".git" / "hooks" / "commit-msg"
    hook_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hook_source, hook_dest)
    os.chmod(hook_dest, 0o755)
    print(f"Installed commit-msg hook to {hook_dest}")
