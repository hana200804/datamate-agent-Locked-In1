"""Package the publishable source tree for Colab; run from any directory."""
from pathlib import Path
import argparse
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "dist/repo.zip")
    a = p.parse_args()
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(filter(None, files))):
            src = ROOT / name
            if src.is_file() and not src.is_symlink() and src.resolve() != a.out.resolve():
                archive.write(src, name)
    print(a.out)


if __name__ == "__main__":
    main()
