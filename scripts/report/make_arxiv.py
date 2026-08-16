"""Assemble the arXiv submission tarball at report/build/arxiv.tar.gz.

Run from the repo root: uv run poe report-arxiv
"""

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REPORT = Path(__file__).resolve().parents[2] / "report"
BUILD = REPORT / "build"
STAGE = BUILD / "arxiv"
TARBALL = BUILD / "arxiv.tar.gz"

INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _sources() -> list[Path]:
    return [REPORT / "main.tex", *sorted((REPORT / "sections").glob("*.tex"))]


def _figures(sources: list[Path]) -> list[Path]:
    found: dict[Path, None] = {}
    for source in sources:
        for name in INCLUDEGRAPHICS.findall(source.read_text(encoding="utf-8")):
            found[REPORT / name] = None
    missing = [path for path in found if not path.exists()]
    if missing:
        raise RuntimeError(f"figures referenced but not on disk: {missing}")
    return list(found)


def _stage(path: Path, relative: str) -> None:
    dest = STAGE / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def main() -> None:
    # `-r` is needed because latexmk reads .latexmkrc from the working directory,
    # which it changes only afterwards, when it acts on `-cd`.
    latexmk = ["latexmk", "-pdf", "-r", str(REPORT / ".latexmkrc"), "-cd"]
    subprocess.run([*latexmk, str(REPORT / "main.tex")], check=True)

    # arXiv never runs bibtex, so the submission carries the compiled bibliography
    # instead of references.bib.
    bbl = BUILD / "main.bbl"
    if not bbl.exists():
        raise RuntimeError(f"{bbl} is missing, so latexmk did not run bibtex")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    sources = _sources()
    for path in [*sources, *_figures(sources)]:
        _stage(path, path.relative_to(REPORT).as_posix())
    _stage(bbl, "main.bbl")

    files = sorted(path for path in STAGE.rglob("*") if path.is_file())
    with tarfile.open(TARBALL, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(STAGE).as_posix())

    print(f"{TARBALL} ({TARBALL.stat().st_size / 1024:.0f} KiB)")
    for path in files:
        print(f"  {path.relative_to(STAGE).as_posix()}")


if __name__ == "__main__":
    main()
