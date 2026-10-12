"""Build the TMLR submission PDF and its anonymized code supplement."""

import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
OUT = REPORT / "build" / "tmlr"
ZIP = OUT / "supplement.zip"

# report/ and docs/ name the authors throughout, and nothing in them is needed to run the code.
INCLUDE = [
    "src",
    "configs",
    "scripts",
    "tests",
    "data",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    ".python-version",
]

REPLACEMENTS = [
    (re.compile(r"suxrobgm@gmail\.com"), "anonymous@example.com"),
    (re.compile(r"suxrobgm/cpjump1-dinov3-features"), "anonymous/cpjump1-dinov3-features"),
]

IDENTIFIERS = re.compile(
    r"suxrob|sukhrob|ilyosbekov|gajjar|rongfei|jin\.ron|northeastern|roux", re.IGNORECASE
)

# The README links the docs site and ends with an author list; the supplement keeps neither.
README_DROP = re.compile(r"suxrobgm\.net")
README_AUTHORS = re.compile(r"^## Authors\n.*?(?=^---$|\Z)", re.MULTILINE | re.DOTALL)

README_REVIEW_NOTE = """\
## Note for Reviewers

The pre-extracted DINOv3 features are hosted under an account that would identify the authors,
so `morphoclip features download` will not work during review. The archive will be made public
on acceptance. Until then, the features can be rebuilt from the public CPJUMP1 images with
`morphoclip data fetch` followed by `morphoclip features extract` (about 3 GB per plate across
51 plates; DINOv3 weights require accepting Meta's license on Hugging Face). The test suite
does not need the features: `uv run poe test`.

"""


def _tracked_files() -> list[str]:
    result = subprocess.run(
        # scripts/report/ builds the paper and lists the author identifiers it scrubs.
        ["git", "ls-files", "-z", "--", *INCLUDE, ":!scripts/report"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def _anonymize(name: str, text: str) -> str:
    if name == "README.md":
        text = README_AUTHORS.sub("", text)
        text = "".join(
            line for line in text.splitlines(keepends=True) if not README_DROP.search(line)
        )
        text = text.replace("## Quick Start\n", README_REVIEW_NOTE + "## Quick Start\n", 1)
    for pattern, replacement in REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def build_supplement() -> list[str]:
    """Zip the tracked runtime files with author identifiers removed.

    Returns:
        The archive member names.

    Raises:
        RuntimeError: If an identifier survives anonymization, or a binary file is included.
    """
    names = _tracked_files()
    leaks: list[str] = []
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            text = _read_text(ROOT / name)
            if text is None:
                raise RuntimeError(
                    f"{name} is not UTF-8 text, so it cannot be checked for identifiers"
                )
            text = _anonymize(name, text)
            leaks.extend(
                f"{name}:{number}: {line.strip()}"
                for number, line in enumerate(text.splitlines(), start=1)
                if IDENTIFIERS.search(line)
            )
            archive.writestr(f"morphoclip/{name}", text)
    if leaks:
        ZIP.unlink()
        raise RuntimeError("author identifiers left in the supplement:\n" + "\n".join(leaks))
    return names


def main() -> None:
    latexmk = ["latexmk", "-pdf", "-r", str(REPORT / "tmlr" / ".latexmkrc"), "-cd"]
    subprocess.run([*latexmk, str(REPORT / "tmlr" / "tmlr.tex")], check=True)

    pdf_text = subprocess.run(
        ["pdftotext", str(OUT / "tmlr.pdf"), "-"], check=True, capture_output=True
    ).stdout.decode("utf-8", errors="replace")
    if found := IDENTIFIERS.findall(pdf_text):
        raise RuntimeError(f"the anonymized PDF names the authors: {sorted(set(found))}")

    names = build_supplement()
    print(f"{OUT / 'tmlr.pdf'}")
    print(f"{ZIP} ({ZIP.stat().st_size / 1024:.0f} KiB, {len(names)} files)")


if __name__ == "__main__":
    main()
