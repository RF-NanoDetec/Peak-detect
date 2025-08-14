#!/usr/bin/env python
"""Generate PDF documentation from Markdown.

This helper tries to use `pypandoc` (wrapper around Pandoc). If Pandoc is not
available it exits with a helpful error message.

Usage:
    python scripts/generate_docs.py

The resulting PDF is written to docs/user_manual.pdf
"""

from pathlib import Path
import sys
import subprocess
from datetime import date
import tempfile

MARKDOWN_PATH = Path(__file__).resolve().parent.parent / "docs" / "user_manual.md"
PDF_PATH = MARKDOWN_PATH.with_suffix(".pdf")

# Metadata
APP_VERSION = None
try:
    # Import lazily to avoid pulling heavy deps if not needed
    from config.environment import APP_VERSION as _APP_VERSION

    APP_VERSION = _APP_VERSION
except Exception:
    APP_VERSION = "dev"


def substitute_placeholders(src_markdown: Path) -> Path:
    """Return a temporary markdown file with {{ placeholders }} replaced."""
    md_text = src_markdown.read_text(encoding="utf-8")
    today = date.today().isoformat()
    md_text = (
        md_text.replace("{{ DATE }}", today)
        .replace("{{ APP_VERSION }}", APP_VERSION)
    )

    tmp_file = Path(tempfile.mkstemp(suffix=".md", prefix="user_manual_tmp_")[1])
    tmp_file.write_text(md_text, encoding="utf-8")
    return tmp_file


def convert_with_pandoc():
    """Convert markdown to PDF using Pandoc via subprocess."""
    # Prefer xelatex for full Unicode support; fall back to pdflatex if not available
    preferred_engine = "xelatex"
    try:
        subprocess.check_call([preferred_engine, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        preferred_engine = "pdflatex"  # Fallback, may still fail on exotic chars

    # Try fancy Eisvogel template first, fall back to default
    template = "eisvogel"
    try:
        subprocess.check_call(["kpsewhich", "eisvogel.latex"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        template = "default"
        # default template needs fontspec manually activated for xelatex; pandoc adds automatically when mainfont variable present

    tmp_md = substitute_placeholders(MARKDOWN_PATH)
    
    # Change to docs directory so relative paths work
    import os
    original_dir = os.getcwd()
    os.chdir(MARKDOWN_PATH.parent)

    cmd = [
        "pandoc",
        str(tmp_md),
        "-o",
        str(PDF_PATH),
        "--from=markdown",
        "--toc",
        "--toc-depth=2",
        "--highlight-style=tango",
        "--template",
        template,
        "--metadata",
        f"title=Peak Analysis Tool User Manual",
        "--metadata",
        f"version={APP_VERSION}",
        f"--pdf-engine={preferred_engine}",
        "--citeproc",
        "--dpi",
        "300",
        "--bibliography",
        str(MARKDOWN_PATH.parent / "references.bib"),
        "--csl",
        str(MARKDOWN_PATH.parent / "nature.csl"),
    ]

    # Don't specify fonts on Windows unless we're sure they exist
    # This avoids font errors with MiKTeX
    if preferred_engine == "xelatex" and sys.platform != "win32":
        cmd.extend(
            [
                "--variable", "mainfont=DejaVu Serif",
                "--variable", "sansfont=DejaVu Sans",
                "--variable", "monofont=DejaVu Sans Mono",
            ]
        )

    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    finally:
        # Change back to original directory
        os.chdir(original_dir)

    # cleanup tmp
    try:
        tmp_md.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass


def main():
    if not MARKDOWN_PATH.exists():
        print("Markdown file not found:", MARKDOWN_PATH)
        sys.exit(1)

    # Try to run pandoc directly
    try:
        subprocess.check_call(["pandoc", "--version"], stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Pandoc executable not found. Please install Pandoc (https://pandoc.org) \
              and ensure it is on your PATH.")
        sys.exit(1)

    try:
        convert_with_pandoc()
    except subprocess.CalledProcessError as e:
        print("Error while converting with Pandoc:", e)
        sys.exit(e.returncode)

    print("PDF generated at", PDF_PATH)


if __name__ == "__main__":
    main() 