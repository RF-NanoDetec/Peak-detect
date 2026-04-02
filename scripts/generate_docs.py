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

MARKDOWN_DIR = Path(__file__).resolve().parent.parent / "docs"
PUBLIC_DOCS_DIR = Path(__file__).resolve().parent.parent / "ui-web" / "public" / "docs"

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

    tmp_file = Path(tempfile.mkstemp(suffix=".md", prefix=f"{src_markdown.stem}_tmp_")[1])
    tmp_file.write_text(md_text, encoding="utf-8")
    return tmp_file


def convert_with_pandoc(markdown_path: Path, output_pdf_path: Path, title: str):
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

    tmp_md = substitute_placeholders(markdown_path)
    
    # Change to docs directory so relative paths work
    import os
    original_dir = os.getcwd()
    os.chdir(markdown_path.parent)

    # Ensure output directory exists
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pandoc",
        str(tmp_md),
        "-o",
        str(output_pdf_path),
        "--from=markdown",
        "--toc",
        "--toc-depth=2",
        "--highlight-style=tango",
        "--template",
        template,
        "--metadata",
        f"title={title}",
        "--metadata",
        f"version={APP_VERSION}",
        f"--pdf-engine={preferred_engine}",
        "--citeproc",
        "--dpi",
        "300",
        # Only add bibliography if files exist
        # "--bibliography", str(markdown_path.parent / "references.bib"),
        # "--csl", str(markdown_path.parent / "nature.csl"),
    ]
    
    if (markdown_path.parent / "references.bib").exists():
        cmd.extend(["--bibliography", str(markdown_path.parent / "references.bib")])
    
    if (markdown_path.parent / "nature.csl").exists():
        cmd.extend(["--csl", str(markdown_path.parent / "nature.csl")])

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

    print(f"Converting {markdown_path.name} -> {output_pdf_path.name}")
    # print("Running:", " ".join(cmd))
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
    # Try to run pandoc directly
    try:
        subprocess.check_call(["pandoc", "--version"], stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Pandoc executable not found. Please install Pandoc (https://pandoc.org) \
              and ensure it is on your PATH.")
        sys.exit(1)

    files_to_convert = [
        (MARKDOWN_DIR / "user_manual.md", PUBLIC_DOCS_DIR / "user_manual.pdf", "Peak Analysis Tool User Manual"),
        (MARKDOWN_DIR / "mathematical_reference.md", PUBLIC_DOCS_DIR / "mathematical_reference.pdf", "Mathematical Reference"),
    ]

    for md_path, pdf_path, title in files_to_convert:
        if not md_path.exists():
            print(f"Warning: {md_path} not found, skipping.")
            continue
        
        try:
            convert_with_pandoc(md_path, pdf_path, title)
            print(f"Success: Generated {pdf_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {md_path.name}: {e}")

if __name__ == "__main__":
    main() 