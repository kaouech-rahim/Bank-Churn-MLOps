"""Convertit docs/RAPPORT.md en docs/RAPPORT.pdf (Markdown -> HTML -> PDF).

Utilise le paquet "markdown" (deja installe avec MLflow) et le navigateur
Microsoft Edge (ou Chrome) en mode headless : aucune dependance a ajouter.

Usage :
    poetry run python scripts/build_report_pdf.py
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import markdown

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt;
       line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; color: #1e3a8a; }
h2 { font-size: 13pt; color: #1e3a8a; border-bottom: 1px solid #1e3a8a;
     padding-bottom: 2pt; margin: 14pt 0 5pt 0; }
h3 { font-size: 11pt; margin: 9pt 0 3pt 0; }
p, li { margin: 3pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0; font-size: 8.6pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #94a3b8; padding: 2.5pt 4pt; text-align: left; }
th { background: #e2e8f0; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f5f9; }
pre { background: #f1f5f9; padding: 5pt; font-size: 8pt; white-space: pre-wrap;
      page-break-inside: avoid; }
img { max-width: 100%; display: block; margin: 4pt auto; page-break-inside: avoid; }
.capture { border: 1.5px dashed #b45309; background: #fffbeb; padding: 6pt;
           margin: 6pt 0; font-size: 9pt; page-break-inside: avoid; }
.row { display: flex; gap: 8pt; } .row img { width: 49%; }
.small { font-size: 8.6pt; color: #475569; }
"""


def find_browser() -> Optional[str]:
    """Renvoie le chemin d'Edge ou de Chrome, ou None."""
    for path in BROWSERS:
        if Path(path).exists():
            return path
    return shutil.which("msedge") or shutil.which("chrome")


def build_pdf(md_path: Path, pdf_path: Path) -> Path:
    """Genere le PDF a partir du Markdown (les images restent relatives a docs/).

    Raises:
        RuntimeError: Si aucun navigateur compatible n'est trouve.
    """
    browser = find_browser()
    if browser is None:
        raise RuntimeError("Edge ou Chrome introuvable : impossible de generer le PDF.")

    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "md_in_html"],
    )
    base = md_path.parent.resolve().as_uri() + "/"
    html = (
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<base href="{base}"><style>{CSS}</style></head><body>{body}</body></html>'
    )
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "rapport.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                f"--print-to-pdf={pdf_path.resolve()}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            timeout=120,
        )
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport Markdown -> PDF")
    parser.add_argument("--md", default="docs/RAPPORT.md")
    parser.add_argument("--pdf", default="docs/RAPPORT.pdf")
    args = parser.parse_args()
    out = build_pdf(Path(args.md), Path(args.pdf))
    print(f"[PDF] {out}")
