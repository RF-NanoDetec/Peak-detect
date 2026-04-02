# Documentation Overview

This folder now groups the authoring material for the Peak Analysis Tool by intent so it is easier to find the right reference without paging through dozens of files.

## Guides (`docs/guides/`)
- Step-by-step walkthroughs, distribution instructions, and feature user guides (cutovers, photon correction, protocol usage, etc.).

## Reports (`docs/reports/`)
- Engineering logs, fix summaries, optimization reports, and other status write-ups that accompany major backend or UI changes.

## Legacy Notes (`docs/legacy/`)
- Historical records from the retired desktop/Tkinter implementation and early migration spikes. Keep for context only.

## References (`docs/references/`)
- Citation metadata (`references.bib`, CSL styles, LaTeX headers) consumed by the document tooling in `scripts/` plus image assets under `docs/figures/`.

## Figures (`docs/figures/`)
- PNG exports that illustrate user documentation. Linked from the guides and README.

When adding new documentation, drop it into the folder that matches its purpose so the root of the repo stays lean. If a document supersedes an older write-up, move the previous file into `docs/legacy/` instead of deleting it outright.
