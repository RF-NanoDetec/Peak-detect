# Documentation Index

This index defines the **canonical documentation path** for the current private/internal-use phase of the Peak Analysis Tool repository.

## Canonical documents

These are the only primary documents a new reader should need for the current product story:

1. [`../README.md`](../README.md) - repository front door, product scope, posture, and run/install overview
2. [`USER_MANUAL.md`](USER_MANUAL.md) - end-to-end operator workflow
3. [`user_interface.md`](user_interface.md) - route-by-route UI guide with screenshots
4. [`mathematical_reference.md`](mathematical_reference.md) - scientific and algorithmic basis

## Documentation classes

### Canonical active docs
- `README.md`
- `docs/README.md`
- `docs/USER_MANUAL.md`
- `docs/user_interface.md`
- `docs/mathematical_reference.md`

### Supplemental guides
`docs/guides/` contains scoped supporting material such as quick-start notes, protocol/correction details, and packaging/distribution notes. These guides are **secondary** and should not override the canonical documents above.

### Engineering reports
`docs/reports/` contains implementation logs, fix summaries, migration notes, and performance write-ups. These are historical engineering references, not primary user documentation.

### Legacy material
`docs/legacy/` contains superseded or historical notes preserved for context.

### Assets and references
- `docs/figures/` - screenshots and figure assets
- `docs/references/` - bibliography/support files used by documentation tooling

## Reader path

Recommended reading order for the current product:
1. `README.md`
2. `docs/USER_MANUAL.md`
3. `docs/user_interface.md`
4. `docs/mathematical_reference.md`

Use supplemental guides only when you need deeper detail on a narrow topic.

## Phase 1 demotions in place

The following documents remain in the repo but are not canonical:
- `docs/guides/DISTRIBUTION_README.md`
- `docs/guides/PHOTON_CORRECTION_AND_PROTOCOL.md`
- `docs/guides/QUICKSTART_WEB_UI.md`
- `docs/guides/user_manual.md`
- `docs/guides/user_manual.pdf`

Reports under `docs/reports/` are intentionally excluded from the primary path.
