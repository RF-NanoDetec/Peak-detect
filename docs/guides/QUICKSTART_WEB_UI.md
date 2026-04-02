# Quick Start (Supplemental Web UI Note)

> **Status:** supplemental quick-start note. This file is not part of the canonical documentation path.

Use this file for a short operational reminder only. For the authoritative workflow and current product boundaries, use:
- `README.md`
- `docs/README.md`
- `docs/USER_MANUAL.md`
- `docs/user_interface.md`

## Quick run reminder

### Start from this checkout
```powershell
.\run_app.ps1
```

or:

```bat
run_app.bat
```

### Development mode
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m service.app
```

```powershell
cd ui-web
npm install
npm run dev
```

## Current UI flow
1. Load Data
2. Process & Detect
3. Analyze Results
4. Double Peak

Export remains a separate dedicated page/action.

## Scope note
This quick-start note is secondary reference material. It should not be used to infer unsupported formats, public support processes, or legacy desktop workflows.
