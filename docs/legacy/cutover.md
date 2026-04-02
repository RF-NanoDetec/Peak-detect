# Cutover to LabOne-Style UI

This release introduces a local web UI (Next.js) backed by a local FastAPI service:

- Backend service: `service/app.py` (packaged as `PeakService.exe`)
- Frontend: `ui-web/` (Next.js app). During development run `npm run dev` and open http://127.0.0.1:3000.
- Installed launcher opens `http://127.0.0.1:8765` (or the UI address you configure).

## Running locally

1) Python backend:

```bash
pip install -r requirements.txt
python -m service.app
```

2) Web UI:

```bash
cd ui-web
npm install
npm run dev
```

Then open http://localhost:3000. Health banner calls the backend on 127.0.0.1:8765.

## Packaging (Windows)

```bat
pip install pyinstaller
tools\build_service.bat
```

EXE at `dist\PeakService.exe`. Build installer with Inno Setup using `installer\inno\PeakTool.iss`.

## Migration Complete

The legacy Tkinter UI has been completely removed. The application now uses only the modern web-based interface (Next.js + FastAPI).


