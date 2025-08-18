### UI Modernization To‑Do

Status owner: UI/UX pass for Peak Analysis Tool

#### Completed
- [x] Add spacing and density tokens to `ui/theme.py` and integrate into ttk styles (buttons, entries, tabs, treeview, progressbar)
- [x] Introduce `ThemeManager.set_density()` and wire tokens inside style application
- [x] Replace one hard-coded color in `ui/components.py` (`preview_label`) with `ThemeManager.get_color('text')`
- [x] Run tests locally to confirm no regressions (observed: 6 passed, 1 warning)

#### Step 0 — Repo hygiene
- [x] Create branch `feature/ui-modernization` and work behind PR

#### Step 1 — Design tokens and matplotlib
- [x] Add tokens for radii, shadows, focus ring; helpers `get_spacing`, `get_radius`, `get_outline`
- [x] Add `apply_matplotlib_theme()` in `ui/theme.py` and call it from `main.py`

#### Step 2 — ttk style polish (modern look)
- [ ] Buttons: visible focus ring; clearer hover/active states
- [ ] Notebook tabs: underline/indicator for active; better selected contrast
- [ ] Scrollbar: slimmer track/handle with hover state
- [ ] Progressbars: semantic variants (Primary/Success/Warning/Error) and consistent thickness

#### Step 3 — Tokenize remaining literals
- [ ] Replace all hard-coded colors/rcParams in UI with `ThemeManager` tokens

#### Step 4 — Top toolbar
- [ ] Implement `Toolbar` component (title, theme toggle, density switch, Run All, Open Recent)
- [ ] Mount toolbar in `Application.__init__` above main grid

#### Step 5 — Componentize control panel
- [ ] Create `Section`, `Card`, `FormRow`, `LabeledField`
- [ ] Refactor tabs to use sections, add collapsible "Advanced" areas

#### Step 6 — Welcome screen refresh
- [ ] Update typography and actions; add "Open Recent" list; ensure theme awareness

#### Step 7 — Feedback & motion
- [ ] Toast notifications helper (auto-dismiss)
- [ ] Busy overlay during long `@ui_action` operations

#### Step 8 — Preferences dialog
- [ ] Theme (light/dark), Accent selection, Density (comfortable/compact), Start on Welcome, Recent limit
- [ ] Persist via `load_user_preferences`/`save_user_preferences`

#### Step 9 — Matplotlib consistency
- [ ] Remove ad-hoc styling in `plotting/*`; rely on `ThemeManager` rcParams and helpers

#### Step 10 — Accessibility & keyboard
- [ ] Ensure focus visibility and tab order; AA contrast checks
- [ ] Surface shortcuts in tooltips and welcome screen

#### Step 11 — Cleanups & docs
- [ ] Remove dead styles, update screenshots in `docs/` and `README.md`
- [ ] Final QA on Windows scaling 100–150%

Notes:
- Files updated so far: `ui/theme.py`, `ui/components.py`

