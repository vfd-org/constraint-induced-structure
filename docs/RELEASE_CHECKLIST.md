# CISE v1.0.0 Release Checklist

Use this checklist before tagging and publishing the release.

---

## Pre-Release Verification

### Installation
- [ ] Clean install works: `pip install -e .`
- [ ] All dependencies install correctly
- [ ] Python 3.8+ compatibility verified
- [ ] No GPU or internet required

### Tests
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Smoke tests cover samplers, constraints, reweighting, runner
- [ ] Test coverage adequate for core functionality

### Release Run
- [ ] Release run completes: `python scripts/run_release.py`
- [ ] Completes in <2 minutes on CPU
- [ ] Produces expected output structure:
  - [ ] `outputs/release_run/figures_release/` (6-9 figures)
  - [ ] `outputs/release_run/figures_control/` (2+ control figures)
  - [ ] `outputs/release_run/metrics_release.json`
  - [ ] `outputs/release_run/summary_release.md`

### Figures
- [ ] ESS vs beta figure present and readable
- [ ] Energy histogram overlay shows clear shift
- [ ] Norm distribution shows clear shift
- [ ] PCA variance shows dimensional concentration
- [ ] PCA scatter shows visible difference
- [ ] SV spectrum (matrices) shows rank suppression
- [ ] Hierarchy curve shows steeper decay
- [ ] Control figures show persistent structure difference

### Control Analysis
- [ ] Norm-matched baseline control implemented
- [ ] Control metrics show non-zero deltas
- [ ] Control figures visually demonstrate the point
- [ ] README explains the control clearly

---

## Documentation

### README.md
- [ ] Title and one-sentence description
- [ ] "What it is" section
- [ ] "What it is NOT" bullet list
- [ ] Quickstart with install + run commands
- [ ] Figure embeds render (or link paths work)
- [ ] Key observations table
- [ ] Anti-dismissal control section
- [ ] Reproducibility section
- [ ] "How to extend" with 3 steps
- [ ] Citation section with BibTeX
- [ ] License section
- [ ] Disclaimer clearly visible
- [x] Repository URL set correctly

### Other Documentation
- [ ] `CITATION.cff` present and valid
- [ ] `LICENSE` (MIT) present
- [ ] `docs/CISE_OnePage_Note/cise_note.tex` compiles
- [ ] `docs/LAUNCH_KIT.md` has thread copy and video script
- [ ] `docs/RELEASE_CHECKLIST.md` (this file) present

### Configs
- [ ] `configs/default.yaml` works for full run
- [ ] `configs/release.yaml` works for curated run
- [ ] Configs are well-commented

---

## Code Quality

- [ ] No hardcoded paths
- [ ] Seed is configurable and deterministic
- [ ] No debug prints or commented-out code
- [ ] Docstrings on public functions
- [ ] Type hints on function signatures
- [ ] No physics claims in code comments
- [ ] Neutral language throughout ("constraint experiment", "distributional shifts")

---

## Repository

- [ ] `.gitignore` excludes outputs, __pycache__, etc.
- [ ] No large binary files committed
- [ ] No secrets or credentials
- [ ] `pyproject.toml` version is 1.0.0
- [ ] `cise/__init__.py` version is 1.0.0

---

## Final Steps

1. [x] Repository URL set to https://github.com/vfd-org/constraint-induced-structure
2. [ ] Run full test suite one more time
3. [ ] Run release script and verify outputs
4. [ ] Commit all changes
5. [ ] Create git tag: `git tag -a v1.0.0 -m "CISE v1.0.0"`
6. [ ] Push tag: `git push origin v1.0.0`
7. [ ] Create GitHub release with:
   - Tag: v1.0.0
   - Title: CISE v1.0.0 - Constraint-Induced Structure Explorer
   - Description: Copy from README intro + key findings
   - Attach: One-page note PDF (if compiled)

---

## Post-Release

- [ ] Verify release page looks correct
- [ ] Test `pip install` from GitHub (if applicable)
- [ ] Post announcement (use LAUNCH_KIT.md)
- [ ] Monitor for issues/questions
- [ ] Respond to feedback constructively

---

## Notes

- **Tone**: Keep all communication neutral and scientific
- **Claims**: Never claim this explains physics; it's a constraint experiment
- **Openness**: Welcome counterexamples and critiques
- **Reproducibility**: Emphasize deterministic, fast, CPU-only execution
