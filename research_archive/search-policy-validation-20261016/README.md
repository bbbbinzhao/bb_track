# Search Policy Validation Snapshot (2026-10-16)

This branch preserves the search-region diagnostic and policy-learning experiments
before returning to the original FocusTrack implementation for other research.

## Versions

- `baseline-before-search-validation`: original source from the supplied
  `FocusTrack-main.zip`, excluding four `.DS_Store` files. This is a reconstructed
  baseline, not recovered upstream Git history.
- `archive/search-policy-validation-20261016`: current experimental source,
  selected reports, plots, protocols, and sequence-level statistics.
- `search-validation-20261016`: immutable experiment snapshot tag.
- `main`: restored original source, ready for subsequent research.

The baseline restores the original tracker, dataset constructor, and analysis
script. It also restores the original `tracking.plot_utils` import in
`tracking/analysis_results.py`; that module is absent from the supplied archive.
Consequently this original analysis entry point may require repair before use.
Machine-local dataset paths, data, frozen weights, and raw results remain local.
No claim is made that importing the source recreates historical runtime results.

## Saved Evidence

`reports/` mirrors selected files under `output/search_policy/`. Start with:

- `reports/full_val_analysis/full_val_report.md`
- `reports/post_reboot_check_20261015/recheck_report.md`
- `reports/learning_pilot_seed0/analysis/research_assessment.md`
- `reports/conservative_coverage_seed0/phase_assessment.md`
- `reports/fixed8_horizon_gate/phase_assessment.md`
- `reports/horizon_diagnostic_train/phase_assessment.md`

The experimental usage documentation is `tracking/SEARCH_POLICY.md` on this
branch. Existing reports remain as originally produced, including their original
local paths. `full_inventory.json` provides per-file hashes for the local backup;
`provenance.json` records source and checkpoint identity and backup integrity.

The primary source archive SHA256 is
`53612f2a94f4b995bbdd656480b6b409bc476e53d99d828a7f2f8e889cb293c5`.
The frozen FocusTrack checkpoint SHA256 is
`c259c341eca7179ff5612a034898678337f9d10c2c8d456ca4aae184de7a0564`.

## Research Status

Full official validation included 90 sequences and 94,711 frames. The original
SRA visible-frame mean IoU was approximately 0.699356; fixed factor 8 achieved
0.714341. The current aggregate policy achieved approximately 0.703776 and did
not beat fixed factor 8. Train-side long-horizon diagnostics covered 200
sequences and 213,995 frames, with 954 candidate states and 940 paired states.
Longer oracle reward windows improved selection in some event states, but gains
were concentrated in a few sequences and did not establish a deployable policy.
The current aggregate policy also failed its complete 20-sequence internal-dev
check against fixed factor 8. GRU/SLT training and the three-seed study were not
completed. The test split was not used. Consult each report for its protocol,
scope, uncertainty, and exclusions before citing any result.

## Local Backup and Recovery

The complete local backup, including raw experiment outputs, experimental model
checkpoints, the frozen tracking checkpoint, local path settings, and reference
sources, is `output/version_archives/20261016/search-validation-full.tar.gz`.
It is intentionally excluded from Git. Data images are not duplicated. Raw
outputs also remain in their original `output/search_policy/` locations.

A self-contained Git history backup is saved locally as
`output/version_archives/20261016/focustrack-history.bundle`.

Switch to the saved experiment code with:

```bash
git switch archive/search-policy-validation-20261016
```

Return to the original code with:

```bash
git switch main
```

Commit or stash any subsequent code changes before switching branches. GitHub
contains source and selected evidence only; full replay requires the local
datasets, environment, weights, and raw-result backup.
