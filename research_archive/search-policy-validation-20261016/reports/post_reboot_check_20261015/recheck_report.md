# Post-reboot reproducibility check (2026-10-15)

## Scope and environment

The earlier nvidia-smi/gpustat error reported a driver/NVML version mismatch.
This does not itself establish that CUDA numerical results were corrupted.
The rebooted system reports driver and NVIDIA-SMI version 595.99.02 on all
four RTX 4090 GPUs. PyTorch 2.0.0 (CUDA build 11.8) detects all four GPUs.

Historical experiment files were preserved. New results are in this directory.
No tracking code, checkpoint, dataset or experiment protocol was changed.

## Accuracy replay

One sequence from each original validation shard was replayed on its GPU.
These are shard-start samples, not a random or exhaustive hardware audit.
Each intervention run first reproduces the full baseline trajectory against
baseline_val_v2_full.json with atol=1e-4 and rtol=0. All four passed.

| GPU | Sequence | Frames | Intervention points | Candidate records |
|---|---|---:|---:|---:|
| 0 | 03_7951_1500-2999 | 1500 | 33 | 297 |
| 1 | 20190925_200320_1_8 | 39 | 32 | 288 |
| 2 | 20190926_141816_1_9 | 1000 | 34 | 306 |
| 3 | 20190926_200510_1_7 | 1000 | 32 | 288 |
| Total | | 3539 | 131 | 1179 |

Compared by sequence and frame against intervention_val_v2_full.json:

- All candidate records compare exactly equal as parsed JSON values,
  including next-frame IoU, future IoU, crop geometry and coverage.
- Baseline future IoU values compare exactly equal at every sampled point.
- There are no added or missing intervention points in these four sequences.
- The intervention protocol is search_intervention_v2_hann_integer_geometry.

## Runtime and protocol checks

All seven tracking.test_search_policy_protocol tests passed again.
runtime_verification.json is exactly equal to its pre-reboot counterpart:
frozen parameters/buffers hash, A/B/A branch isolation, confidence-dependent
Hann updates, crop override ownership, coordinate mapping and motion reset.

The isolated 3-repeat, 200-frame SRA/motion latency benchmark was rerun.
See isolated_runtime.json in this directory for the new measurements.
Accuracy agreement does not establish speed agreement across driver states.

| Measurement | Before reboot | After reboot |
|---|---:|---:|
| SRA median FPS | 40.76 | 39.23 |
| Motion median FPS | 39.29 | 38.17 |
| Motion relative FPS change | -3.62% | -2.70% |

These short-run timings vary between repetitions and do not identify the
driver mismatch as the cause of any speed difference. Use matched current
environment measurements for future method comparisons.

Kernel logs were not accessible to this account; no conclusion about the
absence of historical NVIDIA Xid/hardware errors is drawn from those logs.

## Decision

The sampled replays provide no evidence that the NVML mismatch affected the
reported tracking accuracy or candidate intervention results. Preserve the
full-validation findings for the next feasibility phase. This is a sampled
reproducibility check, not a repeat of all 90 validation sequences or of the
complete motion/coordinate-control evaluations. Full reruns remain warranted
if numerical divergence, CUDA failures or incomplete outputs are discovered,
and should be included when producing the final paper's experimental tables.
