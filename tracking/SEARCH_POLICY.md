# Frozen FocusTrack Search Policy Experiments

`search_policy_experiment.py` evaluates a small controller while keeping the
FocusTrack checkpoint frozen. The default checkpoint is
`output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar`.

Run commands from the repository root with the `focustrack` environment:

```bash
PYTHONPATH=. python tracking/search_policy_experiment.py diagnose \
  --split val --output output/search_policy/baseline_val

PYTHONPATH=. python tracking/search_policy_experiment.py intervene \
  --split val --output output/search_policy/intervention_val

PYTHONPATH=. python tracking/search_policy_experiment.py cache-teacher \
  --output output/search_policy --max-teacher-samples 10000 --seed 0

PYTHONPATH=. python tracking/search_policy_experiment.py train-supervised \
  --output output/search_policy --epochs 10 --policy-type gru --seed 0

PYTHONPATH=. python tracking/search_policy_experiment.py evaluate \
  --split val --policy-checkpoint \
  output/search_policy/policy_supervised_seed0_ep010.pth \
  --output output/search_policy/supervised_val

PYTHONPATH=. python tracking/search_policy_experiment.py train-slt \
  --output output/search_policy --policy-checkpoint \
  output/search_policy/policy_supervised_seed0_ep010.pth \
  --epochs 10 --episodes-per-epoch 1024 --seed 0

PYTHONPATH=. python tracking/search_policy_experiment.py evaluate \
  --split val --policy-checkpoint \
  output/search_policy/policy_slt_seed0_ep010.pth \
  --output output/search_policy/slt_val

PYTHONPATH=. python tracking/search_policy_experiment.py compare \
  --baseline-report output/search_policy/baseline_val/diagnose_val.json \
  --candidate-report output/search_policy/slt_val/evaluate_val.json \
  --output output/search_policy/comparison
```

Use `--max-sequences 1` for a smoke run. The intervention report contains the
one-step and eight-frame counterfactual IoU for all nine actions. The policy
controller uses the same nine actions: three causal motion extrapolations and
search factors 4, 6, and 8. `--action-mode center` and `--action-mode scale`
provide the two action ablations.

The tracker requires CUDA. The policy controller receives a 32-dimensional
observation built from a 4x4 pooled score map, peak statistics, presence
confidence, predicted box geometry, causal motion, scale change, crop factor,
and crop center. An external region owns the crop center and search factor;
Hann updates still follow the original confidence thresholds. The diagnostic
`sra_next_factor` supplies the counterfactual next SRA scale when reverting to
SRA after a single intervention.

## Full Validation Protocol

Use `--sequence-start` (inclusive) and `--sequence-end` (exclusive) to shard the
sorted official sequence list. Bind a process with `CUDA_VISIBLE_DEVICES=N`
and use `--device cuda:0` inside that process. `--cpu-threads 1` avoids CPU
oversubscription. Use separate output directories per worker.

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/search_policy_experiment.py intervene \
  --split val --sequence-start 0 --sequence-end 23 \
  --baseline-report output/search_policy/baseline_val_full.json \
  --output output/search_policy/intervention_v2_gpu0 --resume

python tracking/search_policy_experiment.py merge \
  --input-reports output/search_policy/intervention_v2_gpu0/intervention_val.json \
  output/search_policy/intervention_v2_gpu1/intervention_val.json \
  output/search_policy/intervention_v2_gpu2/intervention_val.json \
  output/search_policy/intervention_v2_gpu3/intervention_val.json \
  --merged-name intervention_val_v2_full

python tracking/analyze_search_policy.py \
  --intervention output/search_policy/intervention_val_v2_full.json

python -m unittest tracking.test_search_policy_protocol -v
python tracking/verify_search_runtime.py
```

Interventions write atomic progress after each completed sequence and support
`--resume`. A supplied baseline report must match the checkpoint hash, and its
predictions are reproduced before each sequence's interventions. Candidates
affect one crop, followed by original SRA; the changed frame also updates Hann
and the subsequent SRA factor. `--expanded-candidates` adds the diagnostic
3x3 center grid at offsets of +/-25% crop side, for 27 candidates.

Only reports with protocol `search_intervention_v2_hann_integer_geometry`
are valid for the current gate. Earlier `intervention_gpu*` outputs skipped
Hann updates on override frames and must not support research conclusions.
All policy checkpoints from earlier smoke runs validate execution only.

`--motion-baseline` evaluates causal constant-velocity centers with original
SRA scales. `--save-raw-responses` saves unwindowed score maps and observations
as per-sequence NPZ files. FPS from concurrent workers measures throughput
under that load, not an isolated runtime benchmark.

`analyze_search_policy.py` checks exact full-val sequence coverage and writes
audited metrics, per-sequence CSV, and recovery plots. Visible IoU/AUC exclude
initialization and invalid GT. The official Anti-UAV mixed score includes
initialization and scores nonempty predictions on invisible frames as zero.
Recovery is confirmed at the third consecutive IoU>=0.5 frame, within 30
frames after the first legal visible GT following disappearance. A new
disappearance ends the previous recovery attempt. Both sequence averages
and event-weighted recovery rates are retained. The 21-point visible AUC
uses `>=` at thresholds from 0 to 1; keep this definition consistent when
comparing experiments.

The training commands above remain experimental entry points. Full policy
training is conditional on the complete diagnostic gate; smoke checkpoints
do not establish gains from temporal modeling or SLT.

The completed full-val report is
`output/search_policy/full_val_analysis/full_val_report.md`. It includes the
original SRA, causal motion, and coordinate-only control on all 90 sequences,
paired bootstrap intervals, intervention evidence, and an isolated latency
check. `closed_loop_deltas_and_recovery.png` compares per-sequence deltas and
GT-defined recovery curves. Training and SLT results are not included yet.

## Continuous Supervised Feasibility Pilot

The legacy `cache-teacher`/`train-supervised` commands above are smoke entry
points. Use the continuous-cache pipeline for actual policy learning:

```bash
python tracking/search_policy_learning.py cache \
  --data-root data/antiuav410 --sequence-start 0 --sequence-end 25 \
  --output output/search_policy/learning_pilot_seed0/teacher_000_025 --resume

python tracking/search_policy_learning.py train --device cpu \
  --policy-type mlp --action-mode scale --epochs 1 \
  --episodes-per-epoch 1024 --batch-size 4 --seed 0 \
  --output output/search_policy/learning_pilot_seed0/mlp_scale/train \
  --input-manifests <all-complete-training-shard-manifests>

python -m unittest tracking.test_search_policy_learning tracking.test_search_policy_protocol -v
```

The pilot uses eight uniformly placed consecutive 12-observation windows per
training sequence, with four warmup observations and eight supervised steps.
Observation t selects the crop at t+1. All observations come from causal
baseline tracking. Invisible/illegal next-frame boxes mask localization
teachers without removing temporal steps. Labels contain all nine one-step
candidate IoUs; action restrictions are applied before teacher softmax.

Caches use global sequence names and validate train-only split, annotation
hash, checkpoint hash, contiguous frame IDs and exact coverage of all 200
training sequences. Ten percent of training sequences are held out by a
fixed name hash for internal development; official val labels never train
the policy. Development utility describes baseline states, not closed-loop
tracking. Resume checkpoints contain optimizer and Python/NumPy/PyTorch/
sampler RNG states; bitwise CPU resume equivalence is tested.

`SearchPolicyActor` and `SearchPolicyTrainer` are independent adapters;
the original FocusTrackActor and LTRTrainer are unchanged. The MLP uses two
64-unit hidden layers and the GRU uses 64 units. Legacy smoke MLP checkpoints
with three hidden layers are still readable. SLT remains deferred until
learning and temporal controls justify its cost; the old SLT smoke entry
point is not a validated training protocol.

`--fixed-factor 4`, `6`, or `8` on the evaluation command gives constant
scale controls with zero center extrapolation and actual integer crop
coordinates. Like learned policies, they use the baseline first tracked
frame and change the crop from the following frame onward. Hann confidence
rules are unchanged; SRA does not overwrite controller crops.

`run_search_learning_pilot.py` starts or resumes the fixed-factor controls,
teacher collection, policy training and validation. Use `--attach-existing`
only when the three fixed controls and first teacher shard are already running.
It trains seed-zero scale/joint MLPs for one screening epoch, evaluates each
on all 90 val sequences using four GPUs, and writes audited comparisons,
per-sequence CSV, action counts, recovery curves and a report under
`output/search_policy/learning_pilot_seed0/analysis/`. This screening budget
is explicitly smaller than the final ten-epoch, three-seed protocol.
The scheduler reads `teacher_schedule.json` when present. Refining unfinished
shards reuses completed per-sequence files; retain the original teacher
directories referenced by the new manifests.

After all GPU workers finish, benchmark the learned checkpoints with:

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/benchmark_search_policy.py \
  --policy-checkpoints <scale-checkpoint> <joint-checkpoint> \
  --output output/search_policy/learning_pilot_seed0/isolated_runtime.json
python tracking/analyze_search_learning.py --root output/search_policy/learning_pilot_seed0
```

The convergence extension preserves epoch-one screening results. Resume each
mode's epoch-one checkpoint with `search_policy_learning.py train --epochs 10
--resume-checkpoint <epoch-one-checkpoint> --output <model>/continued_train`
and the same seed, batch size, episode count and complete manifest list.
`select_search_supervised.py --root <pilot-root>` selects by train-internal
development teacher IoU, then `validate_search_selection.py --root <pilot-root>`
evaluates the selected policies on complete val. It reads the initial
validation allocation logs and fills free GPU slots up to two workers per
GPU. The pilot evaluates epoch one and the selected checkpoint; it does not
claim the final protocol's per-epoch complete validation or three-seed checks.

The completed seed-zero convergence pilot and all 90-sequence validation
results are in `output/search_policy/learning_pilot_seed0/analysis/`.
`learning_pilot_report.md` contains audited scores, paired intervals, crop
statistics, recovery and isolated runtime; `research_assessment.md` records
the research interpretation and proposed next diagnostic. Fixed factor 8
outperformed the current learned policies. These results do not validate
GRU or SLT; the next proposed gate is opportunity relative to fixed 8 with
multi-frame teachers, before expanding the training matrix.

## Opportunity Beyond Fixed Eight

`search_fixed8_intervention.py` starts from fixed-eight states and changes
only one crop. All remaining frames in the eight-frame horizon follow
fixed eight, with the original Hann rules. A baseline-action replay is
checked at every point; A/B/A equality is checked on each sequence. It saves
causal observations and candidate trajectories separately from GT labels.

```bash
python -m unittest tracking.test_fixed8_intervention -v
python tracking/search_fixed8_intervention.py run \
  --reference output/search_policy/learning_pilot_seed0/fixed_8/evaluate_val.json \
  --output output/search_policy/fixed8_horizon_gate/val \
  --gpus 0 1 2 3 --workers-per-gpu 2 --shard-size 6
python tracking/analyze_fixed8_intervention.py \
  --root output/search_policy/fixed8_horizon_gate/val
```

Completed sequence files are resumable via shard manifests. The analyzer
requires the exact full split and matching identities. It separates uniform,
failure and reappearance groups, excludes shortened horizons from the primary
groups, bootstraps over sequences and exports per-point/per-action CSVs.
GT-selected one-step and eight-step choices are optimistic diagnostics,
not learned controllers. The diagnostic contains no test entry point.

Once the fixed-eight opportunity gate passes, the matched teacher screening
uses eight uniform full-horizon states per training sequence. The two scale
MLPs use the same observations, initialization, sampled states and updates;
only one-frame versus eight-frame teacher utility differs. Both train on the
same states with a legal visible next-frame label. This single-state screening
does not train a recurrent controller.

```bash
python tracking/search_fixed8_intervention.py run --split train \
  --output output/search_policy/fixed8_horizon_gate/train \
  --uniform-points 8 --event-limit 0 --horizon 8 --shard-size 5
python tracking/train_fixed8_teacher.py \
  --teachers output/search_policy/fixed8_horizon_gate/train \
  --output output/search_policy/fixed8_horizon_gate/learning_scale_seed0
python tracking/validate_fixed8_teacher.py \
  --root output/search_policy/fixed8_horizon_gate/learning_scale_seed0
python tracking/analyze_fixed8_learning.py \
  --root output/search_policy/fixed8_horizon_gate/learning_scale_seed0 \
  --teachers output/search_policy/fixed8_horizon_gate/train
```

Teacher selection is sequence-disjoint within train; official val is used
only for the resulting complete closed-loop checks. Teacher caches validate
exact coverage of all 200 train sequences. In the matched screening, 8,192
states per epoch with batch 32 matches 256 updates per epoch for both models.
Checkpoints record optimizer and random states; `train_fixed8_teacher.py`
currently starts a fresh run and has no resume CLI. Do not reuse an output
directory to change its training settings. The validation scheduler resumes
per-sequence GPU evaluation and exports both selected model reports.

An additional control separates reward averaging from teacher strength. The
eight-frame temperature is chosen by bisection to match the training states'
mean single-step teacher entropy at temperature 0.1. Calibration never reads
development or val labels. The model uses the same data, seed and budget.

```bash
python tracking/train_fixed8_teacher.py \
  --teachers output/search_policy/fixed8_horizon_gate/train \
  --output output/search_policy/fixed8_horizon_gate/learning_entropy_seed0 \
  --objectives eight --match-one-step-entropy
python tracking/validate_fixed8_teacher.py \
  --root output/search_policy/fixed8_horizon_gate/learning_entropy_seed0 \
  --objectives eight
python tracking/analyze_fixed8_learning.py \
  --root output/search_policy/fixed8_horizon_gate/learning_scale_seed0 \
  --teachers output/search_policy/fixed8_horizon_gate/train \
  --entropy-root output/search_policy/fixed8_horizon_gate/learning_entropy_seed0
```

`validate_fixed8_teacher.py --wait-for-root <matched-root>` can share free GPU
slots with the first scheduler once its entire queue has been dispatched.
Benchmarking multiple models of the same type/action mode requires distinct
`benchmark_search_policy.py --policy-labels one eight eight_entropy` labels.
Only run isolated timing once all validation GPU workers have exited.

## Conservative Coverage Screening

The bounded next experiment is registered in
`output/search_policy/conservative_coverage_seed0/protocol.json`. It keeps
factor eight as the default and regresses the eight-frame advantages of
factors four and six. A common helper in `lib/models/search_policy.py`
applies the checkpoint's optional `decision` metadata; older classification
checkpoints retain their original action rule. Regression gates deliberately
reject stochastic sampling and are not SLT checkpoints.

```bash
python -m unittest tracking.test_conservative_search
python tracking/run_conservative_search.py \
  --root output/search_policy/conservative_coverage_seed0 \
  --gpus 0 1 2 3 --workers-per-gpu 2
```

`search_conservative_coverage.py` has no validation/test collection entry
point. It runs all 200 train sequences, retaining eight uniform states and
at most eight states around failure/reappearance events. GT selects training
sample locations and labels only. At each initial state it replays eight
consecutive frames at a cyclically assigned factor 4/6/8, evaluates three
scale actions, and follows fixed eight for the remaining seven frames.
All branches restore full tracker state and pass exact A/B/A checks.

Uniform and event-balanced arms use the same cache, seed and 2,560 updates.
Only uniform states from the existing 20 internal-dev sequences select epochs
and margins; ties prefer fewer interventions and earlier epochs, including
the option to disable intervention. The current event policy then generates
complete train trajectories and candidate continuations. An aggregation arm
mixes old and on-policy event-balanced pools equally; the matched control
continues on the old pool with the same initialization and 2,560 additional
updates. Both retain their parent checkpoint as a selection candidate.
All arms are selected using the same initial-cache uniform dev states;
on-policy dev teacher results are additional diagnostics.

The coordinator fills four-GPU slots with on-policy collection and initial
validation, then evaluates all four selected arms on complete val. It resumes
completed collection/evaluation shards but does not resume partial training.
`--wait-for-coverage` is only for joining an already active first collection;
otherwise the coordinator starts/resumes that collection itself. Do not run
two coordinators on the same experiment directory.

`analyze_conservative_search.py --root <experiment-root>` audits complete
splits and checkpoint identities and exports `analysis/phase_report.md`,
paired sequence CSVs, intervals, recovery curves and training support audits.
Local mixed-crop warmup is not full long-term scale coverage, one-crop teacher
labels still differ from repeated deployment, and this seed-zero screening
does not establish temporal or SLT gains. Frozen-eight remains the mandatory
strong baseline. Test remains untouched.

On multi-socket hosts, use `benchmark_search_policy.py --cpu-core <core>`
with a CPU local to the selected GPU when checking runtime variance. The
controlled repeat here used physical GPU 0 and CPU 2 (NUMA node 0), retained
the initial unpinned measurement, and recorded affinity and sequence name.
Timing covers one sequence's first 200 frames, not the full validation split.

## Nested reward-horizon diagnosis

The next train-only diagnostic does not train a new controller:

```bash
python tracking/search_horizon_diagnostic.py run \
  --output output/search_policy/horizon_diagnostic_train \
  --gpus 0 1 2 3 --workers-per-gpu 2
```

It registers the protocol before collection, covers all 200 train sequences,
and keeps the existing 180/20 sequence split. Each sequence contributes four
interior uniform states and up to two failure/reappearance states, all on the
full fixed-eight trajectory with 64 future frames available. From the same
snapshot, the next crop uses factor 4/6/8; the remaining 63 frames use either
fixed eight or the selected aggregate controller. The common trajectories
yield nested visible-frame rewards at 8/32/64 frames. GT selects diagnostic
states and measures rewards, never causal controller inputs. Every point has
exact 64-frame A/B/A checks; the factor-eight/fixed-eight branch must replay
the original trajectory exactly. Frozen parameters and buffers are hashed.

The coordinator resumes completed shards. The analyzer requires complete
sequence coverage and matching identities; it exports paired point and
sequence CSVs, bootstrap intervals, nested reward curves and sign-reversal
scatter plots under `analysis/`. `pipeline_complete.json` is written only
after collection and analysis both succeed. Re-run analysis using
`python tracking/analyze_search_horizon.py --output <experiment-root>`.
Changing the oracle action with horizon alone is not proof of harm: report
its signed long-horizon effect versus the matched factor-eight action,
and distinguish meaningful reversals (gain8 >= .01 and gain64 <= -.01).
The fixed-eight state distribution and one-crop intervention do not bound
continuous learned-policy performance. Official val/test are not used here.

After that pipeline completes, reproduce the unchanged aggregate model on
the full 20-sequence internal development split, with four GPUs:

```bash
python tracking/search_horizon_diagnostic.py run-dev \
  --output output/search_policy/horizon_diagnostic_train \
  --gpus 0 1 2 3 --workers-per-gpu 1
```

This does not select a new epoch or margin. It compares complete policy
trajectories against the just-repeated fixed-eight baselines, saves raw
predictions under `internal_dev/`, and exports `analysis/internal_dev_report.md`,
`internal_dev_results.json` and per-sequence paired differences. The internal
development split was previously used for model selection; it is not an
independent test set. Its completion marker is separate from the horizon
pipeline marker.

`supplement_search_horizon.py --output <experiment-root>` regenerates
`sensitivity.json`, `illustrative_cases.csv` and candidate-prefix plots from
audited raw results. Contributions are measured after averaging states within
each sequence. Selected extreme cases illustrate mechanisms only; their
selection is recorded and must not substitute for the full-state statistics.
