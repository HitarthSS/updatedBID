# My harness for BID — adaptive candidate budget

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2408.17355). The evaluated behavior is in [`my_harness.py`](my_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | Official repository representative mean maximum overlap: 0.889; not 88.9% complete success |

| Our run of the authors' harness | Overlap 0.8566; 22/120 complete successes (18.33%) |

| **My harness** | **Overlap 0.8848; 28/120 complete successes (23.33%)** |


One reported stochastic Push-T condition (noise 1.0), fresh seeds 20100–20219. Not the paper's seven-task aggregate.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Choose six or fifteen candidates per frozen strong/weak policy based on preceding plan confidence and current public keypoint motion.

2. Pack variable candidate counts into one sampling batch per policy.

3. Keep the original BID ranking, 100 DDPM steps, policy weights and one executed action per decision.


Categories: `uncertainty_compute`, `proposal_selection`.

## Models and inputs

- Authors' EMA strong checkpoint epoch=0150-test_mean_score=0.909.ckpt and weak checkpoint epoch=0050-test_mean_score=0.250.ckpt.

- Native keypoint diffusion policies, not a conversational VLM or a VLA.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/YuejiangLIU/bid_diffusion) before installing its model stack.
2. Controller dependency: Use the pinned upstream diffusion-policy environment.
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
PYTHONPATH=upstream python -m compileall -q my_harness.py
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

In the pinned diffusion-policy environment, instantiate AdaptiveBID once per episode group and call step(strong, weak, obs, prior, active). The original BID sampler performs selection. reference/run.py records the evaluated grouped loop.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

300 physical actions, 1.0x; no reset or retry allowance.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Primary overlap difference interval is -0.0120 to +0.0710; it includes zero.

- Earlier 100-case cohort: overlap 0.88995 versus 0.87863 and successes 21 versus 18; do not pool or hide it.

- Candidate trajectories fell 1,018,080 to 521,442; measured amortized inference time fell 24.2% on shared hardware.

- Modern runtime, batching and reconstructed RNG differ from the historical paper setup.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

Source syntax and recorded-core hashes checked. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
