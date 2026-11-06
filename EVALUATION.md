# Evaluation: My harness for BID — adaptive candidate budget

One reported stochastic Push-T condition (noise 1.0), fresh seeds 20100–20219. Not the paper's seven-task aggregate.

## Budget

300 physical actions, 1.0x; no reset or retry allowance.

## Interpretation

- Primary overlap difference interval is -0.0120 to +0.0710; it includes zero.

- Earlier 100-case cohort: overlap 0.88995 versus 0.87863 and successes 21 versus 18; do not pool or hide it.

- Candidate trajectories fell 1,018,080 to 521,442; measured amortized inference time fell 24.2% on shared hardware.

- Modern runtime, batching and reconstructed RNG differ from the historical paper setup.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
