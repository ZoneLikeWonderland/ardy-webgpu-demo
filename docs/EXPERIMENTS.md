# Selected experiment records

This directory intentionally contains selected summary JSON files rather than
raw datasets, full rollouts, checkpoints, optimizer states, or training logs.
Paths that identified the original workstation have been replaced with public
placeholders; numerical measurements are unchanged.

The summaries cover:

- compact codec and flow checkpoint sweeps;
- NFE and EMA comparisons;
- stage-one joint text/control distillation;
- DMD2/LADD generator learning-rate and iteration sweeps;
- adversarial weight and adversarial/critic timestep matching;
- guidance/critic update ratios;
- critic feature-tap, depth, and head-architecture experiments.

The durable interpretation is in `PROJECT_FINDINGS.md`. Individual scalar
rankings must not be read as a universal quality ordering: several sweeps expose
clear Pareto trade-offs between waypoint following, root drift, FK error, seam
quality, jerk, high-frequency energy, and teacher-paired fidelity. Visual
infinite-rollout testing remains necessary.

