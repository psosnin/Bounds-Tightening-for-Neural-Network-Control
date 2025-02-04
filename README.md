# Scaling Mixed-Integer Programming for Certification of Neural Network Controllers Using Bounds Tightening

This repository contains code to replicate the results of the paper "Scaling Mixed-Integer Programming for Certification of Neural Network Controllers Using Bounds Tightening".

## Usage

The repository is split into three main parts:

1. `src` contains the implementation of the bounds tightening algorithms and applications to MPC.
2. `tests` contains unit tests for the bounds tightening algorithms.
3. `scripts` contains scripts to run experiments on both the worst case error and reachability analysis problems.

Please note that the experimental results are managed using [Weights and Biases](https://wandb.ai/).

1. Install the packages with `pip install -r requirements.txt`.
2. Log in to Weights and Biases with `wandb login`.
3. Run the example scripts in `./scripts`.

To run the worst-case error analysis for the MPC problem, use the following commands:

```bash
python generate_mpc_data.py -s ${SEED}
python train_neural_network.py -s ${SEED} -n 0
python single_step_bounds.py -s ${SEED} -n 0
python worst_case_error.py -s ${SEED} -n 0
```

To run the multi-timestep reachabability analysis, use the following:

```bash
python generate_mpc_data.py -s ${SEED}
python train_neural_network.py -s ${SEED} -n 0
python multi_step_bounds.py -s ${SEED} -n 0
python reachability_analysis.py -s ${SEED} -n 0
```

## References

If you use this code in your research, please cite the following paper:

[Scaling Mixed-Integer Programming for Certification of Neural Network Controllers Using Bounds Tightening](https://arxiv.org/abs/2403.17874)
