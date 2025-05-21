# Reacher with Obstacles
This repo contains some experiments for the paper "Exploiting Robot Abstractions in Episodic RL via Reward Shaping and Heuristics". More specifically, these experiments concern the Reacher task (more information [here](https://gymnasium.farama.org/environments/mujoco/reacher/) ), trained both with the RL methods presented in the paper, and using trajectory optimization. 

Use the pixi environment to run the code. Install pixi on https://pixi.sh/latest/. Then, install the environment by running `pixi install`.

## RL Training


Run 
    
    pixi run rl-train [EXPERIMENT ID]
with experiment id referring to the key
of the `EXPERIMENTS` map

Example:

    pixi run rl-train 1a [--seed <SEED>]

Use tensorboard to monitor the training

Reward function is designed to return values >0 at each step when the goal is reached.
So when the return is stable above 0 a good policy has been found.

Note: models are automatically saved at the end of each training phase and automatically reloaded at each run of the experiment

### Play the learned policy

Example:

    pixi run rl-eval 1a [--seed <SEED>]


Note: `info[reward_goal]` printed at the end of the policy execution is 1.0 if the goal has been reached.

### Configure the environment

To add or modify environment configurations, use `CONFIGS` structure in `reacher_v6.py`

Environments ids are automatically registered using all the keys in `CONFIGS` 

    Reacher-v6_{conf}
    Reacher3-v6_{conf}
    MARRtinoArm_{conf}

These environments use a standard dense reward based on distance to the target.

To use reward heuristics, use the following environments 

    Reacher-v6_{conf}_{rhV|rsV}
    Reacher3-v6_{conf}_{rhV|rsV}
    MARRtinoArm_{conf}_{rhV|rsV}

where `rhV` is a heuristic reward based on discrete V (optimal V of abstract MDP),
while 'rsV' is a reward shaping function based on discrete V.


## Trajectory Optimization

To run and save trajectory optimization, run

    pixi run trajopt-train

To play the found trajectory

    pixi run trajopt-eval


