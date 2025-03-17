#!/usr/bin/env python
"""
train.py

This script uses the model module to train the deep neural network on the Fashion MNIST dataset.
It accepts command line arguments for wandb entity and project and runs the complete pipeline.

Usage:
    python train.py --wandb_entity myname --wandb_project myprojectname
"""

import argparse
import wandb
from model import log_q1_samples, log_q2_experiments, execute_training, sweep_config

# Global variables to pass to wandb initialization
WANDB_ENTITY = None
WANDB_PROJECT = None

def main():
    parser = argparse.ArgumentParser(description="Train a deep neural network with wandb logging")
    parser.add_argument("--wandb_entity", type=str, required=True, help="Wandb entity name")
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name")
    args = parser.parse_args()

    global WANDB_ENTITY, WANDB_PROJECT
    WANDB_ENTITY = args.wandb_entity
    WANDB_PROJECT = args.wandb_project

    # Set default wandb settings for all runs
    wandb.login()

    # 1. Log sample images (Question 1)
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="Q1_Samples", job_type="logging")
    log_q1_samples()
    wandb.finish()

    # 2. Run dedicated Question 2 experiments
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="Dedicated_Q2", job_type="logging")
    log_q2_experiments()
    wandb.finish()

    # 3. Run a hyperparameter sweep for Question 2
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
    wandb.agent(sweep_id, function=execute_training, count=2)

if __name__ == "__main__":
    main()
