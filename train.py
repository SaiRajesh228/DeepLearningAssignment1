#!/usr/bin/env python
"""
train.py

This script uses the model module to train the deep neural network on the Fashion MNIST dataset.
It accepts command line arguments for wandb entity and project and runs the complete pipeline.
"""

import argparse
import wandb
from model import log_q1_samples, log_q2_experiments, execute_training, sweep_config

def main():
    parser = argparse.ArgumentParser(description="Train a deep neural network with wandb logging")
    parser.add_argument("--wandb_entity", type=str, required=True, help="Wandb entity name")
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name")
    args = parser.parse_args()

    # Initialize wandb
    wandb.login()

    # 1. Log sample images (Question 1)
    with wandb.init(entity=args.wandb_entity, project=args.wandb_project, 
                  name="Q1_Samples", job_type="logging"):
        log_q1_samples()

    # 2. Run dedicated Question 2 experiments
    log_q2_experiments(args.wandb_entity, args.wandb_project)

    # 3. Run a hyperparameter sweep for Question 2
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id, function=execute_training, count=2)

if __name__ == "__main__":
    main()