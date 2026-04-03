#!/bin/bash

#Wipe ssh keys, wandb cache and git repos
rm -rf ~/.ssh
rm -rf ~/.cache/wandb
rm -rf ~/.config/wandb
rm -rf RL

echo "Wiped ssh keys, wandb cache and git repos"