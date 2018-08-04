#!/usr/bin/env python

import sys
sys.path.append("../../")

import subprocess
import atexit
import time
import argparse

parser = argparse.ArgumentParser(description="Train or test neural net motor controller")
parser.add_argument(
    "--hparams",
    type=str, required=True)
parser.add_argument(
    "--logdir",
    type=str, required=True)
parser.add_argument(
    "--store-episodes",
    dest="store_episodes",
    action="store_true",
    default=False)
args = parser.parse_args()

ps = []

common_params = [
    "python", "agent.py",
    "--hparams", args.hparams, "--logdir", args.logdir] + \
    (["--store-episodes"] if args.store_episodes else [])

agent_id = 0
for i in range(1):
    ps.append(subprocess.Popen(common_params + \
        ["--visualize", "--id", str(agent_id)]))
    agent_id += 1

for i in range(1):
    ps.append(subprocess.Popen(common_params + \
        ["--visualize", "--validation", "--id", str(agent_id)]))
    agent_id += 1

for i in range(1):
    ps.append(subprocess.Popen(common_params + \
        ["--validation", "--id", str(agent_id)]))
    agent_id += 1

for i in range(4):
    ps.append(subprocess.Popen(common_params + \
        ["--id", str(agent_id)]))
    agent_id += 1

for i in range(0):
    ps.append(subprocess.Popen(common_params + \
        ["--random_start", "--id", str(agent_id)]))
    agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)

