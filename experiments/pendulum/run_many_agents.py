#!/usr/bin/env python

import sys
sys.path.append("../../")

import subprocess
import atexit
import time

ps = []

agent_id = 0
for i in range(0):
    ps.append(subprocess.Popen(
        ["python", "agent.py", "--visualize=1", "--id", str(agent_id)]
    ))
    agent_id += 1

for i in range(4):
    ps.append(subprocess.Popen(["python", "agent.py", "--id", str(agent_id)]))
    agent_id += 1

for i in range(0):
    ps.append(subprocess.Popen(
        ["python", "agent.py", "--random_start", "--id", str(agent_id)]
    ))
    agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)
