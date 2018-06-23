#!/usr/bin/env python

import subprocess
import atexit
import time

ps = []

agent_id = 0
for i in range(1):
    ps.append(subprocess.Popen(
        ['python', 'lunar_lander_agent.py', '--visualize=1', '--id', str(agent_id)]
    ))
    agent_id += 1

for i in range(3):
    ps.append(subprocess.Popen(['python', 'lunar_lander_agent.py', '--id', str(agent_id)]))
    agent_id += 1

for i in range(0):
    ps.append(subprocess.Popen(
        ['python', 'lunar_lander_agent.py', '--random_start', '--id', str(agent_id)]
    ))
    agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)
