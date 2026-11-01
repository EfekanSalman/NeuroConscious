# !/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Internal state increment constants
# Amount each need rises per InternalState.update(delta_time=1.0) call (0.0 to 1.0, where 1.0 is the maximum).
# Note: a normal awake step calls update() twice (once in thinking, once in acting), so the
# effective rise per simulation step is double these values.
DEFAULT_HUNGER_INCREASE = 0.01
DEFAULT_FATIGUE_INCREASE = 0.015
# Thirst rises at the same rate as hunger so water is as available, relative to need, as food.
# At 0.02 agents were in critical thirst (> 0.85) for 57% of steps, which saturates experiments.
DEFAULT_THIRST_INCREASE = 0.01

# Probability that each cell in the agent's 3x3 view is perceived correctly (otherwise it reads 'unknown').
PERCEPTION_ACCURACY = 0.95

# Number of recent percepts the agent's working-memory buffer (FIFO) retains.
WORKING_MEMORY_CAPACITY = 5

# Global workspace (Global Workspace Theory). Off by default: the rule-based DecisionMaker decides.
USE_GLOBAL_WORKSPACE = False
WORKSPACE_CAPACITY = 1       # Bids admitted and broadcast per step.
WORKSPACE_PERSISTENCE = 0.15 # Salience bonus for the topic that won the previous step.
WORKSPACE_FEEDBACK = True    # Whether the broadcast steers perceptual attention.
WORKSPACE_MEMORY_SLOTS = 0   # Sightings the workspace keeps in mind; 0 = use the FIFO working-memory buffer.
WORKSPACE_MEMORY_POLICY = "salience"  # How sightings enter a full memory: 'salience', 'relevance' or 'fifo' (control).
WORKSPACE_MEMORY_RECALL = "nearest"   # Which remembered location is used: 'nearest' or 'recent'.

# The same competitive sighting memory, attached to the agent without a global workspace (an experimental
# control for the workspace). 0 = off. Ignored when the workspace has its own memory.
SIGHTING_MEMORY_SLOTS = 0
SIGHTING_MEMORY_POLICY = "salience"
SIGHTING_MEMORY_RECALL = "nearest"

# Homeostatic drive D(H) = (sum |h* - h|^n)^(1/m) over hunger, thirst and fatigue (set points all 0).
DRIVE_EXPONENT_N = 4
DRIVE_EXPONENT_M = 2

# Homeostatic controller: a learner that picks which need to attend to by drive-reduction reward.
# Off by default; when on it replaces the rule-based DecisionMaker (and cannot be combined with the workspace).
USE_HOMEOSTATIC_CONTROLLER = False
CONTROLLER_GAMMA = 0.95
CONTROLLER_LEARNING_RATE = 0.001
CONTROLLER_EPSILON_DECAY = 0.999   # Per update; 1.0 keeps the choice random (a control).
CONTROLLER_EPSILON_MIN = 0.05
CONTROLLER_REWARD_N = 4            # Exponents of the drive used as reward. The reported metric always uses
CONTROLLER_REWARD_M = 2            # DRIVE_EXPONENT_N / _M so conditions stay comparable.

# Mood threshold values
MOOD_HAPPY_THRESHOLD = 0.3    # Mood values above this indicate a happy state. Range: -1.0 (sad) to 1.0 (happy)
MOOD_SAD_THRESHOLD = 0.8      # Mood values below this indicate a sad state. Range: -1.0 (sad) to 1.0 (happy)

# Navigation. Off by default: the agent steps greedily toward a remembered target and gives it up when every
# closer step is blocked. When on, it plans a shortest path around the obstacles it has seen (breadth-first).
USE_PATH_PLANNING = False

# Exploration. Off by default: with nothing remembered the agent takes the least-visited neighbouring cell.
# When on, it walks to the nearest cell that has not been in view for EXPLORATION_REVISIT_STEPS (or never).
USE_FRONTIER_EXPLORATION = False
EXPLORATION_REVISIT_STEPS = 100
# How the beacon regime's cue (which half of the grid holds resources) reaches the consumers of the cue (memory
# recall, exploration): 'none' (it never does), 'wired' (every consumer reads the sensor directly) or 'broadcast'
# (only through the global workspace, when the cue wins a slot). Has an effect only when World's BEACON_CYCLE > 0.
BEACON_ACCESS = "none"
# Steps a remembered cue (the trace of the last cue that was broadcast) keeps steering recall and exploration after
# the cue has left the workspace. 0 (default) = no trace: the cue counts only on steps when it is in the workspace.
CUE_TRACE_TTL = 0
