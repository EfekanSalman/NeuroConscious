# ──────────────────────────────────────────────────────────────
# NeuroConsciousV2 — Central Configuration Constants
# ──────────────────────────────────────────────────────────────

# Simulation
TOTAL_STEPS = 1000
LOG_INTERVAL = 100
DAY_NIGHT_CYCLE_LENGTH = 10  # steps per full cycle
FOOD_SPAWN_PROBABILITY = 0.5

# Internal State
DEFAULT_HUNGER_INCREASE = 0.1
DEFAULT_FATIGUE_INCREASE = 0.05
NIGHT_TIME_MULTIPLIER = 1.5

# Action Effects
FOOD_HUNGER_REDUCTION = 0.4
REST_FATIGUE_REDUCTION = 0.3
EXPLORE_PENALTY = -0.01

# Mood Thresholds
MOOD_HAPPY_THRESHOLD = 0.3
MOOD_SAD_THRESHOLD = 0.8

# Emotion Update Rates
JOY_FOOD_BOOST = 0.05
JOY_HUNGER_DECAY_FACTOR = 0.02
FEAR_NIGHT_INCREASE = 0.03
FEAR_DAY_DECREASE = 0.02
CURIOSITY_HIGH = 0.8
CURIOSITY_LOW = 0.3
CURIOSITY_NEED_THRESHOLD = 0.3

# Motivation Thresholds
FRUSTRATION_OVERRIDE_THRESHOLD = 0.7
FEAR_INHIBIT_THRESHOLD = 0.6
CURIOSITY_EXPLORE_THRESHOLD = 0.6
JOY_EXPLORE_THRESHOLD = 0.7
JOY_EXPLORE_PROBABILITY = 0.3
NEED_ACTION_THRESHOLD = 0.6
FEAR_SECONDARY_THRESHOLD = 0.4

# Q-Learning Hyperparameters
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_GAMMA = 0.9
Q_LEARNING_EPSILON = 0.2

# Hybrid Decision: weight of emotion-driven vs Q-learning (0.0 = pure Q, 1.0 = pure emotion)
EMOTION_WEIGHT = 0.4

# Episodic Memory
EPISODIC_MEMORY_CAPACITY = 50

# Output Paths
OUTPUT_DIR = "output"
ACTION_COUNTS_FILE = "action_counts.json"
Q_TABLE_FILE = "q_table.json"
ACTION_FREQUENCY_PLOT = "action_frequency.png"
AGENT_STATE_PLOT = "agent_state_plot.png"
