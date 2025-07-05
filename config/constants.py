# Internal state increment constants
DEFAULT_HUNGER_INCREASE = 0.1  # Amount hunger increases per time step (e.g., 0.0 to 1.0 where 1.0 is max hunger)
DEFAULT_FATIGUE_INCREASE = 0.05 # Amount fatigue increases per time step

# Mood threshold values
MOOD_HAPPY_THRESHOLD = 0.3    # Mood values above this indicate a happy state. Range: -1.0 (sad) to 1.0 (happy)
MOOD_SAD_THRESHOLD = 0.8      # Mood values below this indicate a sad state. Range: -1.0 (sad) to 1.0 (happy)