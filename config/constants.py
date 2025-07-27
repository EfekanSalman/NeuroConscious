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
DEFAULT_HUNGER_INCREASE = 0.1  # Amount hunger increases per time step (e.g., 0.0 to 1.0 where 1.0 is max hunger)
DEFAULT_FATIGUE_INCREASE = 0.05 # Amount fatigue increases per time step

# Mood threshold values
MOOD_HAPPY_THRESHOLD = 0.3    # Mood values above this indicate a happy state. Range: -1.0 (sad) to 1.0 (happy)
MOOD_SAD_THRESHOLD = 0.8      # Mood values below this indicate a sad state. Range: -1.0 (sad) to 1.0 (happy)