import unittest
from types import SimpleNamespace
from unittest import mock

from core.consciousness import focused_state
from core.consciousness.focused_state import FocusedState


def _agent(focus, hunger, thirst, decision):
    return SimpleNamespace(
        internal_monologue="", attention_focus=focus, has_external_policy=False,
        internal_state=SimpleNamespace(hunger=hunger, thirst=thirst, fatigue=0.0),
        working_memory_buffer=[], current_time_step=0, pos_x=0, pos_y=0,
        _think_default=lambda decision_mode: decision)


class FocusRewriteTest(unittest.TestCase):
    def test_food_focus_rewrites_drinking_by_default(self):
        state = FocusedState(_agent('food', 0.9, 0.8, 'drink_water'))
        self.assertEqual(state.think(), 'seek_food')

    def test_switch_off_keeps_the_decision(self):
        with mock.patch.object(focused_state, 'FOCUS_REWRITES_DECISION', False):
            state = FocusedState(_agent('food', 0.9, 0.8, 'drink_water'))
            self.assertEqual(state.think(), 'drink_water')


if __name__ == '__main__':
    unittest.main()
