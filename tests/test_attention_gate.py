import unittest
from types import SimpleNamespace
from unittest import mock

from agent import base_agent
from agent.base_agent import Agent


def stub(intended="water", actual="food", last=None, step=100, last_step=95, seen=False):
    return SimpleNamespace(
        attention_focus=intended, attention_actual=actual, attention_last_oriented=last,
        attention_last_orient_step=last_step, attention_silence=0, current_time_step=step,
        perception={"food_in_sight": seen, "water_in_sight": seen},
        attention_stats={"orients": 0, "intended_steps": 0, "correct_steps": 0})


class AttentionControlTest(unittest.TestCase):
    def control(self, agent, policy, free=False):
        with mock.patch.multiple(base_agent, ATTENTION_POLICY=policy, ATTENTION_ORIENT_FREE=free,
                                 ATTENTION_INTERVAL=20, ATTENTION_SILENCE=8):
            return Agent._attention_control(agent, "drink_water")

    def test_none_never_orients(self):
        agent = stub()
        self.assertEqual(self.control(agent, "none"), "drink_water")
        self.assertEqual(agent.attention_actual, "food")

    def test_oracle_orients_only_when_wrong_and_costs_the_step(self):
        agent = stub(intended="water", actual="food")
        self.assertEqual(self.control(agent, "oracle"), "explore")
        self.assertEqual(agent.attention_actual, "water")
        self.assertEqual(self.control(agent, "oracle"), "drink_water")  # already right

    def test_free_orient_keeps_the_action(self):
        agent = stub()
        self.assertEqual(self.control(agent, "oracle", free=True), "drink_water")
        self.assertEqual(agent.attention_actual, "water")

    def test_switch_cannot_see_a_capture(self):
        agent = stub(intended="water", actual="food", last="water")  # captured, but the agent last oriented to water
        self.assertEqual(self.control(agent, "switch"), "drink_water")
        self.assertEqual(agent.attention_actual, "food")

    def test_interval_reorients_when_due(self):
        agent = stub(intended="water", actual="food", last="water", step=120, last_step=95)
        self.assertEqual(self.control(agent, "interval"), "explore")

    def test_silence_reorients_after_unseen_steps(self):
        agent = stub(intended="water", actual="food", last="water")
        agent.attention_silence = 7
        self.assertEqual(self.control(agent, "silence"), "explore")  # the call counts the 8th silent step

    def test_no_intended_channel_leaves_the_action(self):
        agent = stub(intended="rest")
        self.assertEqual(self.control(agent, "oracle"), "drink_water")


if __name__ == '__main__':
    unittest.main()
