import unittest
from types import SimpleNamespace
from unittest import mock

from core.decision import decision_maker
from core.decision.decision_maker import DecisionMaker


def _agent(hunger, thirst, fatigue):
    return SimpleNamespace(internal_state=SimpleNamespace(hunger=hunger, thirst=thirst, fatigue=fatigue),
                           internal_monologue="")


class CriticalOverrideTest(unittest.TestCase):
    def decide(self, agent):
        return DecisionMaker().decide_final_action(agent, "reactive")

    def test_fixed_order_serves_hunger_first(self):
        self.assertEqual(self.decide(_agent(0.9, 0.99, 0.5)), "seek_food")

    def test_neediest_order_serves_the_highest_level(self):
        with mock.patch.object(decision_maker, "CRITICAL_OVERRIDE_ORDER", "neediest"):
            self.assertEqual(self.decide(_agent(0.9, 0.99, 0.5)), "drink_water")
            self.assertEqual(self.decide(_agent(0.9, 0.86, 0.5)), "seek_food")

    def test_neediest_ties_keep_the_fixed_order(self):
        with mock.patch.object(decision_maker, "CRITICAL_OVERRIDE_ORDER", "neediest"):
            self.assertEqual(self.decide(_agent(0.9, 0.9, 0.9)), "seek_food")

    def test_a_lower_level_makes_a_lower_need_critical(self):
        with mock.patch.object(decision_maker, "CRITICAL_OVERRIDE_LEVEL", 0.3):
            self.assertEqual(self.decide(_agent(0.1, 0.4, 0.2)), "drink_water")


if __name__ == "__main__":
    unittest.main()


class OpportunismTest(unittest.TestCase):
    def agent(self, here, hunger=0.9, thirst=0.8):
        agent = _agent(hunger, thirst, 0.1)
        agent.perception = {"local_grid_view": [["x"] * 3, ["x", here, "x"], ["x"] * 3]}
        return agent

    def decide(self, agent, **patches):
        with mock.patch.multiple(decision_maker, CRITICAL_OVERRIDE_LEVEL=0.3, CRITICAL_OVERRIDE_ORDER="neediest", **patches):
            return DecisionMaker().decide_final_action(agent, "reactive")

    def test_off_by_default_seeks_the_neediest(self):
        self.assertEqual(self.decide(self.agent("water")), "seek_food")

    def test_standing_on_water_while_hungrier_and_thirsty_drinks(self):
        self.assertEqual(self.decide(self.agent("water"), CRITICAL_OPPORTUNISM=True), "drink_water")

    def test_the_neediest_need_met_here_is_not_displaced(self):
        self.assertEqual(self.decide(self.agent("food"), CRITICAL_OPPORTUNISM=True), "seek_food")

    def test_nothing_here_keeps_the_neediest(self):
        self.assertEqual(self.decide(self.agent("empty"), CRITICAL_OPPORTUNISM=True), "seek_food")
