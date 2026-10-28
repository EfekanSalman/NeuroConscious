"""Unit tests for the homeostatic drive. Run from the repository root: python -m unittest discover tests"""

import unittest
from types import SimpleNamespace

from core.homeostasis import OPTIONS, HomeostaticController, drive, drive_of_state


class DriveTest(unittest.TestCase):
    def test_zero_when_every_need_is_satisfied(self):
        self.assertEqual(drive([0.0, 0.0, 0.0]), 0.0)

    def test_matches_the_published_form(self):
        self.assertAlmostEqual(drive([0.5, 0.0, 0.0], n=4, m=2), 0.5 ** 2)
        self.assertAlmostEqual(drive([1.0, 1.0, 1.0], n=4, m=2), 3 ** 0.5)

    def test_grows_with_every_single_need(self):
        base = drive([0.3, 0.3, 0.3])
        for index in range(3):
            raised = [0.3, 0.3, 0.3]
            raised[index] = 0.6
            self.assertGreater(drive(raised), base)

    def test_is_convex_so_one_severe_deficit_costs_more_than_two_moderate_ones(self):
        self.assertGreater(drive([0.8, 0.0, 0.0]), drive([0.4, 0.4, 0.0]))

    def test_n_equal_one_is_linear_in_the_summed_deficit(self):
        self.assertAlmostEqual(drive([0.5, 0.25, 0.0], n=1, m=1), 0.75)

    def test_accepts_a_state_object_and_a_snapshot_dictionary(self):
        state = SimpleNamespace(hunger=0.5, thirst=0.0, fatigue=0.0)
        self.assertAlmostEqual(drive_of_state(state), 0.25)
        self.assertAlmostEqual(drive_of_state({"hunger": 0.5, "thirst": 0.0, "fatigue": 0.0, "mood_value": 1.0}), 0.25)


class FakeLearner:
    """Records updates and always picks the first option."""

    def __init__(self):
        self.updates = []

    def choose_action(self, hunger, fatigue, thirst):
        return OPTIONS[0]

    def update(self, *args):
        self.updates.append(args)


def levels(hunger=0.0, thirst=0.0, fatigue=0.0):
    return SimpleNamespace(hunger=hunger, thirst=thirst, fatigue=fatigue)


class HomeostaticControllerTest(unittest.TestCase):
    def test_reward_is_the_drive_reduction(self):
        controller = HomeostaticController(FakeLearner())
        before, after = levels(hunger=0.9), levels(hunger=0.2)
        self.assertAlmostEqual(controller.reward(before, after), drive_of_state(before) - drive_of_state(after))
        self.assertGreater(controller.reward(before, after), 0)
        self.assertLess(controller.reward(after, before), 0)

    def test_reward_uses_the_controllers_own_exponents(self):
        linear = HomeostaticController(FakeLearner(), reward_n=1, reward_m=1)
        self.assertAlmostEqual(linear.reward(levels(hunger=0.9), levels(hunger=0.4)), 0.5)

    def test_convex_reward_values_relieving_a_severe_deficit_above_a_mild_one(self):
        convex, linear = HomeostaticController(FakeLearner()), HomeostaticController(FakeLearner(), 1, 1)
        severe = convex.reward(levels(hunger=0.9), levels(hunger=0.6))
        mild = convex.reward(levels(hunger=0.4), levels(hunger=0.1))
        self.assertGreater(severe, 3 * mild)
        self.assertAlmostEqual(linear.reward(levels(hunger=0.9), levels(hunger=0.6)),
                               linear.reward(levels(hunger=0.4), levels(hunger=0.1)))

    def test_decide_returns_an_option_and_counts_it(self):
        controller = HomeostaticController(FakeLearner())
        agent = SimpleNamespace(internal_state=levels(hunger=0.5))
        self.assertIn(controller.decide(agent), OPTIONS)
        self.assertEqual(controller.option_counts[OPTIONS[0]], 1)

    def test_learn_passes_the_transition_to_the_learner(self):
        learner = FakeLearner()
        controller = HomeostaticController(learner)
        previous = {"hunger": 0.9, "fatigue": 0.1, "thirst": 0.2, "mood_value": 0.0}
        reward = controller.learn(previous, "seek_food", levels(hunger=0.2, fatigue=0.1, thirst=0.2))
        self.assertEqual(learner.updates[0][:5], (0.9, 0.1, 0.2, "seek_food", reward))
        self.assertAlmostEqual(controller.total_reward, reward)

    def test_default_controller_learns_from_a_real_dqn(self):
        controller = HomeostaticController.default()
        controller.learn({"hunger": 0.9, "fatigue": 0.0, "thirst": 0.0}, "seek_food", levels(hunger=0.2))
        self.assertEqual(len(controller.learner.replay_buffer), 1)


class AgentControllerTest(unittest.TestCase):
    def build(self, **flags):
        import io
        import contextlib
        from unittest import mock
        with contextlib.redirect_stdout(io.StringIO()):
            import agent.base_agent as base_agent
            from core.mood.basic_mood import BasicMoodStrategy
            patches = [mock.patch.object(base_agent, name, value) for name, value in flags.items()]
            for patch in patches:
                patch.start()
            try:
                return base_agent.Agent(name="T", mood_strategy=BasicMoodStrategy())
            finally:
                for patch in patches:
                    patch.stop()

    def test_controller_is_off_by_default(self):
        agent = self.build()
        self.assertIsNone(agent.controller)
        self.assertFalse(agent.has_external_policy)

    def test_controller_replaces_the_rule_chain_when_enabled(self):
        agent = self.build(USE_HOMEOSTATIC_CONTROLLER=True)
        self.assertIsNotNone(agent.controller)
        self.assertTrue(agent.has_external_policy)

    def test_controller_and_workspace_cannot_be_combined(self):
        with self.assertRaises(ValueError):
            self.build(USE_HOMEOSTATIC_CONTROLLER=True, USE_GLOBAL_WORKSPACE=True)


if __name__ == "__main__":
    unittest.main()
