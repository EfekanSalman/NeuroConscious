import random
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import base_agent
from agent.base_agent import Agent

DIRS = list(Agent._MOVE_OFFSETS)


def stub(table=None, mapping=None):
    return SimpleNamespace(
        motor_table=table or {d: d for d in DIRS}, motor_history={d: [] for d in DIRS},
        motor_last_seen={d: -1 for d in DIRS}, motor_stats={"moves": 0, "as_intended": 0},
        current_time_step=10, environment=SimpleNamespace(action_map=mapping or {d: d for d in DIRS}),
        motor_q={w: {c: (0.5 if c == w else 0.0) for c in DIRS} for w in DIRS},
        _MOVE_OFFSETS=Agent._MOVE_OFFSETS, _DIRECTION_OF_DELTA=Agent._DIRECTION_OF_DELTA)


class MotorModelTest(unittest.TestCase):
    def command(self, agent, wanted, model):
        with mock.patch.object(base_agent, "MOTOR_MODEL", model):
            return Agent._motor_command(agent, wanted)

    def test_none_issues_the_wanted_direction(self):
        self.assertEqual(self.command(stub(mapping={"move_up": "move_down", "move_down": "move_up",
                                                    "move_left": "move_left", "move_right": "move_right"}),
                                      "move_up", "none"), "move_up")

    def test_oracle_inverts_the_true_mapping(self):
        swapped = {"move_up": "move_down", "move_down": "move_up", "move_left": "move_left", "move_right": "move_right"}
        self.assertEqual(self.command(stub(mapping=swapped), "move_up", "oracle"), "move_down")

    def observe(self, agent, command, delta, wanted, model):
        with mock.patch.object(base_agent, "MOTOR_MODEL", model), mock.patch.object(base_agent, "MOTOR_WINDOW", 3):
            Agent._observe_motor(agent, command, wanted, delta)

    def test_last_learns_what_a_command_did(self):
        agent = stub()
        self.observe(agent, "move_up", (1, 0), "move_up", "last")  # up produced down
        self.assertEqual(agent.motor_table["move_up"], "move_down")
        self.assertEqual(agent.motor_stats, {"moves": 1, "as_intended": 0})
        # now wanting 'down' is best served by the command known to produce it, here move_up (and move_down is stale)
        self.assertEqual(self.command(agent, "move_down", "last"), "move_up")

    def test_unknown_direction_tries_the_longest_unchecked_command(self):
        agent = stub()
        self.observe(agent, "move_up", (0, -1), "move_up", "last")  # up produced left; nothing is known to go up now
        agent.motor_last_seen.update({"move_up": 10, "move_down": 3, "move_left": 7, "move_right": 5})
        self.assertEqual(self.command(agent, "move_up", "last"), "move_down")

    def test_vote_ignores_one_slip(self):
        agent = stub()
        for delta in [(-1, 0), (-1, 0), (0, 1), (-1, 0)]:
            self.observe(agent, "move_up", delta, "move_up", "vote")
        self.assertEqual(agent.motor_table["move_up"], "move_up")

    def test_blocked_moves_carry_no_information(self):
        agent = stub()
        self.observe(agent, "move_up", (0, 0), "move_up", "last")
        self.assertEqual(agent.motor_stats["moves"], 0)

    def test_executed_direction_follows_the_world_map(self):
        agent = stub(mapping={"move_up": "move_left", "move_down": "move_down", "move_left": "move_up",
                              "move_right": "move_right"})
        self.assertEqual(Agent._executed_direction(agent, "move_up"), "move_left")


class ModelFreeTest(unittest.TestCase):
    def command(self, agent, wanted, epsilon=0.0):
        with mock.patch.object(base_agent, "MOTOR_MODEL", "modelfree"), mock.patch.object(base_agent, "MOTOR_EPSILON", epsilon):
            return Agent._motor_command(agent, wanted)

    def observe(self, agent, command, delta, wanted):
        with mock.patch.object(base_agent, "MOTOR_MODEL", "modelfree"), mock.patch.object(base_agent, "MOTOR_LR", 0.5):
            Agent._observe_motor(agent, command, wanted, delta)

    def test_starts_with_the_identity_command(self):
        self.assertEqual(self.command(stub(), "move_left"), "move_left")

    def test_learns_per_wanted_direction_only(self):
        agent = stub()
        # wanting 'up', issuing 'up' went down: the value of up-for-up falls, and 'down' (which went down)... is not
        # credited for 'up'; nothing is learned about other wanted directions
        self.observe(agent, "move_up", (1, 0), "move_up")
        self.assertEqual(agent.motor_q["move_up"]["move_up"], 0.25)
        self.assertEqual(agent.motor_q["move_down"]["move_up"], 0.0)
        self.assertEqual(agent.motor_q["move_left"]["move_left"], 0.5)

    def test_success_raises_the_value_and_blocked_moves_teach_nothing(self):
        agent = stub()
        self.observe(agent, "move_right", (-1, 0), "move_up")  # right produced up: success for wanted 'up'
        self.assertEqual(agent.motor_q["move_up"]["move_right"], 0.5)
        before = dict(agent.motor_q["move_up"])
        self.observe(agent, "move_down", (0, 0), "move_up")
        self.assertEqual(agent.motor_q["move_up"], before)

    def test_epsilon_one_issues_a_random_command(self):
        with mock.patch.object(base_agent.random, "choice", return_value="move_down"):
            self.assertEqual(self.command(stub(), "move_left", epsilon=1.0), "move_down")


class WorldRemapTest(unittest.TestCase):
    def test_remap_happens_only_when_enabled_and_is_a_permutation(self):
        from environment import world as world_module
        world = world_module.World()
        before = dict(world.action_map)
        world.time_step = 150
        world.update_environment()
        self.assertEqual(world.action_map, before)
        with mock.patch.object(world_module, "ACTION_REMAP_PERIOD", 150):
            random.seed(1)
            world.update_environment()
        self.assertEqual(sorted(world.action_map.values()), sorted(DIRS))
        self.assertEqual(sorted(world.action_map), sorted(DIRS))


if __name__ == '__main__':
    unittest.main()
