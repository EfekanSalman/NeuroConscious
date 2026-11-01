"""Tests for the beacon regime: the world, the cue's route through the workspace, and its consumers."""

import contextlib
import io
import random
import unittest
from types import SimpleNamespace
from unittest import mock

from core.navigation.regions import half_of
from core.workspace import GlobalWorkspace
from core.workspace import specialists as specialists_module
from core.workspace.specialists import BeaconSpecialist, NeedSpecialist, read_regime
from environment import world as world_module
from environment.world import World
from tests.test_workspace import make_agent


def beacon_world(seed=0):
    random.seed(seed)
    patches = {"BEACON_CYCLE": 10, "GRID_SIZE": 12, "FOOD_SITES": 6, "WATER_SITES": 6, "SITE_REGROW_STEPS": 3}
    with contextlib.redirect_stdout(io.StringIO()), mock.patch.multiple(world_module, **patches):
        world = World()
    return world, patches


def step_world(world, patches, steps):
    with contextlib.redirect_stdout(io.StringIO()), mock.patch.multiple(world_module, **patches):
        for _ in range(steps):
            world.update_environment()
            world.time_step += 1


class BeaconWorldTest(unittest.TestCase):
    def test_half_of_splits_the_columns(self):
        self.assertEqual([half_of(c, 4) for c in range(4)], ["left", "left", "right", "right"])

    def test_both_halves_get_sites_of_each_kind(self):
        world, _ = beacon_world()
        for kind in ("food", "water"):
            halves = {half_of(col, 12) for _, col in world._sites[kind]}
            self.assertEqual(halves, {"left", "right"})

    def test_only_the_active_half_holds_resources(self):
        world, patches = beacon_world()
        for _ in range(8):
            step_world(world, patches, 10)
            for row in range(12):
                for col in range(12):
                    if world.grid[row][col] in ("food", "water"):
                        self.assertEqual(half_of(col, 12), world.active_half)

    def test_the_active_half_changes_over_time(self):
        world, patches = beacon_world(seed=1)
        seen = set()
        for _ in range(12):
            step_world(world, patches, 10)
            seen.add(world.active_half)
        self.assertEqual(seen, {"left", "right"})

    def test_no_regime_by_default(self):
        random.seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(World().active_half)


class BeaconWorkspaceTest(unittest.TestCase):
    def agent(self, half="left", **levels):
        agent = make_agent(**levels)
        agent.environment = SimpleNamespace(active_half=half)
        agent.regime_cue = None
        return agent

    def test_inverted_sensor_reports_the_other_half(self):
        agent = self.agent("left")
        with mock.patch.object(specialists_module, "BEACON_INVERT", True):
            self.assertEqual(read_regime(agent), "right")
        self.assertEqual(read_regime(agent), "left")

    def test_the_beacon_is_absent_by_default(self):
        names = [s.name for s in GlobalWorkspace.default().specialists]
        self.assertNotIn("beacon", names)
        self.assertIn("beacon", [s.name for s in GlobalWorkspace.default(beacon=True).specialists])

    def test_an_admitted_cue_is_broadcast_but_never_chosen_as_the_action(self):
        workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=2)
        agent = self.agent("right", thirst=0.9)
        self.assertEqual(workspace.decide(agent), "drink_water")
        self.assertEqual(agent.regime_cue, "right")

    def test_a_cue_outcompeted_for_a_slot_is_not_available(self):
        workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=1)
        agent = self.agent("right", thirst=0.9)  # salience 0.81 + 0.5 outbids the cue's 0.6
        self.assertEqual(workspace.decide(agent), "drink_water")
        self.assertIsNone(agent.regime_cue)

    def test_a_cue_that_takes_the_only_slot_is_broadcast_and_the_agent_idles(self):
        with mock.patch.object(specialists_module, "BEACON_SALIENCE", 2.0):
            workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=1)
            agent = self.agent("left", thirst=0.9)
            self.assertEqual(workspace.decide(agent), "explore")
        self.assertEqual(agent.regime_cue, "left")
        self.assertEqual(workspace.winner_counts["cue"], 1)

    def test_a_cue_alone_does_not_make_the_agent_act(self):
        workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=2)
        agent = self.agent("left", thirst=0.1)
        self.assertEqual(workspace.decide(agent), "explore")
        self.assertEqual(workspace.winner_counts["none"], 1)

    def test_without_a_beacon_specialist_the_cue_stays_untouched(self):
        workspace = GlobalWorkspace([NeedSpecialist("thirst")], capacity=2)
        agent = self.agent("left", thirst=0.9)
        agent.regime_cue = "left"
        workspace.decide(agent)
        self.assertEqual(agent.regime_cue, "left")


if __name__ == "__main__":
    unittest.main()


class CueTraceTest(unittest.TestCase):
    def agent(self, half="left", age=None, **levels):
        agent = make_agent(**levels)
        agent.environment = SimpleNamespace(active_half=half)
        agent.regime_cue = None
        if age is not None:
            agent.cue_age = age
        return agent

    def test_interval_policy_is_silent_until_the_cue_is_old_enough(self):
        with mock.patch.object(specialists_module, "BEACON_POLICY", "interval"), \
                mock.patch.object(specialists_module, "BEACON_INTERVAL", 10):
            self.assertIsNone(BeaconSpecialist().propose(self.agent(age=9)))
            bid = BeaconSpecialist().propose(self.agent(age=10))
        self.assertGreater(bid.salience, 5)

    def test_interval_policy_bids_at_once_when_the_cue_was_never_seen(self):
        with mock.patch.object(specialists_module, "BEACON_POLICY", "interval"):
            self.assertIsNotNone(BeaconSpecialist().propose(self.agent()))

    def test_a_broadcast_cue_resets_the_age_and_a_missed_step_ages_it(self):
        workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=2)
        agent = self.agent("right", thirst=0.9, age=7)
        workspace.decide(agent)
        self.assertEqual((agent.cue_trace, agent.cue_age), ("right", 0))
        with mock.patch.object(specialists_module, "BEACON_POLICY", "interval"):
            workspace.decide(agent)  # age 0 < interval: silent
        self.assertEqual(agent.cue_age, 1)

    def test_a_forced_cue_wins_the_only_slot_and_idles_the_agent_once(self):
        with mock.patch.object(specialists_module, "BEACON_POLICY", "interval"):
            workspace = GlobalWorkspace([NeedSpecialist("thirst"), BeaconSpecialist()], capacity=1)
            agent = self.agent("left", thirst=0.95)
            self.assertEqual(workspace.decide(agent), "explore")
            self.assertEqual(workspace.decide(agent), "drink_water")

    def test_the_trace_steers_only_while_fresh(self):
        from agent.base_agent import Agent
        from agent import base_agent
        stub = SimpleNamespace(regime_cue=None, cue_trace="left", cue_age=5)
        with mock.patch.object(base_agent, "CUE_TRACE_TTL", 10):
            self.assertEqual(Agent.effective_cue(stub), "left")
            stub.cue_age = 11
            self.assertIsNone(Agent.effective_cue(stub))
            stub.regime_cue = "right"
            self.assertEqual(Agent.effective_cue(stub), "right")
        self.assertIsNone(Agent.effective_cue(SimpleNamespace(regime_cue=None, cue_trace="left", cue_age=0)))
