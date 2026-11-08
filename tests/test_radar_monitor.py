import random
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import base_agent
from agent.base_agent import Agent
from core.workspace.memory import Sighting, WorkspaceMemory


def view_agent(center_kind=None, right=None):
    """An agent at (5, 5) whose 3x3 view shows `right` one cell to the right (5, 6) and `center_kind` at its feet."""
    grid = [["empty"] * 3 for _ in range(3)]
    grid[1][1] = center_kind or "empty"
    grid[1][2] = right or "empty"
    log = []
    return SimpleNamespace(
        pos_x=5, pos_y=5, current_time_step=10, perception={"local_grid_view": grid, "ghost_radar": []},
        internal_state=SimpleNamespace(hunger=0.5, thirst=0.5, fatigue=0.5), radar_trust_now=True, radar_monitor="none",
        radar_verified=lambda hit: log.append(hit), log=log, cue_allows=lambda cell: True)


class RadarSightingTest(unittest.TestCase):
    def test_a_radar_sighting_confirmed_by_the_view_is_a_hit(self):
        agent, memory = view_agent(right="food"), WorkspaceMemory(5)
        memory.sightings.append(Sighting("food", (5, 6), 5, "radar"))
        memory.observe(agent)
        self.assertEqual(agent.log, [True])
        self.assertEqual(memory.sightings[0].source, "view")

    def test_a_radar_sighting_the_view_contradicts_is_a_ghost(self):
        agent, memory = view_agent(), WorkspaceMemory(5)
        memory.sightings.append(Sighting("food", (5, 6), 5, "radar"))
        self.assertIsNone(memory.recall("food", agent))
        self.assertEqual(agent.log, [False])
        self.assertEqual(memory.sightings, [])

    def test_untrusted_radar_sightings_are_not_recalled_but_stay(self):
        agent, memory = view_agent(), WorkspaceMemory(5)
        memory.sightings.append(Sighting("water", (9, 9), 5, "radar"))
        agent.radar_trust_now = False
        self.assertIsNone(memory.recall("water", agent))
        self.assertEqual(len(memory.sightings), 1)
        agent.radar_trust_now = True
        self.assertEqual(memory.recall("water", agent), (9, 9))

    def test_radar_detections_enter_memory_as_radar_sightings_unless_never(self):
        agent, memory = view_agent(), WorkspaceMemory(5)
        agent.perception["ghost_radar"] = [("food", (8, 8))]
        memory.observe(agent)
        self.assertEqual([(s.kind, s.source) for s in memory.sightings], [("food", "radar")])
        agent2, memory2 = view_agent(), WorkspaceMemory(5)
        agent2.perception["ghost_radar"] = [("food", (8, 8))]
        agent2.radar_monitor = "never"
        memory2.observe(agent2)
        self.assertEqual(memory2.sightings, [])


class MonitorTest(unittest.TestCase):
    def stub(self, **kw):
        base = dict(current_time_step=0, radar_phase=0, radar_reliability=0.5, radar_coin=(-1, True),
                    radar_stats={"hits": 0, "ghosts": 0})
        base.update(kw)
        agent = SimpleNamespace(**base)
        agent.radar_high_ghost_regime = lambda: Agent.radar_high_ghost_regime(agent)
        return agent

    def trust(self, agent, mode):
        with mock.patch.object(base_agent, "RADAR_MONITOR", mode), mock.patch.object(base_agent, "GHOST_PERIOD", 100):
            return Agent.trust_radar(agent)

    def test_regimes_alternate_every_period(self):
        with mock.patch.object(base_agent, "GHOST_PERIOD", 100):
            flags = [Agent.radar_high_ghost_regime(self.stub(current_time_step=t)) for t in (0, 99, 100, 199, 200)]
        self.assertEqual(flags, [False, False, True, True, False])

    def test_oracle_trusts_only_in_the_low_regime(self):
        self.assertTrue(self.trust(self.stub(current_time_step=50), "oracle"))
        self.assertFalse(self.trust(self.stub(current_time_step=150), "oracle"))

    def test_none_and_never(self):
        self.assertTrue(self.trust(self.stub(), "none"))
        self.assertFalse(self.trust(self.stub(), "never"))

    def test_learned_follows_the_running_average(self):
        agent = self.stub(radar_stats={"hits": 0, "ghosts": 0})
        with mock.patch.object(base_agent, "RADAR_MONITOR_RATE", 0.5), mock.patch.object(base_agent, "RADAR_PROBE", 0.0), \
                mock.patch.object(base_agent, "RADAR_TRUST", 0.6):
            self.assertFalse(self.trust(agent, "learned"))  # starts at 0.5, below the trust level
            Agent.radar_verified(agent, True)
            self.assertEqual(agent.radar_reliability, 0.75)
            Agent.radar_verified(agent, False)
            Agent.radar_verified(agent, False)
            self.assertEqual(agent.radar_reliability, 0.1875)
            self.assertFalse(self.trust(agent, "learned"))
        self.assertEqual(agent.radar_stats, {"hits": 1, "ghosts": 2})

    def test_wrong_is_constant_within_a_block(self):
        agent = self.stub(current_time_step=10)
        random.seed(3)
        first = self.trust(agent, "wrong")
        agent.current_time_step = 60
        self.assertEqual(self.trust(agent, "wrong"), first)


if __name__ == "__main__":
    unittest.main()
