"""Unit tests for the global workspace. Run from the repository root: python -m unittest discover tests"""

import unittest
from unittest import mock
from types import SimpleNamespace

from core.workspace import (GlobalWorkspace, GoalSpecialist, NeedSpecialist, Proposal, Specialist,
                            WorkspaceMemory)

MOVE_OFFSETS = {"move_up": (-1, 0), "move_down": (1, 0), "move_left": (0, -1), "move_right": (0, 1)}


def view(centre="empty", **cells):
    """A 3x3 local view; keyword names are up/down/left/right and the centre cell."""
    grid = [["empty"] * 3 for _ in range(3)]
    grid[1][1] = centre
    positions = {"up": (0, 1), "down": (2, 1), "left": (1, 0), "right": (1, 2)}
    for name, content in cells.items():
        row, col = positions[name]
        grid[row][col] = content
    return grid


def make_agent(hunger=0.0, thirst=0.0, fatigue=0.0, local_view=None, goals=(), pos=(5, 5), step=0, others=()):
    return SimpleNamespace(
        internal_state=SimpleNamespace(hunger=hunger, thirst=thirst, fatigue=fatigue),
        perception={"local_grid_view": local_view or view(),
                    "other_agents_in_sight": [{"pos_x": r, "pos_y": c} for r, c in others]},
        current_time_step=step,
        active_goals=list(goals), pos_x=pos[0], pos_y=pos[1],
        internal_monologue="", attention_focus=None, _MOVE_OFFSETS=MOVE_OFFSETS)


class FixedSpecialist(Specialist):
    """Always bids the same proposal."""

    def __init__(self, proposal):
        self.name = proposal.source
        self.proposal = proposal

    def propose(self, agent):
        return self.proposal


def bid(source, salience, action="rest", **content):
    return Proposal(source, action, salience, content)


class NeedSpecialistTest(unittest.TestCase):
    def test_silent_below_threshold(self):
        self.assertIsNone(NeedSpecialist("hunger").propose(make_agent(hunger=0.29)))

    def test_salience_grows_faster_than_level(self):
        low = NeedSpecialist("thirst").propose(make_agent(thirst=0.4)).salience
        high = NeedSpecialist("thirst").propose(make_agent(thirst=0.8)).salience
        self.assertAlmostEqual(low, 0.16)
        self.assertAlmostEqual(high, 0.64)

    def test_critical_need_gets_bonus(self):
        self.assertAlmostEqual(NeedSpecialist("hunger").propose(make_agent(hunger=0.9)).salience, 0.81 + 0.5)

    def test_critical_needs_are_ranked_by_level_by_default(self):
        agent = make_agent(hunger=0.9, thirst=0.99)
        self.assertGreater(NeedSpecialist("thirst").propose(agent).salience, NeedSpecialist("hunger").propose(agent).salience)

    def test_fixed_priority_ranks_critical_needs_hunger_first(self):
        agent = make_agent(hunger=0.9, thirst=0.99, fatigue=1.0)
        with mock.patch("core.workspace.specialists.FIXED_NEED_PRIORITY", True):
            bids = {n: NeedSpecialist(n).propose(agent).salience for n in ("hunger", "thirst", "fatigue")}
        self.assertGreater(bids["hunger"], bids["thirst"])
        self.assertGreater(bids["thirst"], bids["fatigue"])

    def test_fixed_priority_leaves_non_critical_needs_alone(self):
        with mock.patch("core.workspace.specialists.FIXED_NEED_PRIORITY", True):
            self.assertAlmostEqual(NeedSpecialist("hunger").propose(make_agent(hunger=0.5)).salience, 0.25)

    def test_consume_here_follows_the_perceived_cell(self):
        on_water = make_agent(thirst=0.5, local_view=view(centre="water"))
        on_food = make_agent(thirst=0.5, local_view=view(centre="food"))
        self.assertTrue(NeedSpecialist("thirst").propose(on_water).content["consume_here"])
        self.assertFalse(NeedSpecialist("thirst").propose(on_food).content["consume_here"])

    def test_fatigue_never_consumes_a_resource(self):
        proposal = NeedSpecialist("fatigue").propose(make_agent(fatigue=0.5, local_view=view(centre="water")))
        self.assertEqual(proposal.action, "rest")
        self.assertFalse(proposal.content["consume_here"])


class GoalSpecialistTest(unittest.TestCase):
    GOAL = {"id": "g", "type": "reach_location", "priority": 0.9, "completed": False,
            "target_x": 5, "target_y": 9}

    def goal(self, **changes):
        return dict(self.GOAL, **changes)

    def test_steps_toward_the_target(self):
        proposal = GoalSpecialist().propose(make_agent(goals=[self.goal()], pos=(5, 5)))
        self.assertEqual(proposal.action, "move_right")
        self.assertAlmostEqual(proposal.salience, 0.45)

    def test_closes_the_larger_gap_first(self):
        proposal = GoalSpecialist().propose(make_agent(goals=[self.goal(target_x=9, target_y=6)], pos=(5, 5)))
        self.assertEqual(proposal.action, "move_down")

    def test_goes_around_a_blocked_axis(self):
        agent = make_agent(goals=[self.goal(target_x=7, target_y=6)], pos=(5, 5), local_view=view(down="obstacle"))
        self.assertEqual(GoalSpecialist().propose(agent).action, "move_right")

    def test_pushes_when_every_useful_move_is_blocked(self):
        agent = make_agent(goals=[self.goal()], pos=(5, 5), local_view=view(right="obstacle"))
        self.assertEqual(GoalSpecialist().propose(agent).action, "move_object")

    def test_completes_the_goal_on_arrival(self):
        goal = self.goal()
        self.assertIsNone(GoalSpecialist().propose(make_agent(goals=[goal], pos=(5, 9))))
        self.assertTrue(goal["completed"])


class GlobalWorkspaceTest(unittest.TestCase):
    def workspace(self, bids, **kwargs):
        return GlobalWorkspace([FixedSpecialist(b) for b in bids], **kwargs)

    def test_most_salient_bid_wins(self):
        ws = self.workspace([bid("a", 0.2, "rest"), bid("b", 0.7, "seek_food")])
        self.assertEqual(ws.decide(make_agent()), "seek_food")

    def test_capacity_limits_what_is_admitted(self):
        bids = [bid("a", 0.2), bid("b", 0.7), bid("c", 0.5)]
        for capacity in (1, 2, 3):
            ws = self.workspace(bids, capacity=capacity)
            ws.decide(make_agent())
            self.assertEqual([p.source for p in ws.contents], ["b", "c", "a"][:capacity])

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            GlobalWorkspace([], capacity=0)

    def test_persistence_keeps_the_previous_topic_in_the_workspace(self):
        ws = self.workspace([bid("a", 0.50, "rest"), bid("b", 0.45, "seek_food")], persistence=0.15)
        ws._last_winner_key = ("b", None)
        self.assertEqual(ws.decide(make_agent()), "seek_food")

    def test_without_persistence_the_stronger_bid_wins(self):
        ws = self.workspace([bid("a", 0.50, "rest"), bid("b", 0.45, "seek_food")], persistence=0.0)
        ws._last_winner_key = ("b", None)
        self.assertEqual(ws.decide(make_agent()), "rest")

    def test_persistence_does_not_override_a_much_stronger_bid(self):
        ws = self.workspace([bid("a", 0.90, "rest"), bid("b", 0.45, "seek_food")], persistence=0.15)
        ws._last_winner_key = ("b", None)
        self.assertEqual(ws.decide(make_agent()), "rest")

    def test_extra_capacity_enables_opportunistic_consumption(self):
        bids = [bid("hunger", 0.8, "seek_food", need="hunger", consume_here=False),
                bid("thirst", 0.4, "drink_water", need="thirst", consume_here=True)]
        narrow = self.workspace(bids, capacity=1)
        wide = self.workspace(bids, capacity=2)
        self.assertEqual(narrow.decide(make_agent()), "seek_food")
        self.assertEqual(wide.decide(make_agent()), "drink_water")
        self.assertEqual((narrow.opportunistic_actions, wide.opportunistic_actions), (0, 1))

    def test_winner_that_can_consume_here_is_not_displaced(self):
        bids = [bid("hunger", 0.8, "seek_food", need="hunger", consume_here=True),
                bid("thirst", 0.4, "drink_water", need="thirst", consume_here=True)]
        self.assertEqual(self.workspace(bids, capacity=2).decide(make_agent()), "seek_food")

    def test_broadcast_steers_attention_when_feedback_is_on(self):
        for need, focus in (("hunger", "food"), ("thirst", "water"), ("fatigue", "rest")):
            agent = make_agent()
            self.workspace([bid(f"need:{need}", 0.9, need=need)], feedback=True).decide(agent)
            self.assertEqual(agent.attention_focus, focus)
        agent = make_agent()
        self.workspace([bid("goal", 0.9, goal="g")], feedback=True).decide(agent)
        self.assertEqual(agent.attention_focus, "location_target")

    def test_attention_is_untouched_when_feedback_is_off(self):
        agent = make_agent()
        self.workspace([bid("need:hunger", 0.9, need="hunger")], feedback=False).decide(agent)
        self.assertIsNone(agent.attention_focus)

    def test_winners_are_counted(self):
        ws = self.workspace([bid("a", 0.9), bid("b", 0.1)])
        for _ in range(3):
            ws.decide(make_agent())
        self.assertEqual(dict(ws.winner_counts), {"a": 3})


class WorkspaceMemoryTest(unittest.TestCase):
    def full_of(self, memory, kind, count, seen=0):
        from core.workspace import Sighting
        memory.sightings = [Sighting(kind, (0, i), seen) for i in range(count)]

    def test_sightings_get_absolute_locations(self):
        memory = WorkspaceMemory(slots=9)
        memory.observe(make_agent(pos=(5, 5), local_view=view(up="food", right="water", left="obstacle"), others=[(6, 6)]))
        found = {(s.kind, s.location) for s in memory.sightings}
        self.assertEqual(found, {("food", (4, 5)), ("water", (5, 6)), ("obstacle", (5, 4)), ("agent", (6, 6))})

    def test_fills_up_to_capacity(self):
        memory = WorkspaceMemory(slots=2)
        memory.observe(make_agent(local_view=view(up="food", down="food", left="food")))
        self.assertEqual(len(memory.sightings), 2)

    def test_seeing_something_again_refreshes_it(self):
        memory = WorkspaceMemory(slots=5)
        memory.observe(make_agent(local_view=view(up="food"), step=1))
        memory.observe(make_agent(local_view=view(up="food"), step=9))
        self.assertEqual([(s.location, s.seen) for s in memory.sightings], [((4, 5), 9)])

    def test_fifo_evicts_the_oldest_sighting(self):
        memory = WorkspaceMemory(slots=2, policy="fifo")
        memory.observe(make_agent(pos=(1, 1), local_view=view(up="food"), step=1))     # (0, 1)
        memory.observe(make_agent(pos=(3, 3), local_view=view(up="food"), step=2))     # (2, 3)
        memory.observe(make_agent(pos=(7, 7), local_view=view(up="water"), step=3))    # (6, 7) evicts (0, 1)
        self.assertEqual(sorted(s.location for s in memory.sightings), [(2, 3), (6, 7)])

    def test_salience_admits_what_the_current_need_values(self):
        memory = WorkspaceMemory(slots=2)
        self.full_of(memory, "obstacle", 2)
        memory.observe(make_agent(hunger=0.9, local_view=view(up="food")))
        self.assertIn("food", [s.kind for s in memory.sightings])

    def test_salience_rejects_a_less_valuable_newcomer(self):
        memory = WorkspaceMemory(slots=2)
        self.full_of(memory, "food", 2)
        memory.observe(make_agent(hunger=0.9, others=[(4, 4)]))
        self.assertEqual([s.kind for s in memory.sightings], ["food", "food"])

    def test_fifo_does_not_discriminate(self):
        memory = WorkspaceMemory(slots=2, policy="fifo")
        self.full_of(memory, "food", 2)
        memory.observe(make_agent(hunger=0.9, others=[(4, 4)], step=5))
        self.assertIn("agent", [s.kind for s in memory.sightings])

    def test_old_sightings_are_worth_less(self):
        from core.workspace import Sighting
        memory = WorkspaceMemory(slots=1)
        state = SimpleNamespace(hunger=0.5, thirst=0.0)
        self.assertGreater(memory.worth(Sighting("food", (0, 0), 90), state, 100),
                           memory.worth(Sighting("food", (0, 0), 0), state, 100))

    def test_recall_returns_the_nearest_of_the_requested_kind(self):
        from core.workspace import Sighting
        memory = WorkspaceMemory(slots=5)
        memory.sightings = [Sighting("food", (5, 9), 0), Sighting("food", (7, 5), 0), Sighting("water", (5, 6), 0)]
        agent = make_agent(pos=(5, 5), local_view=view())
        self.assertEqual(memory.recall("food", agent), (7, 5))
        self.assertIsNone(memory.recall("obstacle", agent))

    def test_recall_forgets_what_the_view_contradicts(self):
        from core.workspace import Sighting
        memory = WorkspaceMemory(slots=5)
        memory.sightings = [Sighting("food", (5, 6), 0)]
        self.assertIsNone(memory.recall("food", make_agent(pos=(5, 5), local_view=view(right="empty"))))
        self.assertEqual(memory.sightings, [])

    def test_recall_keeps_a_sighting_that_is_merely_unseen(self):
        from core.workspace import Sighting
        memory = WorkspaceMemory(slots=5)
        memory.sightings = [Sighting("food", (5, 6), 0)]
        self.assertEqual(memory.recall("food", make_agent(pos=(5, 5), local_view=view(right="unknown"))), (5, 6))

    def test_relevance_policy_keeps_resources_over_distractors(self):
        memory = WorkspaceMemory(slots=2, policy="relevance")
        self.full_of(memory, "food", 2)
        memory.observe(make_agent(hunger=0.0, local_view=view(up="obstacle", down="obstacle"), others=[(4, 4)]))
        self.assertEqual([s.kind for s in memory.sightings], ["food", "food"])

    def test_relevance_policy_ignores_the_current_need_that_salience_follows(self):
        thirsty = dict(thirst=0.9, hunger=0.0, local_view=view(up="water"))
        relevance, salience = WorkspaceMemory(slots=2, policy="relevance"), WorkspaceMemory(slots=2)
        for memory in (relevance, salience):
            self.full_of(memory, "food", 2)
            memory.observe(make_agent(**thirsty))
        self.assertEqual([s.kind for s in relevance.sightings], ["food", "food"])   # equal worth: nothing displaced
        self.assertIn("water", [s.kind for s in salience.sightings])               # thirst makes water worth more

    def test_relevance_policy_replaces_an_older_resource_with_a_newer_one(self):
        memory = WorkspaceMemory(slots=1, policy="relevance")
        memory.observe(make_agent(pos=(1, 1), local_view=view(up="food"), step=0))
        memory.observe(make_agent(pos=(7, 7), local_view=view(up="food"), step=50))
        self.assertEqual([s.location for s in memory.sightings], [(6, 7)])

    def test_recent_recall_prefers_the_latest_sighting_over_the_nearest(self):
        from core.workspace import Sighting
        sightings = [Sighting("food", (5, 6), 1), Sighting("food", (9, 9), 8)]
        agent = make_agent(pos=(5, 5), local_view=view(right="food"))   # (5, 6) is visible and still there
        nearest, recent = WorkspaceMemory(5, recall="nearest"), WorkspaceMemory(5, recall="recent")
        nearest.sightings, recent.sightings = list(sightings), list(sightings)
        self.assertEqual(nearest.recall("food", agent), (5, 6))
        self.assertEqual(recent.recall("food", agent), (9, 9))

    def test_forget_drops_only_the_given_sighting(self):
        from core.workspace import Sighting
        memory = WorkspaceMemory(slots=5)
        memory.sightings = [Sighting("food", (1, 1), 0), Sighting("food", (2, 2), 0), Sighting("water", (1, 1), 0)]
        memory.forget("food", (1, 1))
        self.assertEqual([(s.kind, s.location) for s in memory.sightings], [("food", (2, 2)), ("water", (1, 1))])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkspaceMemory(slots=3, recall="random")
        with self.assertRaises(ValueError):
            WorkspaceMemory(slots=0)
        with self.assertRaises(ValueError):
            WorkspaceMemory(slots=3, policy="random")

    def test_workspace_feeds_percepts_into_its_memory(self):
        memory = WorkspaceMemory(slots=3)
        ws = GlobalWorkspace([FixedSpecialist(bid("a", 0.5))], memory=memory)
        ws.decide(make_agent(local_view=view(up="water")))
        self.assertEqual([s.kind for s in memory.sightings], ["water"])


class AgentIntegrationTest(unittest.TestCase):
    def test_consumption_levels_record_the_need_at_the_moment_of_eating(self):
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            from agent.base_agent import Agent
            from core.mood.basic_mood import BasicMoodStrategy
            from environment.world import World
            world = World()
            for row in world.grid:
                row[:] = ["empty"] * len(row)
            agent = Agent(name="T", mood_strategy=BasicMoodStrategy())
            world.add_agent(agent, start_pos=(2, 2))
            world.grid[2][2] = "food"
            agent.internal_state.hunger = 0.6
            agent._act_default("seek_food")
        self.assertEqual(agent.consumption_levels["hunger"], [0.6])
        self.assertEqual(agent.consumption_levels["thirst"], [])

    def test_workspace_is_off_by_default(self):
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            from agent.base_agent import Agent
            from core.mood.basic_mood import BasicMoodStrategy
            agent = Agent(name="T", mood_strategy=BasicMoodStrategy())
        self.assertIsNone(agent.workspace)

    def test_standalone_sighting_memory_works_without_a_workspace(self):
        import io
        import contextlib
        from unittest import mock
        with contextlib.redirect_stdout(io.StringIO()):
            import agent.base_agent as base_agent
            from core.mood.basic_mood import BasicMoodStrategy
            from environment.world import World
            with mock.patch.object(base_agent, "SIGHTING_MEMORY_SLOTS", 3), \
                    mock.patch.object(base_agent, "SIGHTING_MEMORY_POLICY", "relevance"):
                world = World()
                agent = base_agent.Agent(name="T", mood_strategy=BasicMoodStrategy(), perception_accuracy=1.0)
            world.add_agent(agent, start_pos=(4, 4))
        self.assertIsNone(agent.workspace)
        self.assertEqual((agent.sighting_memory.slots, agent.sighting_memory.policy), (3, "relevance"))
        agent.sighting_memory.sightings = [__import__("core.workspace", fromlist=["Sighting"]).Sighting("water", (9, 9), 0)]
        self.assertEqual(agent._recall_resource_location("water"), (9, 9))   # recall reads the standalone memory

    def test_search_gives_up_a_target_whose_approach_is_blocked(self):
        """Regression: a remembered target behind an obstacle trapped the agent in an endless back-and-forth."""
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            from agent.base_agent import Agent
            from core.mood.basic_mood import BasicMoodStrategy
            from environment.world import World
            world = World()
            for row in world.grid:
                row[:] = ["empty"] * len(row)
            agent = Agent(name="T", mood_strategy=BasicMoodStrategy(), perception_accuracy=1.0)
            world.add_agent(agent, start_pos=(3, 5))
        world.grid[3][4] = "obstacle"
        world.grid[3][3] = "water"
        agent.working_memory_buffer.append({"type": "perceived_water", "location": (3, 3), "time": 0})
        with contextlib.redirect_stdout(io.StringIO()):
            agent.sense()
        action = agent._search_resource_action("drink_water")
        self.assertNotEqual(action, "move_left")                      # that cell is an obstacle
        self.assertNotIn((3, 3), [item["location"] for item in agent.working_memory_buffer
                                  if item["type"] == "perceived_water"])


if __name__ == "__main__":
    unittest.main()
