"""Unit tests for the global workspace. Run from the repository root: python -m unittest discover tests"""

import unittest
from types import SimpleNamespace

from core.workspace import GlobalWorkspace, GoalSpecialist, NeedSpecialist, Proposal, Specialist

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


def make_agent(hunger=0.0, thirst=0.0, fatigue=0.0, local_view=None, goals=(), pos=(5, 5)):
    return SimpleNamespace(
        internal_state=SimpleNamespace(hunger=hunger, thirst=thirst, fatigue=fatigue),
        perception={"local_grid_view": local_view or view()},
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


class AgentIntegrationTest(unittest.TestCase):
    def test_workspace_is_off_by_default(self):
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            from agent.base_agent import Agent
            from core.mood.basic_mood import BasicMoodStrategy
            agent = Agent(name="T", mood_strategy=BasicMoodStrategy())
        self.assertIsNone(agent.workspace)


if __name__ == "__main__":
    unittest.main()
