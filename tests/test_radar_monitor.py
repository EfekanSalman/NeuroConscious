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
        radar_verified=lambda hit, strength=None: log.append(hit), log=log, cue_allows=lambda cell: True)


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


class PerceptMonitorTest(unittest.TestCase):
    def admit(self, mode, report, minimum=1.0):
        with mock.patch.object(base_agent, "RADAR_MONITOR", mode), mock.patch.object(base_agent, "RADAR_STRENGTH_MIN", minimum):
            return Agent.admit_radar(None, report)

    def test_reports_carry_a_strength_only_with_the_signal_on(self):
        with mock.patch.object(base_agent, "GHOST_STRENGTH_SEP", 0.0):
            self.assertEqual(Agent._radar_report("food", (1, 2), True), ("food", (1, 2)))
        with mock.patch.object(base_agent, "GHOST_STRENGTH_SEP", 2.0):
            report = Agent._radar_report("food", (1, 2), False)
        self.assertEqual(len(report), 4)
        self.assertFalse(report[3])

    def test_oracle_admits_exactly_the_real_reports(self):
        self.assertTrue(self.admit("percept_oracle", ("food", (1, 1), -3.0, True)))
        self.assertFalse(self.admit("percept_oracle", ("food", (1, 1), 5.0, False)))

    def test_threshold_admits_strong_reports_only(self):
        self.assertTrue(self.admit("percept_threshold", ("food", (1, 1), 1.2, False)))
        self.assertFalse(self.admit("percept_threshold", ("food", (1, 1), 0.2, True)))

    def test_other_modes_and_plain_reports_are_admitted(self):
        self.assertTrue(self.admit("none", ("food", (1, 1), -9.0, False)))
        self.assertTrue(self.admit("percept_threshold", ("food", (1, 1))))

    def test_a_sticky_ghost_survives_a_glance_but_not_a_visit(self):
        agent, memory = view_agent(), WorkspaceMemory(5)
        agent.radar_sticky = True
        memory.sightings.append(Sighting("food", (5, 6), 5, "radar"))
        self.assertEqual(memory.recall("food", agent), (5, 6))
        self.assertEqual(agent.log, [])
        memory.sightings = [Sighting("food", (5, 5), 5, "radar")]  # standing on it, and nothing is there
        self.assertIsNone(memory.recall("food", agent))
        self.assertEqual(agent.log, [False])


class LearnedPerceptTest(unittest.TestCase):
    def stub(self):
        agent = SimpleNamespace(radar_stats={"hits": 0, "ghosts": 0}, radar_labels=[], radar_tau=float("-inf"),
                                radar_reliability=0.5, radar_admitted=[0, 0])
        agent._radar_label = lambda strength, hit: Agent._radar_label(agent, strength, hit)
        return agent

    def test_the_threshold_separates_hits_from_ghosts_once_there_are_enough_labels(self):
        agent = self.stub()
        with mock.patch.object(base_agent, "RADAR_MIN_LABELS", 6), mock.patch.object(base_agent, "RADAR_LABEL_WINDOW", 60):
            for strength, hit in [(2.5, True), (-0.5, False), (3.0, True), (0.2, False), (2.0, True), (0.8, False)]:
                Agent.radar_verified(agent, hit, strength)
        self.assertGreater(agent.radar_tau, 0.8)
        self.assertLessEqual(agent.radar_tau, 2.0)

    def test_admits_everything_before_enough_labels_and_counts_offers(self):
        agent = self.stub()
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_learned"), mock.patch.object(base_agent, "RADAR_PROBE", 0.0):
            self.assertTrue(Agent.admit_radar(agent, ("food", (1, 1), -5.0, False)))
            agent.radar_tau = 1.0
            self.assertFalse(Agent.admit_radar(agent, ("food", (1, 1), 0.5, True)))
        self.assertEqual(agent.radar_admitted, [1, 2])

    def test_random_admission_ignores_the_strength(self):
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_random"), mock.patch.object(base_agent, "RADAR_RANDOM_P", 1.0):
            self.assertTrue(Agent.admit_radar(None, ("food", (1, 1), -9.0, True)))
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_random"), mock.patch.object(base_agent, "RADAR_RANDOM_P", 0.0):
            self.assertFalse(Agent.admit_radar(None, ("food", (1, 1), 9.0, True)))

    def test_a_cheaper_ghost_lowers_the_fitted_threshold(self):
        pairs = [(2.5, True), (1.2, True), (1.0, False), (0.4, False), (3.0, True), (0.0, False), (2.0, True), (1.4, False)]
        taus = []
        for weight in (1.0, 0.1):
            agent = self.stub()
            with mock.patch.object(base_agent, "RADAR_MIN_LABELS", 8), mock.patch.object(base_agent, "RADAR_GHOST_WEIGHT", weight):
                for strength, hit in pairs:
                    Agent._radar_label(agent, strength, hit)
            taus.append(agent.radar_tau)
        self.assertLess(taus[1], taus[0])

    def test_free_labels_come_from_the_offered_report_itself(self):
        agent = self.stub()
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_learned"), mock.patch.object(base_agent, "RADAR_LABELS", "free"), \
                mock.patch.object(base_agent, "RADAR_PROBE", 0.0):
            Agent.admit_radar(agent, ("food", (1, 1), 2.0, True))
        self.assertEqual(agent.radar_labels, [(2.0, True)])


class AuditTest(unittest.TestCase):
    def stub(self, pending):
        agent = SimpleNamespace(radar_stats={"hits": 0, "ghosts": 0, "audits": 0}, radar_labels=[], radar_tau=float("-inf"),
                                radar_admitted=[0, 0], radar_pending=list(pending))
        agent._radar_label = lambda strength, hit: Agent._radar_label(agent, strength, hit)
        return agent

    def audit(self, agent, probability, free=False):
        with mock.patch.object(base_agent, "RADAR_AUDIT_P", probability), mock.patch.object(base_agent, "RADAR_AUDIT_FREE", free):
            return Agent._radar_audit(agent, "move_up")

    def test_an_audit_costs_the_step_and_labels_one_rejected_report(self):
        agent = self.stub([(0.5, True)])
        self.assertEqual(self.audit(agent, 1.0), "explore")
        self.assertEqual(agent.radar_labels, [(0.5, True)])
        self.assertEqual(agent.radar_stats["audits"], 1)
        self.assertEqual(agent.radar_pending, [])

    def test_a_free_audit_keeps_the_action(self):
        self.assertEqual(self.audit(self.stub([(0.5, False)]), 1.0, free=True), "move_up")

    def test_no_audit_without_a_pending_report_or_at_zero_probability(self):
        self.assertEqual(self.audit(self.stub([]), 1.0), "move_up")
        agent = self.stub([(0.5, True)])
        self.assertEqual(self.audit(agent, 0.0), "move_up")
        self.assertEqual(agent.radar_stats["audits"], 0)

    def test_rejected_reports_become_pending_only_in_audit_mode(self):
        for labels, expected in (("audit", 1), ("audit_only", 1), ("verified", 0)):
            agent = self.stub([])
            agent.radar_tau = 5.0
            with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_learned"), mock.patch.object(base_agent, "RADAR_LABELS", labels), \
                    mock.patch.object(base_agent, "RADAR_PROBE", 0.0):
                Agent.admit_radar(agent, ("food", (1, 1), 0.5, True))
            self.assertEqual(len(agent.radar_pending), expected)


class AuditKeepsVerifiedLabelsTest(unittest.TestCase):
    def test_verified_outcomes_still_label_in_audit_mode_but_not_in_free_mode(self):
        for labels, expected in (("audit", 1), ("verified", 1), ("free", 0), ("audit_only", 0)):
            agent = SimpleNamespace(radar_stats={"hits": 0, "ghosts": 0}, radar_labels=[], radar_tau=float("-inf"), radar_reliability=0.5)
            agent._radar_label = lambda strength, hit, a=agent: Agent._radar_label(a, strength, hit)
            with mock.patch.object(base_agent, "RADAR_LABELS", labels):
                Agent.radar_verified(agent, True, 1.5)
            self.assertEqual(len(agent.radar_labels), expected, labels)


class AuditScopeTest(unittest.TestCase):
    def offer(self, scope, strength):
        agent = SimpleNamespace(radar_stats={"hits": 0, "ghosts": 0, "audits": 0}, radar_labels=[], radar_tau=1.0,
                                radar_admitted=[0, 0], radar_pending=[])
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_learned"), mock.patch.object(base_agent, "RADAR_LABELS", "audit"), \
                mock.patch.object(base_agent, "RADAR_AUDIT_SCOPE", scope), mock.patch.object(base_agent, "RADAR_PROBE", 0.0):
            admitted = Agent.admit_radar(agent, ("food", (1, 1), strength, False))
        return admitted, len(agent.radar_pending)

    def test_rejected_scope_remembers_only_what_was_turned_away(self):
        self.assertEqual(self.offer("rejected", 0.2), (False, 1))
        self.assertEqual(self.offer("rejected", 3.0), (True, 0))

    def test_all_scope_also_remembers_admitted_reports(self):
        self.assertEqual(self.offer("all", 0.2), (False, 1))
        self.assertEqual(self.offer("all", 3.0), (True, 1))


class FreeLabelLeakTest(unittest.TestCase):
    def decide(self, labels):
        agent = SimpleNamespace(radar_stats={"hits": 0, "ghosts": 0}, radar_labels=[(2.0, True)] * 10, radar_tau=1.0,
                                radar_admitted=[0, 0], radar_pending=[])
        agent._radar_label = lambda strength, hit: Agent._radar_label(agent, strength, hit)
        with mock.patch.object(base_agent, "RADAR_MONITOR", "percept_learned"), mock.patch.object(base_agent, "RADAR_LABELS", labels), \
                mock.patch.object(base_agent, "RADAR_PROBE", 0.0), mock.patch.object(base_agent, "RADAR_GHOST_WEIGHT", 30):
            return Agent.admit_radar(agent, ("food", (1, 1), 1.5, False))

    def test_free_labels_use_the_truth_of_the_report_being_judged_free_after_does_not(self):
        self.assertFalse(self.decide("free"))        # the ghost's own label raised the threshold above it
        self.assertTrue(self.decide("free_after"))   # judged with the old threshold


class AccumulatedMonitorTest(unittest.TestCase):
    def judge(self, monitor, reports, pooled=False, **consts):
        agent = SimpleNamespace(radar_recur={}, radar_admitted=[0, 0], current_time_step=0)
        agent._admit_accumulated = lambda report: Agent._admit_accumulated(agent, report)
        out = []
        patches = [mock.patch.object(base_agent, "RADAR_MONITOR", monitor), mock.patch.object(base_agent, "RADAR_RECUR_POOLED", pooled)]
        patches += [mock.patch.object(base_agent, name, value) for name, value in consts.items()]
        for patch in patches:
            patch.start()
        try:
            for step, cell, strength in reports:
                agent.current_time_step = step
                out.append(Agent.admit_radar(agent, ("food", cell, strength, True)))
        finally:
            for patch in patches:
                patch.stop()
        return out

    def test_recurrent_state_accumulates_over_repeated_reports_of_one_cell(self):
        got = self.judge("percept_recurrent", [(0, (1, 1), 1.5), (1, (1, 1), 1.5)], RADAR_RECUR_DECAY=1.0, RADAR_RECUR_TAU=2.5)
        self.assertEqual(got, [False, True])

    def test_state_leaks_with_time_and_is_per_cell(self):
        got = self.judge("percept_recurrent", [(0, (1, 1), 1.5), (30, (1, 1), 1.5), (31, (2, 2), 1.5)],
                         RADAR_RECUR_DECAY=0.5, RADAR_RECUR_TAU=2.5)
        self.assertEqual(got, [False, False, False])

    def test_pooled_state_mixes_cells(self):
        got = self.judge("percept_recurrent", [(0, (1, 1), 1.5), (1, (2, 2), 1.5)], pooled=True,
                         RADAR_RECUR_DECAY=1.0, RADAR_RECUR_TAU=2.5)
        self.assertEqual(got, [False, True])

    def test_window_sums_recent_reports_and_forgets_old_ones(self):
        got = self.judge("percept_window", [(0, (1, 1), 1.5), (5, (1, 1), 1.5), (40, (1, 1), 1.5)],
                         RADAR_RECUR_WINDOW=10, RADAR_RECUR_TAU=2.5)
        self.assertEqual(got, [False, True, False])

    def test_repeat_counts_reports_and_ignores_strength(self):
        got = self.judge("percept_repeat", [(0, (1, 1), -3.0), (2, (1, 1), -3.0)], RADAR_RECUR_WINDOW=10, RADAR_RECUR_COUNT=2)
        self.assertEqual(got, [False, True])
