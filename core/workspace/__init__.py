"""Global Workspace Theory components: specialists bid, a limited-capacity workspace broadcasts."""

from core.workspace.proposal import Proposal
from core.workspace.specialists import GoalSpecialist, HabitSpecialist, NeedSpecialist, Specialist
from core.workspace.workspace import GlobalWorkspace

__all__ = ["Proposal", "Specialist", "NeedSpecialist", "GoalSpecialist", "HabitSpecialist", "GlobalWorkspace"]
