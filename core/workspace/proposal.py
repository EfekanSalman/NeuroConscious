from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Proposal:
    """
    A specialist's bid for access to the global workspace.

    Attributes:
        source (str): Name of the specialist that made the bid.
        action (str): The action the specialist wants executed.
        salience (float): How strongly the bid competes for workspace access.
        content (Dict[str, Any]): What the bid represents, e.g. {"need": "thirst", "level": 0.9}.
    """
    source: str
    action: str
    salience: float
    content: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, Optional[str]]:
        """Identity of the *topic* of the bid, used to recognise the same content across steps."""
        return self.source, self.content.get("need") or self.content.get("goal")
