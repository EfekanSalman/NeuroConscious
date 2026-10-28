from typing import Iterable

from config.constants import DRIVE_EXPONENT_N, DRIVE_EXPONENT_M


def drive(levels: Iterable[float], n: float = None, m: float = None) -> float:
    """
    Homeostatic drive of Keramati & Gutkin (2014): D(H) = (sum_i |h*_i - h_i|^n)^(1/m).

    Every need is a deficit whose set point is 0 (fully satisfied), so |h* - h| is the level itself.
    With n > 1 the drive grows convexly: two moderate deficits cost less than one severe deficit,
    which is what makes it worth acting before a need becomes critical.

    Args:
        levels: Current levels of the regulated needs, each in [0, 1].
        n (float, optional): Deviation exponent. Defaults to DRIVE_EXPONENT_N.
        m (float, optional): Root exponent. Defaults to DRIVE_EXPONENT_M.

    Returns:
        float: 0 when every need is satisfied, larger the further the agent is from its set points.
    """
    n = DRIVE_EXPONENT_N if n is None else n
    m = DRIVE_EXPONENT_M if m is None else m
    return sum(abs(level) ** n for level in levels) ** (1.0 / m)


def drive_of_state(state, n: float = None, m: float = None) -> float:
    """Drive of an InternalState or its snapshot() dictionary (hunger, thirst, fatigue)."""
    get = state.get if isinstance(state, dict) else (lambda key: getattr(state, key))
    return drive([get("hunger"), get("thirst"), get("fatigue")], n, m)
