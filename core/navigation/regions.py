"""The two halves of the grid that the beacon regime switches between."""


def half_of(col: int, size: int) -> str:
    """'left' for columns in the first half of a grid of `size` columns, otherwise 'right'."""
    return "left" if col < size / 2 else "right"
