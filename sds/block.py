"""Block data structure for fixed-outline floorplanning and slack-driven shaping."""

from __future__ import annotations
import math
from typing import Optional, Tuple


class Block:
    """Represents a circuit block in floorplanning (hard or soft).

    Attributes:
        id: Unique block identifier (1 to n, 0 is source dummy, n+1 is sink dummy).
        name: Human-readable name.
        area: Block area A_i (w_i * h_i = A_i).
        is_hard: True if aspect ratio and dimensions are fixed, False if soft.
        w_min: Minimum width bound.
        w_max: Maximum width bound.
        h_min: Minimum height bound.
        h_max: Maximum height bound.
        w: Current width.
        h: Current height.
        x: Current x-coordinate of bottom-left corner.
        y: Current y-coordinate of bottom-left corner.
        slack_h: Computed horizontal slack s_i^h.
        slack_v: Computed vertical slack s_i^v.
        subset: Current classified subset ('I', 'II', ..., 'VII').
    """

    def __init__(
        self,
        id: int,
        name: str = "",
        area: float = 1.0,
        is_hard: bool = False,
        aspect_ratio_range: Tuple[float, float] = (1.0 / 3.0, 3.0),
        w_fixed: Optional[float] = None,
        h_fixed: Optional[float] = None,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        h_min: Optional[float] = None,
        h_max: Optional[float] = None,
    ):
        self.id = id
        self.name = name if name else f"B{id}"
        self.is_hard = is_hard

        if is_hard:
            if w_fixed is not None and h_fixed is not None:
                self.w = float(w_fixed)
                self.h = float(h_fixed)
                self.area = self.w * self.h
            elif area is not None:
                self.area = float(area)
                # default hard block is square if dimensions not explicitly given
                self.w = math.sqrt(self.area)
                self.h = math.sqrt(self.area)
            else:
                raise ValueError("Must specify area or (w_fixed, h_fixed) for hard block.")
            self.w_min = self.w_max = self.w
            self.h_min = self.h_max = self.h
            self.aspect_ratio_range = (self.h / self.w, self.h / self.w)
        else:
            self.area = float(area)
            r_min, r_max = aspect_ratio_range
            self.aspect_ratio_range = (float(r_min), float(r_max))

            # Aspect ratio r = h / w => w = sqrt(A / r), h = sqrt(A * r)
            # Min width corresponds to max aspect ratio (tallest, narrowest)
            # Max width corresponds to min aspect ratio (flattest, widest)
            computed_w_min = math.sqrt(self.area / r_max)
            computed_w_max = math.sqrt(self.area / r_min)
            computed_h_min = math.sqrt(self.area * r_min)
            computed_h_max = math.sqrt(self.area * r_max)

            self.w_min = float(w_min) if w_min is not None else computed_w_min
            self.w_max = float(w_max) if w_max is not None else computed_w_max
            self.h_min = float(h_min) if h_min is not None else computed_h_min
            self.h_max = float(h_max) if h_max is not None else computed_h_max

            # Initialize at minimal width as specified in Section 3.3 and Section 6
            self.w = self.w_min
            self.h = self.area / self.w

        self.x = 0.0
        self.y = 0.0
        self.slack_h = 0.0
        self.slack_v = 0.0
        self.subset = "I" if is_hard else "II"

    def set_width(self, new_w: float) -> None:
        """Update block width and recompute height to preserve area."""
        if self.is_hard:
            return
        clamped_w = max(self.w_min, min(self.w_max, float(new_w)))
        self.w = clamped_w
        self.h = self.area / clamped_w

    def set_height(self, new_h: float) -> None:
        """Update block height and recompute width to preserve area."""
        if self.is_hard:
            return
        clamped_h = max(self.h_min, min(self.h_max, float(new_h)))
        self.h = clamped_h
        self.w = self.area / clamped_h

    def reset_to_min_width(self) -> None:
        """Reset soft block to minimum width W_i^min (initial state in SDS)."""
        if not self.is_hard:
            self.w = self.w_min
            self.h = self.area / self.w

    def aspect_ratio(self) -> float:
        """Return aspect ratio h / w."""
        return self.h / self.w if self.w > 0 else 0.0

    def clone(self) -> Block:
        """Create a deep copy of the block."""
        b = Block(
            id=self.id,
            name=self.name,
            area=self.area,
            is_hard=self.is_hard,
            aspect_ratio_range=self.aspect_ratio_range,
            w_min=self.w_min,
            w_max=self.w_max,
            h_min=self.h_min,
            h_max=self.h_max,
        )
        b.w = self.w
        b.h = self.h
        b.x = self.x
        b.y = self.y
        b.slack_h = self.slack_h
        b.slack_v = self.slack_v
        b.subset = self.subset
        return b

    def __repr__(self) -> str:
        kind = "Hard" if self.is_hard else "Soft"
        return (
            f"Block({self.name}, id={self.id}, {kind}, A={self.area:.2f}, "
            f"w={self.w:.4f}, h={self.h:.4f}, x={self.x:.4f}, y={self.y:.4f}, "
            f"sh={self.slack_h:.4f}, sv={self.slack_v:.4f}, subset={self.subset})"
        )

