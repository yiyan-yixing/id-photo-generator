"""Output presets: the standard Chinese ID-photo sizes and background colours."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    key: str
    label: str
    w: int                 # output pixels
    h: int
    mm: tuple[int, int] | None   # physical print size, when the spec defines one
    dpi: int

    @property
    def aspect(self) -> float:
        """Height / width. Drives the crop box, so it must be exact."""
        return self.h / self.w

    @property
    def margin(self) -> float:
        """crop_w / crop_h. The largest of these is the widest frame."""
        return self.w / self.h


SIZES: dict[str, Size] = {
    "xueji":    Size("xueji",    "学籍照",      480, 640, None,      300),
    "one_inch": Size("one_inch", "一寸",        295, 413, (25, 35),  300),
    "two_inch": Size("two_inch", "二寸",        413, 579, (35, 49),  300),
    "passport": Size("passport", "护照/大二寸", 390, 567, (33, 48),  300),
}

# Background colours. The spec is NOT unified: the light "standard sky blue" is
# what most studios use, but civil-service and exam portals commonly require the
# deeper 考公蓝 instead, and mixing them up is a routine rejection. Both are
# offered, and they are visibly different -- publishing the exact RGB matters
# more than picking a "nicer" blue.
COLORS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "white":  ("白底", (255, 255, 255)),
    "blue":   ("蓝底", (67, 142, 219)),      # #438EDB 标准天蓝
    "navy":   ("深蓝（考公）", (0, 102, 204)),  # #0066CC
    "red":    ("红底", (255, 0, 0)),
}

# What the page pre-selects. Kept narrow on purpose: one size and one colour is
# the common case, and the result is always available on demand for the rest.
DEFAULT_SIZES = ["one_inch"]
DEFAULT_COLORS = ["white"]
