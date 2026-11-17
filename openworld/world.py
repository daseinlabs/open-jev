"""Random open-world map: terrain, a river, bridges and named landmarks (numpy only, no rendering).

Everything the game draws and the player collides with comes from ``World``, so later stages
(pathfinding, instruction labels, headless capture) can use the same map without the engine.
Coordinates: x east, z north, y up, metres; the map spans [0, SIZE) on x and z.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIZE = 160.0          # metres per side
CELL = 2.0            # heightmap spacing
WATER_Y = 0.0         # water surface
BANK = 4.0            # sloped bank beyond the river's edge
RAIL = 0.4            # walkers keep this far inside a bridge's railings
LANDMARKS = ("red house", "stone tower", "windmill", "big tree", "blue tent")
FOOTPRINT = {"red house": 5.0, "stone tower": 3.5, "windmill": 3.5, "big tree": 2.0, "blue tent": 3.0}


@dataclass
class Bridge:
    x0: float          # west end (centre line)
    x1: float          # east end
    z: float           # centre line
    y0: float          # deck height at the west end (= the bank there)
    y1: float          # deck height at the east end
    width: float = 3.0

    def deck_y(self, x: float) -> float:
        """Straight between the banks, with a slight arch so the middle clears the water."""
        t = float(np.clip((x - self.x0) / (self.x1 - self.x0), 0, 1))
        return self.y0 + (self.y1 - self.y0) * t + 0.8 * np.sin(np.pi * t)


@dataclass
class Landmark:
    name: str
    x: float
    z: float
    radius: float


@dataclass
class World:
    seed: int
    heights: np.ndarray                  # (n, n) terrain height at grid points
    river_x: np.ndarray                  # river centre x at each grid row (z = row * CELL)
    river_half: float                    # half-width of the water
    bridges: list[Bridge] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)
    trees: list[tuple[float, float, float]] = field(default_factory=list)  # x, z, scale

    # -------------------------------------------------------------- queries
    def terrain_y(self, x: float, z: float) -> float:
        """Bilinear terrain height (the river bed inside the river)."""
        n = self.heights.shape[0]
        gx, gz = np.clip(x / CELL, 0, n - 1.001), np.clip(z / CELL, 0, n - 1.001)
        i, j = int(gz), int(gx)
        fz, fx = gz - i, gx - j
        h = self.heights
        return float((h[i, j] * (1 - fx) + h[i, j + 1] * fx) * (1 - fz)
                     + (h[i + 1, j] * (1 - fx) + h[i + 1, j + 1] * fx) * fz)

    def river_dist(self, x: float, z: float) -> float:
        """Horizontal distance from the river's centre line (along x, the river runs north)."""
        rx = float(np.interp(z, np.arange(len(self.river_x)) * CELL, self.river_x))
        return abs(x - rx)

    def bridge_at(self, x: float, z: float) -> Bridge | None:
        for b in self.bridges:
            if b.x0 - 0.5 <= x <= b.x1 + 0.5 and abs(z - b.z) <= b.width / 2 - RAIL:
                return b
        return None

    def ground_y(self, x: float, z: float) -> float:
        """Height the player stands on: bridge deck or terrain."""
        b = self.bridge_at(x, z)
        return max(b.deck_y(x), self.terrain_y(x, z)) if b else self.terrain_y(x, z)

    def walkable(self, x: float, z: float) -> bool:
        if not (1.0 <= x <= SIZE - 1.0 and 1.0 <= z <= SIZE - 1.0):
            return False
        if self.bridge_at(x, z) is None:
            if self.terrain_y(x, z) < WATER_Y + 0.15:
                return False  # in the water
            if any(b.x0 <= x <= b.x1 and abs(z - b.z) <= b.width / 2 + RAIL for b in self.bridges):
                return False  # a railing, also where the deck crosses the bank (keep RAIL outside it too)
        if any(np.hypot(x - tx, z - tz) < 0.5 * k for tx, tz, k in self.trees):
            return False  # tree trunk
        return all(np.hypot(x - l.x, z - l.z) > l.radius for l in self.landmarks)

    def nearest_landmark(self, x: float, z: float) -> tuple[Landmark, float]:
        d = [(np.hypot(x - l.x, z - l.z), l) for l in self.landmarks]
        dist, lm = min(d, key=lambda t: t[0])
        return lm, float(dist)

    def spawn(self) -> tuple[float, float]:
        """A walkable point west of the river, away from landmarks."""
        rng = np.random.default_rng(self.seed + 1)
        while True:
            z = rng.uniform(20, SIZE - 20)
            x = rng.uniform(10, float(np.interp(z, np.arange(len(self.river_x)) * CELL, self.river_x)) - 15)
            if self.walkable(x, z) and all(np.hypot(x - l.x, z - l.z) > l.radius + 6 for l in self.landmarks):
                return float(x), float(z)


def _value_noise(rng, n, cells):
    """Smooth noise in [0, 1] on an (n, n) grid from a (cells+1)^2 random lattice."""
    lat = rng.random((cells + 1, cells + 1))
    t = np.linspace(0, cells, n, endpoint=False)
    i, f = t.astype(int), t - t.astype(int)
    f = f * f * (3 - 2 * f)
    a = lat[np.ix_(i, i)] * (1 - f)[None, :] + lat[np.ix_(i, i + 1)] * f[None, :]
    b = lat[np.ix_(i + 1, i)] * (1 - f)[None, :] + lat[np.ix_(i + 1, i + 1)] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def generate(seed: int) -> World:
    rng = np.random.default_rng(seed)
    n = int(SIZE / CELL) + 1
    hills = sum(_value_noise(rng, n, c) * a for c, a in ((3, 7.0), (7, 2.5), (17, 0.8)))
    heights = 1.0 + hills - hills.min()                       # 1 .. ~9 m above water

    # River: meanders north across the map, between x = 60 and 100.
    z = np.arange(n) * CELL
    river_x = (SIZE / 2 + rng.uniform(-12, 12) + 14 * np.sin(z / SIZE * 2 * np.pi * rng.uniform(0.6, 1.2) + rng.uniform(0, 6.3))
               + 5 * np.sin(z / SIZE * 2 * np.pi * rng.uniform(1.5, 2.5) + rng.uniform(0, 6.3)))
    half = 4.0
    xs = np.arange(n) * CELL
    d = np.abs(xs[None, :] - river_x[:, None])
    bed = WATER_Y - 1.5
    t = np.clip((d - half) / BANK, 0, 1)                      # 0 in the water, 1 beyond the bank
    heights = np.where(d <= half, bed, bed + (heights - bed) * (t * t * (3 - 2 * t)))
    heights = np.where((d > half) & (heights < WATER_Y + 0.3), WATER_Y + 0.3, heights)  # banks stay dry

    w = World(seed, heights, river_x, half)

    # Bridges: 2-3, spread along the river.
    k = int(rng.integers(2, 4))
    for zc in np.linspace(SIZE * 0.2, SIZE * 0.8, k) + rng.uniform(-8, 8, k):
        rx = float(np.interp(zc, z, river_x))
        x0, x1 = rx - half - BANK, rx + half + BANK
        w.bridges.append(Bridge(x0, x1, float(zc), w.terrain_y(x0, zc) + 0.05, w.terrain_y(x1, zc) + 0.05))

    # Landmarks: away from the river, the edges and each other; both banks get some.
    for name in rng.permutation(LANDMARKS):
        for _ in range(500):
            x, zz = rng.uniform(12, SIZE - 12, 2)
            far = all(np.hypot(x - l.x, zz - l.z) > 35 for l in w.landmarks)
            if w.river_dist(x, zz) > half + BANK + 10 and far:
                w.landmarks.append(Landmark(str(name), float(x), float(zz), FOOTPRINT[name]))
                break

    # Scenery trees: not in the water, on bridges or near landmarks.
    for _ in range(170):
        x, zz = rng.uniform(3, SIZE - 3, 2)
        if (w.river_dist(x, zz) > half + 2 and w.bridge_at(x, zz) is None
                and all(np.hypot(x - l.x, zz - l.z) > l.radius + 12 for l in w.landmarks)
                and all(abs(zz - b.z) > 6 or not (b.x0 - 6 < x < b.x1 + 6) for b in w.bridges)):
            w.trees.append((float(x), float(zz), float(rng.uniform(0.8, 1.4))))
    return w
