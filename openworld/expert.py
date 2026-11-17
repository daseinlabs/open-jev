"""Instructions, the expert and the agent's moves in the open world (numpy only, no rendering).

Once a second the agent either says it has arrived or makes one of the four moves;
``move`` carries a move out over one second with the game's walking rules. The expert (the
reference for scoring, it sees the whole map) labels every state from a distance field to the
goal (Dijkstra on a 1 m grid, once per task), so it is right from any state, including ones
the agent reached by mistake; at the goal it returns ``STOP``. Headings are Panda's: 0 faces north, 90 west (``play.py``); x east, z north.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .world import CELL, RAIL, SIZE, WATER_Y, World

FORWARD, LEFT, RIGHT, AROUND, STOP = range(5)  # STOP: arrived (asked as its own yes/no question)
WALK, TURN, FPS = 2.5, 45.0, 2       # metres and degrees per decision (one second), frames per second
HFOV = 80.0                          # horizontal field of view (play.py)
RES = 1.0                            # expert grid spacing, metres
TOP = {"red house": 6.5, "stone tower": 19.0, "windmill": 11.0, "big tree": 12.0, "blue tent": 2.4}

TEMPLATES = {
    "landmark": ("Go to the {name}.", "Walk over to the {name}.", "Head for the {name}.",
                 "Find the {name} and stop next to it."),
    "cross": ("Cross the river.", "Cross the bridge.", "Get to the other side of the river.",
              "Use a bridge to reach the far bank."),
    "bridge": ("Walk onto the bridge.", "Go to the nearest bridge.", "Get on the closest bridge.",
               "Stand on the nearest bridge."),
}
KINDS, KIND_P = ("landmark", "cross", "bridge"), (0.5, 0.25, 0.25)
# Asked every second; "yes" ends the episode (success if the goal is reached).

def river_x(w: World, z):
    return np.interp(z, np.arange(len(w.river_x)) * CELL, w.river_x)


def side(w: World, x, z):
    """-1 west of the river, +1 east."""
    return np.sign(x - river_x(w, z))


def on_bridge_mask(w: World, X, Z):
    m = np.zeros(np.shape(X), bool)
    for b in w.bridges:
        m |= (X >= b.x0 - 0.5) & (X <= b.x1 + 0.5) & (np.abs(Z - b.z) <= b.width / 2 - RAIL)
    return m


def terrain(w: World, X, Z):
    """Vectorised ``World.terrain_y``."""
    h, n = w.heights, w.heights.shape[0]
    gx, gz = np.clip(X / CELL, 0, n - 1.001), np.clip(Z / CELL, 0, n - 1.001)
    i, j = gz.astype(int), gx.astype(int)
    fz, fx = gz - i, gx - j
    return (h[i, j] * (1 - fx) + h[i, j + 1] * fx) * (1 - fz) + (h[i + 1, j] * (1 - fx) + h[i + 1, j + 1] * fx) * fz


def walkable_grid(w: World, X, Z):
    """Vectorised ``World.walkable`` at the points (X, Z)."""
    ok = (X >= 1) & (X <= SIZE - 1) & (Z >= 1) & (Z <= SIZE - 1)
    on = on_bridge_mask(w, X, Z)
    ok &= on | (terrain(w, X, Z) >= WATER_Y + 0.15)
    for b in w.bridges:  # railings (``World.walkable``)
        ok &= on | ~((X >= b.x0) & (X <= b.x1) & (np.abs(Z - b.z) <= b.width / 2 + RAIL))
    for tx, tz, k in w.trees:
        ok &= np.hypot(X - tx, Z - tz) >= 0.5 * k
    for l in w.landmarks:
        ok &= np.hypot(X - l.x, Z - l.z) > l.radius
    return ok


def grid(n=int(SIZE / RES)):
    c = (np.arange(n) + 0.5) * RES
    return np.meshgrid(c, c)  # X[row=z, col=x], Z


@dataclass
class Task:
    kind: str
    text: str
    target: str | None      # landmark name (kind "landmark")
    goal_side: int | None   # +1 / -1 (kind "cross")
    dist: np.ndarray        # metres to the goal along walkable cells (inf: unreachable)

    def done(self, w: World, x: float, z: float) -> bool:
        return bool(goal_mask(w, self.kind, self.target, self.goal_side, np.array(x), np.array(z)))


def goal_mask(w: World, kind, target, goal_side, X, Z):
    if kind == "landmark":
        l = next(l for l in w.landmarks if l.name == target)
        return np.hypot(X - l.x, Z - l.z) <= l.radius + 4.0
    if kind == "cross":
        # just past the end of a bridge deck (decks span river_half + BANK from the centre line)
        return (side(w, X, Z) == goal_side) & (np.abs(X - river_x(w, Z)) > w.river_half + 5.0) & ~on_bridge_mask(w, X, Z)
    m = np.zeros(np.shape(X), bool)  # bridge: on a deck, away from its ends
    for b in w.bridges:
        m |= (X >= b.x0 + 3) & (X <= b.x1 - 3) & (np.abs(Z - b.z) <= b.width / 2 - RAIL)
    return m


STEPS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


@dataclass
class Nav:
    """Walkable cells and, per step direction, whether the straight step from a cell stays on
    walkable ground (checked every 0.25 m: a cell grid alone steps over the thin strip of
    water between a river bank and the side of a bridge, which the agent cannot)."""
    walk: np.ndarray
    edge: dict

    @classmethod
    def of(cls, w: World):
        X, Z = grid()
        walk = walkable_grid(w, X, Z)
        edge = {}
        for di, dj in STEPS:
            ok = walk.copy()
            for t in (0.25, 0.5, 0.75):
                ok &= walkable_grid(w, X + t * dj * RES, Z + t * di * RES)
            edge[di, dj] = ok
        return cls(walk, edge)


def distance_field(nav: Nav, goal):
    """Dijkstra from every goal cell over 8-connected steps; cells next to an obstacle cost
    double, so paths keep a little clearance."""
    walk = nav.walk
    n = walk.shape[0]
    pad = np.pad(~walk, 1, constant_values=True)
    near = np.zeros_like(walk)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near |= pad[1 + di:n + 1 + di, 1 + dj:n + 1 + dj]
    cost = np.where(near, 2.0, 1.0) * RES
    dist = np.full(walk.shape, np.inf)
    heap = [(0.0, int(i), int(j)) for i, j in zip(*np.nonzero(goal & walk))]
    for _, i, j in heap:
        dist[i, j] = 0.0
    heapq.heapify(heap)
    steps = [(di, dj, math.hypot(di, dj), nav.edge[-di, -dj]) for di, dj in STEPS]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, s, ok in steps:
            a, b = i + di, j + dj  # the step (a, b) -> (i, j) must be walkable
            if 0 <= a < n and 0 <= b < n and ok[a, b]:
                nd = d + s * 0.5 * (cost[i, j] + cost[a, b])
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(heap, (nd, a, b))
    return dist


def make_task(w: World, rng: np.random.Generator, nav: Nav, x: float, z: float) -> Task:
    """A random instruction (any of its wordings) for an agent standing at (x, z)."""
    kind = str(rng.choice(KINDS, p=KIND_P))
    temps = TEMPLATES[kind]
    text = temps[int(rng.integers(len(temps)))]
    target = goal_side = None
    if kind == "landmark":
        target = str(rng.choice([l.name for l in w.landmarks]))
        text = text.format(name=target)
    if kind == "cross":
        goal_side = -int(side(w, x, z))
    X, Z = grid()
    return Task(kind, text, target, goal_side, distance_field(nav, goal_mask(w, kind, target, goal_side, X, Z)))


def cell(x, z):
    n = int(SIZE / RES)
    return min(n - 1, max(0, int(z / RES))), min(n - 1, max(0, int(x / RES)))


def remaining(task: Task, x, z, w: World | None = None) -> float:
    """Path metres to the goal from (x, z): its own cell, or the best neighbour plus a step when
    the cell centre sits inside an obstacle the agent is standing next to. With ``w``, only the
    neighbours it can walk straight to (not through a railing thinner than a cell)."""
    i, j = cell(x, z)
    if np.isfinite(task.dist[i, j]):
        return float(task.dist[i, j])
    best = np.inf
    for a in range(max(0, i - 1), min(task.dist.shape[0], i + 2)):
        for b in range(max(0, j - 1), min(task.dist.shape[1], j + 2)):
            cx, cz = (b + 0.5) * RES, (a + 0.5) * RES
            if w is None or all(w.walkable(x + (cx - x) * t, z + (cz - z) * t) for t in (0.25, 0.5, 0.75, 1.0)):
                best = min(best, float(task.dist[a, b]))
    return best + RES


def heading_to(x, z, tx, tz) -> float:
    """Panda heading facing (tx, tz) from (x, z): forward is (-sin h, cos h)."""
    return math.degrees(math.atan2(-(tx - x), tz - z))


def value(w: World, task: Task, x, z) -> float:
    return -1.0 if task.done(w, x, z) else remaining(task, x, z, w)


def expert(w: World, task: Task, x: float, z: float, heading: float) -> int:
    """Try one second of walking in each of the 8 headings reachable by 45-degree turns (with the
    real sliding rules) and head for the one that ends closest to the goal; walk forward when
    the current heading is within 0.5 m of the best."""
    if task.done(w, x, z):
        return STOP
    here = value(w, task, x, z)
    ends = [move(w, x, z, heading + 45 * k, FORWARD)[-1][:2] for k in range(8)]
    end = [value(w, task, *e) for e in ends]
    k = min(range(8), key=lambda k: (end[k], k != 0))
    if end[k] >= here:  # no heading gets closer in one second: the best one that moves at all
        moving = [k for k in range(8) if math.hypot(ends[k][0] - x, ends[k][1] - z) > 0.5]
        k = min(moving, key=lambda k: (end[k], k != 0)) if moving else 4
    elif end[0] <= end[k] + 0.5 and end[0] < here:
        k = 0
    return {0: FORWARD, 1: LEFT, 2: LEFT, 6: RIGHT, 7: RIGHT}.get(k, AROUND)


def move(w: World, x: float, z: float, heading: float, action: int) -> list[tuple[float, float, float]]:
    """The agent's pose at each frame time of the next second, (x, z, heading) x FPS."""
    poses, sub = [], 10
    turn = {LEFT: TURN, RIGHT: -TURN, AROUND: 180.0}.get(action, 0.0)
    for k in range(1, sub + 1):
        if action == FORWARD:
            h = math.radians(heading)
            dx, dz = -math.sin(h) * WALK / sub, math.cos(h) * WALK / sub
            for sx, sz in ((dx, dz), (dx, 0), (0, dz)):  # slide along blocked edges, as in play.py
                if w.walkable(x + sx, z + sz):
                    x, z = x + sx, z + sz
                    break
        else:
            heading += turn / sub
        if k % (sub // FPS) == 0:
            poses.append((x, z, heading % 360))
    return poses


def visible(w: World, x, z, heading, tx, tz, ty, max_dist: float = 120.0) -> bool:
    """Point (tx, ty, tz) inside the horizontal field of view and not hidden behind terrain
    (trees ignored)."""
    d = math.hypot(tx - x, tz - z)
    diff = (heading_to(x, z, tx, tz) - heading + 180) % 360 - 180
    if d > max_dist or abs(diff) > HFOV / 2 - 4:
        return False
    eye = w.ground_y(x, z) + 1.7
    for t in np.linspace(0.05, 0.95, 24):
        if w.terrain_y(x + (tx - x) * t, z + (tz - z) * t) > eye + (ty - eye) * t:
            return False
    return True


def in_view(w: World, x: float, z: float, heading: float, name: str) -> bool:
    l = next(l for l in w.landmarks if l.name == name)
    return visible(w, x, z, heading, l.x, l.z, w.terrain_y(l.x, l.z) + 0.6 * TOP[name])


def target_in_view(w: World, task: Task, x, z, heading) -> bool:
    """The landmark, or (cross / bridge) any bridge's railing at mid-span."""
    if task.kind == "landmark":
        return in_view(w, x, z, heading, task.target)
    return any(visible(w, x, z, heading, (b.x0 + b.x1) / 2, b.z, b.deck_y((b.x0 + b.x1) / 2) + 0.9) for b in w.bridges)


def findable(w: World, task: Task, x, z) -> bool:
    """The target comes into view by turning on the spot (one of 8 headings)."""
    return any(target_in_view(w, task, x, z, h) for h in range(0, 360, 45))


def spawn(w: World, rng: np.random.Generator, nav: Nav) -> tuple[float, float, float]:
    """A random walkable start and heading, away from landmarks."""
    while True:
        x, z = rng.uniform(4, SIZE - 4, 2)
        i, j = cell(x, z)
        if nav.walk[i, j] and w.walkable(x, z) and all(math.hypot(x - l.x, z - l.z) > l.radius + 6 for l in w.landmarks):
            return float(x), float(z), float(rng.uniform(0, 360))
