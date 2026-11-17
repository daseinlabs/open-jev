"""Play the open world in first person (Panda3D).

    .venv/bin/python -m openworld.play            # random map
    .venv/bin/python -m openworld.play --seed 3   # the same map every time

WASD move, mouse or arrow keys look, Shift run, Esc frees the mouse (click to grab it again), Q quits.

Plain Panda3D with the fixed-function pipeline (no shaders: macOS only offers OpenGL 2.1 here
by default). Sunlight is baked into vertex colours. World coordinates (x east, z north, y up)
map to Panda's (x, y, z) as (x, z, y).
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (AntialiasAttrib, ClockObject, Fog, Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, KeyboardButton, TextNode, TransparencyAttrib,
                          WindowProperties, loadPrcFileData)

from .world import CELL, SIZE, WATER_Y, generate

EYE, WALK, RUN, LOOK = 1.7, 5.0, 10.0, 0.15        # metres, m/s, m/s, degrees per pixel
KEY_TURN, KEY_PITCH = 90.0, 60.0                   # arrow keys, degrees per second
SUN = np.array([-0.5, -0.35, 0.75]) / np.linalg.norm([-0.5, -0.35, 0.75])  # towards the sun (Panda axes)
SKY = (0.59, 0.75, 0.9, 1)


def shade(rgb, normal):
    k = 0.5 + 0.5 * max(0.0, float(np.dot(normal, SUN)))
    return (rgb[0] * k, rgb[1] * k, rgb[2] * k, rgb[3] if len(rgb) > 3 else 1.0)


def mesh(name, verts, colors, tris):
    vd = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vd.setNumRows(len(verts))
    vw, cw = GeomVertexWriter(vd, "vertex"), GeomVertexWriter(vd, "color")
    for v, c in zip(verts, colors):
        vw.addData3(*v)
        cw.addData4(*c)
    prim = GeomTriangles(Geom.UHStatic)
    for t in tris:
        prim.addVertices(*t)
    g = Geom(vd)
    g.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(g)
    return node


CUBE_FACES = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
              ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
              ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
              ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
              ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
              ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]


class Builder:
    """Collects shaded boxes and spheres into one mesh (one draw call for the whole scene)."""

    def __init__(self):
        self.v, self.c, self.t = [], [], []

    @staticmethod
    def _rot(hpr):
        h, p, r = (math.radians(a) for a in hpr)
        rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
        return rz @ rx @ ry

    def box(self, pos, size, rgb, hpr=(0, 0, 0)):
        """Box centred at ``pos`` (Panda axes), full ``size``; ``hpr`` heading/pitch/roll in degrees."""
        R, half = self._rot(hpr), np.array(size) / 2
        for n, quad in CUBE_FACES:
            base = len(self.v)
            col = shade(rgb, R @ np.array(n, float))
            for q in quad:
                self.v.append(tuple(np.array(pos) + R @ (np.array(q) * half)))
                self.c.append(col)
            self.t += [(base, base + 1, base + 2), (base, base + 2, base + 3)]

    def sphere(self, pos, radius, rgb, rings=6, segs=10):
        base = len(self.v)
        for i in range(rings + 1):
            th = math.pi * i / rings
            for j in range(segs + 1):
                ph = 2 * math.pi * j / segs
                n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                self.v.append(tuple(np.array(pos) + radius * n))
                self.c.append(shade(rgb, n))
        for i in range(rings):
            for j in range(segs):
                a, b = base + i * (segs + 1) + j, base + (i + 1) * (segs + 1) + j
                self.t += [(a, b, a + 1), (a + 1, b, b + 1)]

    def node(self, name):
        return mesh(name, self.v, self.c, self.t)


def P(x, y, z):
    """World (x east, y up, z north) -> Panda (x, y north, z up)."""
    return (x, z, y)


def terrain(w):
    h = w.heights
    n = h.shape[0]
    gz, gx = np.gradient(h, CELL)
    normals = np.stack([-gx, -gz, np.ones_like(h)], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    rock = np.quantile(h, 0.93)
    jitter = np.random.default_rng(w.seed).uniform(0.9, 1.1, h.shape)
    verts, cols = [], []
    for i in range(n):
        for j in range(n):
            y = h[i, j]
            if y < WATER_Y + 0.6:
                c = (0.76, 0.70, 0.50)                       # sand at the water
            elif y > rock:
                c = (0.50, 0.50, 0.47)                       # rock on hill tops
            else:
                g = jitter[i, j]
                c = (0.30 * g, 0.55 * g, 0.22 * g)
            verts.append(P(j * CELL, y, i * CELL))
            cols.append(shade(c, normals[i, j]))
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    return mesh("terrain", verts, cols, tris)


def scenery(w):
    """Bridges, landmarks and trees as one mesh; also the windmill hub positions."""
    s = Builder()
    wood, dark = (0.55, 0.37, 0.22), (0.35, 0.24, 0.14)
    for b in w.bridges:
        steps = 12
        for k in range(steps):
            xa, xb = b.x0 + (b.x1 - b.x0) * k / steps, b.x0 + (b.x1 - b.x0) * (k + 1) / steps
            ya, yb = b.deck_y(xa), b.deck_y(xb)
            seg, tilt = math.hypot(xb - xa, yb - ya), math.degrees(math.atan2(yb - ya, xb - xa))
            mid = np.array(P((xa + xb) / 2, (ya + yb) / 2 - 0.1, b.z))
            s.box(mid, (seg + 0.05, b.width, 0.2), wood, hpr=(0, 0, -tilt))
            for side in (-1, 1):  # railings and posts
                s.box(mid + (0, side * b.width / 2, 0.9), (seg + 0.05, 0.08, 0.08), dark, hpr=(0, 0, -tilt))
                s.box(P(xa, ya + 0.45, b.z + side * b.width / 2), (0.12, 0.12, 0.9), dark)
        for t in (0.3, 0.7):  # piers down to the river bed
            x = b.x0 + (b.x1 - b.x0) * t
            top, bot = b.deck_y(x) - 0.2, WATER_Y - 1.6
            s.box(P(x, (top + bot) / 2, b.z), (0.5, b.width * 0.8, top - bot), dark)
    hubs = []
    for l in w.landmarks:
        g = w.terrain_y(l.x, l.z)
        at = lambda dx, dy, dz: P(l.x + dx, g + dy, l.z + dz)
        if l.name == "red house":
            s.box(at(0, 2, 0), (7, 6, 4), (0.67, 0.16, 0.14))
            s.box(at(0, 4.4, 0), (7.4, 4.6, 4.6), (0.30, 0.18, 0.16), hpr=(0, 45, 0))   # roof ridge east-west
            s.box(at(0, 1.1, -3.02), (1.2, 0.1, 2.2), dark)                              # door, south side
        elif l.name == "stone tower":
            for k in range(6):
                v = 0.6 - k * 0.025
                s.box(at(0, 1.5 + 3 * k, 0), (4.5, 4.5, 3), (v, v, v * 0.97), hpr=(k * 15, 0, 0))
            s.box(at(0, 18.6, 0), (5.2, 5.2, 1.2), (0.43, 0.43, 0.41))
        elif l.name == "windmill":
            s.box(at(0, 5, 0), (4, 4, 10), (0.92, 0.9, 0.84))
            s.box(at(0, 10.6, 0), (4.4, 4.4, 1.2), (0.47, 0.24, 0.16))
            hubs.append(at(0, 9, -2.3))
        elif l.name == "big tree":
            s.box(at(0, 3, 0), (1.4, 1.4, 6), (0.39, 0.27, 0.16))
            for (dx, dy, dz), r in (((0, 8, 0), 4), ((2, 6.5, 1), 2.5), ((-2, 6.8, -1), 2.5)):
                s.sphere(at(dx, dy, dz), r, (0.16, 0.43, 0.18))
        elif l.name == "blue tent":
            s.box(at(0, 1.2, 0), (3.4, 4.5, 3.4), (0.16, 0.31, 0.75), hpr=(0, 0, 45))
    for x, z, k in w.trees:
        y = w.terrain_y(x, z)
        s.box(P(x, y + 1.2 * k, z), (0.4 * k, 0.4 * k, 2.4 * k), (0.37, 0.25, 0.16))
        s.sphere(P(x, y + 3.2 * k, z), 1.3 * k, (0.12, 0.38 + 0.1 * (k - 1), 0.15), rings=5, segs=8)
    return s.node("scenery"), hubs


def blades_node():
    s = Builder()
    for k in range(4):
        a = math.radians(k * 90)
        s.box((2 * math.sin(a), 0, 2 * math.cos(a)), (0.9, 0.15, 4), (0.78, 0.76, 0.7), hpr=(0, 0, k * 90))
    return s.node("blades")


def build_scene(root, w):
    """Terrain, water, scenery and fog under ``root``; returns the windmill blade nodes."""
    root.attachNewNode(terrain(w))
    corners = [P(0, WATER_Y, 0), P(SIZE, WATER_Y, 0), P(SIZE, WATER_Y, SIZE), P(0, WATER_Y, SIZE)]
    water = root.attachNewNode(mesh("water", corners, [(0.15, 0.4, 0.75, 0.8)] * 4, [(0, 1, 2), (0, 2, 3)]))
    water.setTransparency(TransparencyAttrib.MAlpha)
    static, hub_pos = scenery(w)
    root.attachNewNode(static)
    hubs = []
    for p in hub_pos:
        h = root.attachNewNode(blades_node())
        h.setPos(*p)
        hubs.append(h)
    fog = Fog("fog")
    fog.setColor(*SKY[:3])
    fog.setLinearRange(50, 170)
    root.setFog(fog)
    return hubs


def setup_lens(lens):
    lens.setFov(80)
    lens.setNear(0.1)


class Game(ShowBase):
    def __init__(self, seed):
        loadPrcFileData("", f"window-title open world (seed {seed})\nwin-size 1280 800\nsync-video 1")
        super().__init__()
        self.seed, self.w = seed, generate(seed)
        self.disableMouse()
        self.setBackgroundColor(*SKY)
        self.render.setShaderOff()
        self.hubs = build_scene(self.render, self.w)
        self.render.setAntialias(AntialiasAttrib.MAuto)

        x, z = self.w.spawn()
        self.pos, self.heading, self.pitch = np.array([x, self.w.ground_y(x, z), z]), -90.0, 0.0  # facing east
        setup_lens(self.camLens)
        self.hud = OnscreenText(text="", pos=(-1.55, 0.92), scale=0.045, fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.45),
                                align=TextNode.ALeft, mayChange=True)
        OnscreenText(text="WASD move   Shift run   mouse / arrows look   Esc free mouse   Q quit", pos=(-1.55, -0.95),
                     scale=0.04, fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.45), align=TextNode.ALeft)
        self.grabbed = False
        self.grab(True)
        self.accept("escape", self.grab, [False])
        self.accept("mouse1", self.grab, [True])
        self.accept("q", self.userExit)
        self.taskMgr.add(self.update, "update")

    def grab(self, on):
        """Hide the cursor while looking. Absolute mouse mode on every platform: macOS mishandles
        relative mode plus re-centring every frame (the view jumps instead of turning)."""
        props = WindowProperties()
        props.setCursorHidden(on)
        self.win.requestProperties(props)
        self.grabbed, self.last = on, None
        self.center()

    def center(self):
        if self.grabbed:
            self.win.movePointer(0, self.win.getXSize() // 2, self.win.getYSize() // 2)
            self.last = None  # the warp is not a movement: skip one frame

    def update(self, task):
        dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
        if self.grabbed and self.mouseWatcherNode.hasMouse():
            p = self.win.getPointer(0)
            x, y = p.getX(), p.getY()
            if self.last is not None:  # turn by the movement since the last frame
                self.heading -= (x - self.last[0]) * LOOK
                self.pitch = float(np.clip(self.pitch - (y - self.last[1]) * LOOK, -85, 85))
            self.last = (x, y)
            m = 100  # re-centre only near the edge, so the cursor never leaves the window
            if not (m < x < self.win.getXSize() - m and m < y < self.win.getYSize() - m):
                self.center()
        down = self.mouseWatcherNode.isButtonDown
        self.heading += (down(KeyboardButton.left()) - down(KeyboardButton.right())) * KEY_TURN * dt
        self.pitch = float(np.clip(self.pitch + (down(KeyboardButton.up()) - down(KeyboardButton.down()))
                                   * KEY_PITCH * dt, -85, 85))
        key = lambda c: down(KeyboardButton.asciiKey(c))
        f, r = key("w") - key("s"), key("d") - key("a")
        if f or r:
            h = math.radians(self.heading)
            fwd, right = np.array([-math.sin(h), math.cos(h)]), np.array([math.cos(h), math.sin(h)])  # Panda x, y
            step = fwd * f + right * r
            step = step / np.linalg.norm(step) * (RUN if down(KeyboardButton.shift()) else WALK) * dt
            x, z = self.pos[0], self.pos[2]
            for dx, dz in ((step[0], step[1]), (step[0], 0), (0, step[1])):  # slide along blocked edges
                if self.w.walkable(x + dx, z + dz):
                    self.pos[0], self.pos[2] = x + dx, z + dz
                    break
        ground = self.w.ground_y(self.pos[0], self.pos[2])
        self.pos[1] += (ground - self.pos[1]) * min(1.0, dt * 12)
        self.camera.setPos(*P(self.pos[0], self.pos[1] + EYE, self.pos[2]))
        self.camera.setHpr(self.heading, self.pitch, 0)
        for h in self.hubs:
            h.setR(h.getR() + 40 * dt)
        lm, d = self.w.nearest_landmark(self.pos[0], self.pos[2])
        on = " on the bridge" if self.w.bridge_at(self.pos[0], self.pos[2]) else ""
        self.hud.setText(f"seed {self.seed}   nearest: {lm.name} ({d:.0f} m){on}")
        return task.cont


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    Game(random.randrange(10_000) if args.seed is None else args.seed).run()


if __name__ == "__main__":
    main()
