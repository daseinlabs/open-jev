"""Headless first-person frames of the open world (Panda3D offscreen buffer, same scene as play.py).

    r = Renderer()                    # once per process (Panda allows one ShowBase)
    r.load(generate(seed))            # rebuild the scene for another map
    img = r.frame(x, z, heading, t)   # PIL RGB image from eye height; t (s) turns the windmill
    cams = r.cameras(x, z, heading, t)   # 4 head cameras, 90° each: front, left, back, right
"""
from __future__ import annotations

import numpy as np
from PIL import Image

W, H = 640, 400  # the game window's 16:10 at half size
HFOV = 80.0       # play.setup_lens
CAM_FOV = 90.0    # the 4 head cameras: 90° apart and 90° wide, so no gaps


class Renderer:
    def __init__(self, width: int = W, height: int = H):
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import loadPrcFileData
        from .play import SKY, setup_lens
        loadPrcFileData("", f"window-type offscreen\nwin-size {width} {height}\naudio-library-name null\n"
                            "sync-video 0\nframebuffer-multisample 0")
        self.base = ShowBase()
        self.base.disableMouse()
        self.base.setBackgroundColor(*SKY)
        self.base.render.setShaderOff()
        setup_lens(self.base.camLens)
        self.root, self.w, self.hubs = None, None, []

    def load(self, w):
        from .play import build_scene
        if self.root is not None:
            self.root.removeNode()
        self.root = self.base.render.attachNewNode("world")
        self.w, self.hubs = w, build_scene(self.root, w)

    def frame(self, x: float, z: float, heading: float, t: float = 0.0) -> Image.Image:
        from .play import EYE, P
        self.base.camera.setPos(*P(x, self.w.ground_y(x, z) + EYE, z))
        self.base.camera.setHpr(heading, 0, 0)
        for h in self.hubs:
            h.setR(40 * t)
        self.base.graphicsEngine.renderFrame()
        tex = self.base.win.getScreenshot()
        a = np.frombuffer(bytes(tex.getRamImageAs("RGB")), np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
        return Image.fromarray(a[::-1].copy())

    def cameras(self, x: float, z: float, heading: float, t: float = 0.0) -> list[Image.Image]:
        """The 4 head cameras at one moment, each ``CAM_FOV`` wide: front, left, back, right."""
        lens = self.base.camLens
        lens.setFov(CAM_FOV)
        try:
            return [self.frame(x, z, heading + d, t) for d in (0, 90, 180, -90)]
        finally:
            lens.setFov(HFOV)
