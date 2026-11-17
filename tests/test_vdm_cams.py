import math

import numpy as np

from openworld import expert as E
from openworld.world import FOOTPRINT
from video_train.vdm_cams import (BACK, FRONT, LEFT, NEAR_M, RIGHT, VIEW_WIDTH, WARMUP, box_distance, cx_to_angle,
                                     decide, parse_cx, side_of, truth, videos)
from video_train.vdm_game import landmark_episodes


def first_episode():
    (_, (w, (x, z, h), task)), = list(landmark_episodes("dev", 1))
    l = next(l for l in w.landmarks if l.name == task.target)
    return w, x, z, task, E.heading_to(x, z, l.x, l.z)


def test_rule():
    assert decide({"cam": None})[0] == E.FORWARD
    assert decide({"cam": LEFT})[0] == E.LEFT
    assert decide({"cam": RIGHT})[0] == E.RIGHT
    assert decide({"cam": BACK})[0] == E.AROUND
    assert [decide({"cam": FRONT, "side": s, "near": False})[0] for s in range(3)] == [E.LEFT, E.FORWARD, E.RIGHT]
    assert decide({"cam": FRONT, "side": 1, "near": True})[0] == E.STOP


def test_blocked_forward_turns_toward_the_landmark():
    assert decide({"cam": None, "blocked": True})[0] == E.LEFT
    assert decide({"cam": FRONT, "side": 1, "near": False, "angle": 13.7, "blocked": True})[0] == E.LEFT
    assert decide({"cam": FRONT, "side": 1, "near": False, "angle": -5.0, "blocked": True})[0] == E.RIGHT
    assert decide({"cam": LEFT, "blocked": True})[0] == E.LEFT  # not walking forward: unchanged
    assert decide({"cam": FRONT, "side": 1, "near": True, "blocked": True})[0] == E.STOP


def test_box_distance():
    # A box whose edges are atan(w/2 / d) either side of the centre gives back d.
    for name, d in [("blue tent", 7.0), ("stone tower", 20.0)]:
        half = math.degrees(math.atan(VIEW_WIDTH[name] / 2 / d))
        edge = lambda a: 500 * (1 - math.tan(math.radians(a)))  # inverse of cx_to_angle
        out = f'[{{"bbox_2d": [{edge(half):.3f}, 100, {edge(-half):.3f}, 400]}}]'
        assert abs(box_distance(out, name) - d) < 0.01
    assert box_distance("no box", "windmill") is None
    assert box_distance('[{"bbox_2d": [500, 100, 500, 400]}]', "windmill") is None
    assert set(VIEW_WIDTH) == set(FOOTPRINT) and NEAR_M == 4.0  # expert.goal_mask's + 4 m


def test_warmup_video_comes_first():
    from PIL import Image
    clips = [[Image.new("RGB", (64, 40), (k * 60, 0, 0))] * 8 for k in range(4)]
    vs = videos(clips)
    assert len(vs) == 5 and vs[0][0].endswith(WARMUP) and vs[-1][0].startswith("Front camera")


def test_truth_camera_and_side():
    w, x, z, task, facing = first_episode()
    if not truth(w, task, x, z, facing)["cam"] == FRONT:
        return  # not in sight from the start: nothing to check
    # The landmark at +a degrees (left) of the heading facing - a.
    for a, cam, side in [(0, FRONT, 1), (30, FRONT, 0), (-30, FRONT, 2), (90, LEFT, 0), (-90, RIGHT, 2), (180, BACK, 1)]:
        t = truth(w, task, x, z, facing - a)
        assert t["cam"] == cam and (cam != FRONT or t["side"] == side), a


def test_box_centre_to_angle():
    assert cx_to_angle(500) == 0
    assert abs(cx_to_angle(0) - 45) < 1e-9 and abs(cx_to_angle(1000) + 45) < 1e-9  # left edge = 45° left
    assert side_of(cx_to_angle(250)) == 0 and side_of(cx_to_angle(500)) == 1 and side_of(cx_to_angle(750)) == 2
    assert parse_cx('```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "windmill"}]\n```') == 200
    assert parse_cx("I cannot see it.") is None


def test_cameras_are_90_degrees_apart_and_restore_the_lens(renderer):
    r = renderer
    w, x, z, _, h = first_episode()
    r.load(w)
    before = np.asarray(r.frame(x, z, h))
    cams = [np.asarray(c) for c in r.cameras(x, z, h)]
    assert len(cams) == 4 and all(c.shape == before.shape for c in cams)
    assert abs(r.base.camLens.getFov()[0] - 80) < 1e-3
    assert np.array_equal(np.asarray(r.frame(x, z, h)), before)
    assert not np.array_equal(cams[0], before)  # wider than the game camera
