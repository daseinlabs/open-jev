import numpy as np

from openworld import expert as E
from openworld.world import generate


def test_walkable_grid_matches_world():
    w = generate(3)
    pts = np.random.default_rng(0).uniform(0, 160, (2000, 2))
    assert all(bool(E.walkable_grid(w, np.array(x), np.array(z))) == w.walkable(x, z) for x, z in pts)


def test_expert_reaches_every_kind_of_goal():
    seen = set()
    for seed in range(12):
        w = generate(seed)
        nav = E.Nav.of(w)
        rng = np.random.default_rng(seed)
        x, z, h = E.spawn(w, rng, nav)
        task = E.make_task(w, rng, nav, x, z)
        for _ in range(150):
            a = E.expert(w, task, x, z, h)
            if a == E.STOP:
                break
            x, z, h = E.move(w, x, z, h, a)[-1]
        assert a == E.STOP and task.done(w, x, z), (seed, task.text)
        seen.add(task.kind)
    assert seen == {"landmark", "cross", "bridge"}


def test_move_and_headings():
    w = generate(0)
    x, z = w.spawn()
    assert E.move(w, x, z, 0.0, E.LEFT)[-1][2] == 45.0 and E.move(w, x, z, 0.0, E.RIGHT)[-1][2] == 315.0
    assert len(E.move(w, x, z, 0.0, E.FORWARD)) == E.FPS
    assert abs(E.heading_to(0, 0, -1, 0) - 90) < 1e-9 and abs(E.heading_to(0, 0, 1, 0) + 90) < 1e-9  # west, east


def test_episode_targets_can_be_found_by_turning():
    from video_train.vdm_game import episodes
    for seed, (w, (x, z, h), task) in episodes("dev", 4):
        assert E.findable(w, task, x, z) and not task.done(w, x, z)


def test_railings_block_stepping_off_the_deck_over_the_bank():
    w = generate(3)
    b = w.bridges[0]
    x = b.x0 + 1.0  # over the bank: dry ground on both sides of the deck
    assert w.terrain_y(x, b.z + b.width) > 0.15
    assert w.walkable(x, b.z) and not w.walkable(x, b.z + b.width / 2 + 0.2)
    end = E.move(w, x, b.z, 0.0, E.FORWARD)[-1]  # heading 0 = north, straight at the railing
    assert w.bridge_at(end[0], end[1]) is b


def test_landmark_episodes_need_no_river_crossing():
    from video_train.vdm_game import episodes, landmark_episodes, same_side
    eps = list(landmark_episodes("dev", 6))
    assert len(eps) == 6 and all(t.kind == "landmark" and same_side(w, t, p[0], p[1]) for _, (w, p, t) in eps)
    assert {s for s, _ in eps} <= {s for s, _ in episodes("dev", 40)}
