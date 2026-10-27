import numpy as np
from PIL import Image

from video_train import frame_select as fs


def _frames(values):
    return [Image.new("RGB", (32, 32), (v, v, v)) for v in values]


def test_uniform_weights_give_every_other_frame():
    assert fs.pick_by_weight(np.ones(32), 16) == list(range(0, 32, 2))


def test_pick_by_weight_returns_k_distinct_sorted():
    w = np.full(32, 0.01)
    w[10] = 100.0  # one frame takes almost all the weight
    out = fs.pick_by_weight(w, 16)
    assert len(out) == 16 and out == sorted(set(out))
    assert 10 in out


def test_change_puts_frames_where_the_picture_changes():
    # static, then a burst of change in candidates 20-27, then static again
    values = [0] * 20 + [30 * i for i in range(1, 9)] + [240] * 4
    out = fs.select_change(_frames(values), 16)
    assert len(out) == 16
    inside = sum(20 <= i <= 27 for i in out)
    assert inside > 16 * 8 / 32  # more than the uniform share of the burst
    assert out[0] < 10  # static stretches still get frames (floor)


def test_change_on_static_video_is_uniform():
    assert fs.select_change(_frames([128] * 32), 16) == list(range(0, 32, 2))


def test_coverage_keeps_one_frame_per_group():
    s = np.zeros(32)
    s[:4] = 1.0  # all the signal at the start
    out = fs.pick_with_coverage(s, 16)
    assert len(out) == 16 and out == sorted(set(out))
    for g in np.array_split(np.arange(32), 8):
        assert any(i in g for i in out)
    assert set(range(4)) <= set(out)


def test_question_scores_take_best_text():
    img = np.eye(3)
    txt = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    assert fs.question_scores(img, txt).tolist() == [1.0, 0.0, 1.0]


def test_short_videos_keep_all_frames():
    assert fs.pick_by_weight(np.ones(5), 16) == list(range(5))
    assert fs.pick_with_coverage(np.ones(5), 16) == list(range(5))
