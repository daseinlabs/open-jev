import random

from video_train.vdm_data import kind, norm_id, pack, parse_span, sample, turns
from video_train.vdm_eval import iou


def test_turn_formats():
    star = {"source": "star", "question": "What?", "options": ["a", "b"], "answer_id": 1, "clip": [3.21, 9.4]}
    (text, ans), = turns(star)
    assert text.startswith("From 3.2 to 9.4 seconds: What?\nA. a\nB. b\n") and ans == "B"
    clev = {"source": "clevrer", "question": "Which?", "options": ["x", "y"], "correct": [True, False]}
    assert kind(clev) == "yesno" and [a for _, a in turns(clev)] == ["Yes", "No"]
    sta = {"source": "charades_sta", "question": "a person opens a door.", "span": [1.0, 7.25]}
    (text, ans), = turns(sta)
    assert '"a person opens a door"' in text and parse_span(ans) == (1.0, 7.2)


def test_parse_span_and_iou():
    assert parse_span("The event happens from 2.5 to 6 seconds.") == (2.5, 6.0)
    assert parse_span("no idea") is None
    assert iou((2, 6), (4, 8)) == 2 / 6 and iou(None, (0, 1)) == 0.0 and iou((6, 2), (2, 6)) == 1.0


def test_norm_id():
    assert norm_id("ytb_abc-1") == norm_id("abc-1") == norm_id("vdm/videos/llava_video/ytb_abc-1.mp4") == "abc-1"


def test_sample_caps_and_balances():
    rs = [{"video_id": v, "qid": i, "type": t} for v in "abcdef" for i, t in enumerate("xxxy")]
    got = sample(rs, 7, 2, random.Random(0), balance="type")
    per = {}
    for r in got:
        per.setdefault(r["video_id"], []).append(r["type"])
    assert len(got) == 7 and all(len(t) <= 2 for t in per.values())
    assert all(sorted(t) == ["x", "y"] for t in per.values() if len(t) == 2)


def test_answer_turns_groups_each_assistant_turn():
    from video_train.qwen3vl import Inputs
    import numpy as np
    inp = Inputs.__new__(Inputs)
    inp.im_start, inp.im_end, inp.assistant_ids = 1, 2, [9]
    # user turn, assistant "A", user turn, assistant "x y"
    ids = np.array([[1, 5, 6, 2, 1, 9, 70, 2, 1, 5, 2, 1, 9, 80, 81, 2]])
    assert inp.answer_turns(ids) == [([5, 6], [70, 2]), ([12, 13, 14], [80, 81, 2])]


def test_train_order_is_a_fixed_permutation_per_pass():
    from video_train.vdm_train import order
    assert sorted(order(10, 0, 0)) == list(range(10)) and order(10, 0, 0) == order(10, 0, 0)
    assert order(10, 0, 0) != order(10, 0, 1)


def test_trafficqa_rows_drop_empty_options_and_remap_answer(tmp_path):
    import json
    from video_train.vdm_splits import trafficqa
    head = ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type",
            "option0", "option1", "option2", "option3", "answer"]
    lines = [head,
             [1, 10, "b_a_clip_001.mp4", 1, "What happened?", "A", "Crash", "No accident", "", "", 1],
             [2, 11, "b_b_clip_002.mp4", 3, "Why?", "C", "x", "", "y", "z", 2],
             [3, 12, "b_c_clip_003.mp4", 1, "Bad", "U", "x", "y", "", "", 2]]
    p = tmp_path / "R2.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    rs, skipped = trafficqa(p)
    assert skipped == 1 and [r["options"] for r in rs] == [["Crash", "No accident"], ["x", "y", "z"]]
    assert [r["answer_id"] for r in rs] == [1, 1] and rs[0]["video"] == "vdm/videos/trafficqa/b_a_clip_001.mp4"
    assert rs[0]["video_id"] == "b_a_clip_001" and kind(rs[0]) == "mc" and turns(rs[1])[0][1] == "B"


def test_pack_one_row_per_video_with_all_its_turns():
    qs = [{"source": "trafficqa", "video_id": v, "qid": i, "question": "Q?", "options": ["a", "b"], "answer_id": i % 2,
           "video": f"vdm/videos/trafficqa/{v}.mp4"} for v, i in (("x", 0), ("x", 1), ("y", 2))]
    items = {i["video_id"]: i for i in pack(qs, random.Random(0))}
    assert items["x"]["questions"] == 2 and sorted(a for _, a in items["x"]["turns"]) == ["A", "B"]
    assert items["y"]["turns"] == [list(turns(qs[2])[0])] and items["y"]["video"].endswith("y.mp4")
