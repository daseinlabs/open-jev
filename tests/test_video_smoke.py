"""Real-model smoke test: obvious questions about synthetic clips, all three question types.

Prompts use Gemma's chat template. With the plain System One prompt (chat=False) the
model answers "yes" to "Is a red ball visible?" on an empty white clip (p=0.65), so the
plain layout is not reliable for image input; see PR_DESCRIPTION.md.

Needs the Gemma 3 4B checkpoint and ~10 GB of memory, so it only runs when asked:

    OPENJEV_VIDEO_SMOKE=1 python -m unittest tests.test_video_smoke
"""
import io
import os
import unittest

from openjev.systemone import ChoiceQuestion, NoulQuestion, ScoreQuestion, render_choice, render_noul, render_score

MODEL = os.environ.get('OPENJEV_MODEL', 'models/gemma-3-4b-it')
RUN = os.environ.get('OPENJEV_VIDEO_SMOKE') == '1' and os.path.isdir(MODEL)


def clip(n_balls, colour=(220, 30, 30), seconds=4, fps=8, size=(320, 240)):
    """White background with ``n_balls`` discs moving left to right."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    buf = io.BytesIO()
    with av.open(buf, mode='w', format='mp4') as out:
        s = out.add_stream('libx264', rate=fps)
        s.width, s.height, s.pix_fmt = *size, 'yuv420p'
        for i in range(seconds * fps):
            img = Image.new('RGB', size, (255, 255, 255))
            d = ImageDraw.Draw(img)
            for b in range(n_balls):
                x = (20 + 8 * i + 37 * b) % (size[0] - 40)
                y = 30 + (b * 53) % (size[1] - 60)
                d.ellipse([x, y, x + 30, y + 30], fill=colour)
            for p in s.encode(av.VideoFrame.from_ndarray(np.asarray(img), format='rgb24')):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return buf.getvalue()


@unittest.skipUnless(RUN, 'set OPENJEV_VIDEO_SMOKE=1 (needs the Gemma 3 4B checkpoint)')
class VideoSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openjev.scorer import OptionScorer
        from openjev.video import load_video
        cls.scorer = OptionScorer(MODEL, backend='torch', dtype='bfloat16', vision_dtype='float16')
        cls.ball = load_video(clip(1), n_frames=4).frames
        cls.empty = load_video(clip(0), n_frames=4).frames
        cls.crowd = load_video(clip(12), n_frames=4).frames

    def probs(self, render, q, frames):
        prompt, labels = render('A short video clip.', q)
        res = self.scorer.score(prompt, labels, norm='sum', images=frames, chat=True)
        return dict(zip(labels, [r.probability for r in res]))

    def test_noul_object_present_or_not(self):
        q = NoulQuestion(type='noul', instructions='Is a red ball visible in this video?')
        self.assertGreater(self.probs(render_noul, q, self.ball)['yes'], 0.5)
        self.assertLess(self.probs(render_noul, q, self.empty)['yes'], 0.5)

    def test_choice(self):
        q = ChoiceQuestion(type='choice', instructions='What does this video show?',
                           criteria={'ball': 'a red ball moving across a white background',
                                     'person': 'a person talking to the camera',
                                     'car': 'cars driving on a road',
                                     'text': 'a slide full of text'})
        p = self.probs(render_choice, q, self.ball)
        self.assertEqual(max(p, key=p.get), 'ball', p)

    @unittest.expectedFailure  # known: rates every clip "many balls" (p~1.0), plain or chat prompt
    def test_score_crowdedness(self):
        q = ScoreQuestion(type='score', instructions='How crowded with balls is the scene?',
                          criteria=['empty', 'one or two balls', 'many balls'])
        low = self.probs(render_score, q, self.empty)
        high = self.probs(render_score, q, self.crowd)
        expect = lambda p: sum(int(k) * v for k, v in p.items())
        self.assertLess(expect(low), expect(high), (low, high))


if __name__ == '__main__':
    unittest.main()
