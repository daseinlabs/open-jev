"""Frame sampling and video input checks on synthetic clips (no downloads, no model)."""
import base64
import io
import os
import tempfile
import unittest

try:
    import av
    import numpy as np
except ImportError:  # pragma: no cover
    av = None

from openjev.video import (VideoError, VideoLimits, decode_frames, load_video,
                           read_base64, read_source, sample_timestamps)

BITS, BLOCK = 12, 8


def make_clip(n_frames, fps, gop=250, codec='libx264'):
    """Encode a clip whose frame i shows i in binary as black/white blocks."""
    buf = io.BytesIO()
    with av.open(buf, mode='w', format='mp4') as out:
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = BITS * BLOCK, BLOCK * 2, 'yuv420p'
        stream.gop_size = gop
        stream.options = {'crf': '10'} if codec == 'libx264' else {}
        for i in range(n_frames):
            img = np.zeros((BLOCK * 2, BITS * BLOCK, 3), np.uint8)
            for b in range(BITS):
                if i >> b & 1:
                    img[:, b * BLOCK:(b + 1) * BLOCK] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def frame_index(img):
    a = np.asarray(img.convert('L'), dtype=float)
    return sum(1 << b for b in range(BITS)
               if a[2:-2, b * BLOCK + 2:(b + 1) * BLOCK - 2].mean() > 127)


@unittest.skipIf(av is None, 'PyAV not installed (pip install -e ".[video]")')
class FrameSamplingTests(unittest.TestCase):
    def check(self, n_video, fps, n, **kw):
        video = decode_frames(make_clip(n_video, fps, **kw), n_frames=n)
        duration = n_video / fps
        self.assertAlmostEqual(video.duration_s, duration, delta=1.0 / fps)
        self.assertEqual(len(video.frames), n)
        self.assertEqual(len(video.timestamps), n)
        for i, t in enumerate(video.timestamps):
            self.assertAlmostEqual(t, (i + 0.5) * video.duration_s / n, places=6)
        got = [frame_index(f) for f in video.frames]
        want = [min(n_video - 1, int(t * fps + 1e-9)) for t in video.timestamps]
        self.assertEqual(got, want)
        for idx, ft in zip(got, video.meta['frame_times']):
            self.assertAlmostEqual(ft, idx / fps, delta=1e-3)
        return video

    def test_short_video_more_samples_than_frames(self):
        video = self.check(10, 10, 16)  # 1 s, 10 frames, 16 samples -> repeats
        self.assertEqual(len({id(f) for f in video.frames}), 10)

    def test_short_video_default_n(self):
        self.check(48, 24, 16)

    def test_long_video_seeks_across_gops(self):
        self.check(3000, 25, 16, gop=50)   # 120 s

    def test_n_values(self):
        clip_kw = dict(n_video=600, fps=30, gop=30)
        for n in (1, 4, 64):
            with self.subTest(n=n):
                self.check(n=n, **clip_kw)

    def test_sample_timestamps(self):
        self.assertEqual(sample_timestamps(8.0, 4), [1.0, 3.0, 5.0, 7.0])
        with self.assertRaises(VideoError):
            sample_timestamps(8.0, 0)

    def test_max_side_downscale(self):
        video = decode_frames(make_clip(10, 10), n_frames=2, max_side=48)
        self.assertEqual(max(video.frames[0].size), 48)

    def test_sources_and_hash(self):
        data = make_clip(20, 10)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(data)
        try:
            a = load_video(f.name, n_frames=4)
            b = load_video('data:video/mp4;base64,' + base64.b64encode(data).decode(), n_frames=4)
            c = load_video(data, n_frames=4)
            self.assertEqual(a.sha256, b.sha256)
            self.assertEqual(a.sha256, c.sha256)
            self.assertEqual(read_base64(base64.b64encode(data).decode()), data)
        finally:
            os.unlink(f.name)

    def test_errors(self):
        data = make_clip(50, 10)  # 5 s
        cases = [
            (lambda: read_source('/no/such/video.mp4'), 'not found'),
            (lambda: read_source('ftp://x/y.mp4'), 'not found'),
            (lambda: read_base64('not base64!!'), 'base64'),
            (lambda: read_base64('data:video/mp4,abc'), 'base64'),
            (lambda: decode_frames(b''), 'empty'),
            (lambda: decode_frames(b'not a video at all' * 10), 'could not open'),
            (lambda: decode_frames(data, limits=VideoLimits(max_duration_s=2)), 'limit'),
            (lambda: decode_frames(data, n_frames=65), 'exceeds'),
            (lambda: read_base64(base64.b64encode(data).decode(), VideoLimits(max_bytes=10)), 'limit'),
        ]
        for fn, msg in cases:
            with self.subTest(msg=msg), self.assertRaises(VideoError) as ctx:
                fn()
            self.assertIn(msg, str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
