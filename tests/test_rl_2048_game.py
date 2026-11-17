import random
import unittest

from rl_2048.game import Game, apply_move, normalised_values, rollout_values, slide, valid_actions


class SlideTests(unittest.TestCase):
    def test_merges_once_per_tile(self):
        self.assertEqual(slide([2, 2, 2, 2]), ([4, 4, 0, 0], 8))
        self.assertEqual(slide([2, 2, 4, 0]), ([4, 4, 0, 0], 4))
        self.assertEqual(slide([0, 4, 0, 4]), ([8, 0, 0, 0], 8))
        self.assertEqual(slide([2, 4, 8, 16]), ([2, 4, 8, 16], 0))

    def test_directions(self):
        board = [2, 0, 0, 2,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 2, 0, 0, 0]
        left, gain, changed = apply_move(board, "left")
        self.assertEqual((left[:4], gain, changed), ([4, 0, 0, 0], 4, True))
        right, _, _ = apply_move(board, "right")
        self.assertEqual(right[:4], [0, 0, 0, 4])
        up, gain, _ = apply_move(board, "up")
        self.assertEqual((up[0], up[12], gain), (4, 0, 4))
        down, _, _ = apply_move(board, "down")
        self.assertEqual((down[0], down[12]), (0, 4))

    def test_invalid_move_detection(self):
        board = [2, 4, 2, 4,
                 4, 2, 4, 2,
                 2, 4, 2, 4,
                 4, 2, 4, 2]
        self.assertEqual(valid_actions(board), [])
        corner = [2] + [0] * 15
        self.assertEqual(sorted(valid_actions(corner)), ["down", "right"])


class GameTests(unittest.TestCase):
    def test_random_game_ends_and_scores(self):
        game, rng = Game(seed=1), random.Random(1)
        while not game.over():
            game.step(rng.choice(game.valid_actions()))
        self.assertGreater(game.score, 0)
        self.assertGreaterEqual(game.max_tile, 32)

    def test_rollouts_and_normalisation(self):
        board = [2, 2, 0, 0] + [0] * 12
        values = rollout_values(board, random.Random(0), 4, 5, 256)
        self.assertTrue(all(values[a] is not None for a in ("left", "right", "down")))
        self.assertIsNone(values["up"])
        z = normalised_values(values)
        self.assertAlmostEqual(sum(z), 0.0, places=6)
        self.assertEqual(min(z), z[0])  # "up" is invalid, so it ranks last


if __name__ == "__main__":
    unittest.main()
