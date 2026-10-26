import os
import tempfile
import unittest
from unittest.mock import patch

from openjev import backbone


class BackboneTests(unittest.TestCase):
    def test_local_directory_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as d, patch('huggingface_hub.snapshot_download') as download:
            self.assertEqual(backbone.resolve(d), d)
            download.assert_not_called()

    def test_repo_ids_download_only_fingerprinted_files(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('huggingface_hub.snapshot_download', return_value='/cache/snap') as download:
            self.assertEqual(backbone.resolve(), '/cache/snap')
            self.assertEqual(backbone.resolve('org/other'), '/cache/snap')
        self.assertEqual(download.call_args_list[0].args, (backbone.DEFAULT_REPO,))
        self.assertEqual(download.call_args.kwargs['allow_patterns'], backbone.PATTERNS)

    def test_environment_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'OPENJEV_MODEL': d}):
            self.assertEqual(backbone.resolve(), d)
        for bad in ('/no/such/dir', './missing', 'a/b/c'):
            with self.assertRaisesRegex(ValueError, 'no such backbone directory'):
                backbone.resolve(bad)
        with patch('huggingface_hub.snapshot_download', side_effect=OSError('401')):
            with self.assertRaisesRegex(ValueError, 'hf auth login'):
                backbone.resolve('org/gated')


if __name__ == '__main__':
    unittest.main()
