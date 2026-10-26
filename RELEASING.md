# Releasing to PyPI

1. **Name.** The distribution is `openjev-sdk` (`openjev` on PyPI belongs to an unrelated
   project). The import name (`import openjev`) and the `openjev` command are unchanged.
2. **Bump `version`** in `pyproject.toml`.
3. **Build from a clean checkout**, never from a working tree. Hatch packages everything
   under `openjev/`, including untracked folders.

   ```bash
   git clone --depth 1 https://github.com/daseinlabs/open-jev /tmp/release && cd /tmp/release
   python -m venv .build && .build/bin/pip install build twine
   .build/bin/python -m build
   .build/bin/twine check dist/*
   unzip -l dist/*.whl | grep openjev/     # expect only tracked modules
   ```
4. **Try it from TestPyPI first**:

   ```bash
   .build/bin/twine upload --repository testpypi dist/*
   python -m venv /tmp/try && /tmp/try/bin/pip install --extra-index-url https://test.pypi.org/simple/ openjev-sdk
   /tmp/try/bin/python -m unittest discover -s tests -p 'test_tod.py'
   ```
5. **Upload**: `.build/bin/twine upload dist/*` (needs a PyPI API token).

Training and inference run on PyTorch (CUDA, ROCm, MPS, CPU) or MLX (Apple silicon). Before
a release, run cookbook 01 on both if you can; CUDA has only been checked through the shared
PyTorch path on MPS.
