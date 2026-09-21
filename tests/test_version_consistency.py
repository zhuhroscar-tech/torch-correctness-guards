from importlib.metadata import version

import torch_correctness_guards


def test_package_version_matches_metadata():
    assert torch_correctness_guards.__version__ == version("torch-correctness-guards")
