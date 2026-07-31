import os
import sys
from types import ModuleType

import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.zeros(8)
        return np.zeros((len(texts), 8))


if "sentence_transformers" not in sys.modules:
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_module
