from __future__ import annotations

import sys
import types

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub
