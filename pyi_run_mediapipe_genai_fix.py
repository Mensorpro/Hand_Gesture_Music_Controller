"""PyInstaller runtime hook: provide a stub for mediapipe.tasks.python.genai.

This inserts an empty module into sys.modules before the application imports
`mediapipe`, preventing circular import / partially-initialized import errors
when MediaPipe's package `__init__` tries `from . import genai`.
"""
import sys
import types

MODULE_NAME = "mediapipe.tasks.python.genai"

if MODULE_NAME not in sys.modules:
    sys.modules[MODULE_NAME] = types.ModuleType(MODULE_NAME)
