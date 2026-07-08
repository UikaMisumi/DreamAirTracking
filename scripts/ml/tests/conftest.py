"""Pytest bootstrap for the runtime script tests.

Adds ``scripts/ml`` to ``sys.path`` so tests can import runtime modules
(``predict_live_multitask``, etc.) directly without packaging.
"""

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
