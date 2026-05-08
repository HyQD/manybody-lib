import sys
import types
from pathlib import Path

# Make cis_functions and cisd_functions importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "closed_shell" / "cis"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "closed_shell" / "cisd"))

# Provide a minimal stub for the 'utils' module (Counter is only used in CI_step)
utils_stub = types.ModuleType("utils")


class Counter:
    def __init__(self):
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1


utils_stub.Counter = Counter
sys.modules.setdefault("utils", utils_stub)
