import sys
import types

# Provide a minimal stub for the 'utils' module (Counter is only used in CI_step)
utils_stub = types.ModuleType("utils")


class Counter:
    def __init__(self):
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1


utils_stub.Counter = Counter
sys.modules.setdefault("utils", utils_stub)
