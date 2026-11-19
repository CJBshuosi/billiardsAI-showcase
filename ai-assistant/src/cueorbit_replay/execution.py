"""Small shared execution primitives; replay and rules retain their policies."""

import time

from .contracts import Usage


class Budget:
    def __init__(self, total_s: float, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + total_s

    def remaining(self, cap: float) -> float:
        value = min(cap, self.deadline - self.clock())
        if value <= 0:
            raise TimeoutError("REQUEST_DEADLINE")
        return value


class UsageTotals:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def begin(self):
        self.inputs.append(None)
        self.outputs.append(None)

    def complete(self, usage):
        self.inputs[-1] = usage.input_tokens
        self.outputs[-1] = usage.output_tokens

    def snapshot(self) -> Usage:
        return Usage(
            input_tokens=sum(self.inputs)
            if self.inputs and all(v is not None for v in self.inputs)
            else None,
            output_tokens=sum(self.outputs)
            if self.outputs and all(v is not None for v in self.outputs)
            else None,
        )
