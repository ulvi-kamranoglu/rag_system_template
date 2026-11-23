"""Bounded streaming look-behind for pathological literal generation loops."""
import re


class RepetitionGuard:
    # Keep enough text to detect six copies even when units cross stream chunks.
    LOOKBEHIND = 480
    PATTERN = re.compile(r"(?P<unit>.{3,80}?)(?P=unit){5,}", re.DOTALL)

    def __init__(self):
        self.pending = ""
        self.stopped = False

    def feed(self, text):
        if self.stopped:
            return ""
        self.pending += text
        for match in self.PATTERN.finditer(self.pending):
            unit = match.group("unit")
            if any(char.isalpha() for char in unit):
                # Preserve one occurrence, suppress the loop and stop upstream.
                output = self.pending[:match.start() + len(unit)]
                self.pending = ""
                self.stopped = True
                return output
        if len(self.pending) <= self.LOOKBEHIND:
            return ""
        output = self.pending[:-self.LOOKBEHIND]
        self.pending = self.pending[-self.LOOKBEHIND:]
        return output

    def finish(self):
        output, self.pending = self.pending, ""
        return output
