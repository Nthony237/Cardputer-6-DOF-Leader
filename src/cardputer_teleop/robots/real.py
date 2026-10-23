"""Real YAM arm driver (Phase 5). Not implemented yet, on purpose.

This will wrap the i2rt driver running on the Jetson. Until it exists, ``--mode real`` fails
loudly instead of silently doing nothing. Requirements before filling this in:
heartbeat watchdog that holds the arm, deadman key, the same speed/workspace limits as sim,
a hardware e-stop within reach, and supervised testing only.
"""
from __future__ import annotations


class YamRobot:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Real YAM control is Phase 5 and is not implemented yet. Use --mode sim."
        )
