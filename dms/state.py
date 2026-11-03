"""Human-readable driver state for the overlay.

The condition signals stay precise and per-behaviour; this layer collapses
them into the single sentence the person watching the feed reads first.
Priority follows danger: sleeping beats yawning beats drowsiness, and any
active distraction outranks a merely low EAR.
"""

from __future__ import annotations

import math

from .alerts import ConditionSignal, Severity
from .face import FaceObservation

# BGR, bright enough to read on the dark overlay panel.
_ATTENTIVE = (80, 220, 80)
_INFO = (205, 205, 205)
_MINOR = (120, 210, 255)
_MAJOR = (60, 170, 255)
_CRITICAL = (70, 70, 255)


def summarize(
    face: FaceObservation,
    signals: list[ConditionSignal],
) -> tuple[str, tuple[int, int, int]]:
    """(text, BGR colour) for the driver's current, human-readable state.

    The preview shows one short word first. The precise condition (PERCLOS
    percentage, eyes-closed seconds, phone-at-ear seconds) stays in the
    active-message lines at the bottom of the frame.
    """
    active = {s.key: s for s in signals if s.active}

    if not face.present:
        return "No driver", _INFO

    if "drowsy_microsleep" in active:
        return "Sleeping", _CRITICAL
    if "drowsy_perclos" in active:
        return "Drowsy", _MAJOR
    if "drowsy_yawning" in active:
        return "Drowsy", _MINOR

    # A phone that is merely visible (MINOR) does not outrank a glance; a
    # phone at the ear or used while looking away (MAJOR/CRITICAL) does.
    phone = active.get("phone_use")
    if phone is not None and phone.severity is not Severity.MINOR:
        return "Distracted - phone", _MAJOR

    if "gaze_off_road" in active:
        return "Distracted - looking away", _MAJOR

    # Drinking/eating are reported in the active-message lines but do not
    # override the primary state label: a water bottle in hand used to flip
    # the whole preview to "Drinking" and confused the drowsiness readout.
    if "smoking" in active:
        return "Smoking", _MAJOR

    return "Attentive", _ATTENTIVE
