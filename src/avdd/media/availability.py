"""Map probe results to modality availability (stream-level only).

AVAILABLE here means "a stream exists and the container parsed", NOT "the
modality is usable": silence, absent faces, and similar quality problems are
detected by later phases and reported as UNRELIABLE.
"""

from __future__ import annotations

from avdd.common.types import Modality, ModalityStatus
from avdd.media.probe import MediaProbe, ProbeStatus


def availability_from_probe(probe: MediaProbe) -> dict[Modality, ModalityStatus]:
    """Return per-modality stream availability for a probed file.

    Raises RuntimeError when the probe could not run (file missing, ffprobe not
    installed): that is a caller/environment problem, not a property of the media,
    and must not be reported as "corrupted" or "missing".
    """
    if probe.status in (ProbeStatus.NOT_FOUND, ProbeStatus.TOOL_MISSING):
        raise RuntimeError(f"cannot determine availability: {probe.status.value}")
    if probe.status is not ProbeStatus.OK:  # FAILED or TIMEOUT: container unreadable
        return {
            Modality.AUDIO: ModalityStatus.CORRUPTED,
            Modality.VISUAL: ModalityStatus.CORRUPTED,
        }
    return {
        Modality.AUDIO: ModalityStatus.AVAILABLE if probe.has_audio else ModalityStatus.MISSING,
        Modality.VISUAL: ModalityStatus.AVAILABLE if probe.has_video else ModalityStatus.MISSING,
    }
