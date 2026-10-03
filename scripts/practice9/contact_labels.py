"""Conservative HumanML3D foot phase labels; unknown frames carry zero weight."""
import numpy as np

VERSION = 'humanml263_contact4_with_geometry_mask_v1'

def infer(joints, fps=20.0):
    a = np.asarray(joints, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (22, 3) or len(a) < 9 or not np.isfinite(a).all():
        raise ValueError('Expected finite [T>=9,22,3]')
    # Lowest ankle/toe marker; per-foot offset absorbs marker/limb asymmetry.
    h = np.stack([a[:, [7, 10], 1].min(1), a[:, [8, 11], 1].min(1)], 1)
    h -= np.quantile(h, .05, axis=0)
    vz = np.gradient(h, 1/fps, axis=0)
    stance = (h <= .025) & (np.abs(vz) <= .15)
    swing = h >= .05
    known = stance | swing
    # No unsupported double-flight assumptions for scoped walk/stand data.
    known[np.min(h, axis=1) > .10] = False
    # Mask transitions, uncertain phases and endpoints rather than force a label.
    for foot in range(2):
        edges = np.flatnonzero(stance[1:, foot] != stance[:-1, foot]) + 1
        for t in edges:
            known[max(0,t-1):min(len(a),t+2), foot] = False
    known[:2] = False
    known[-2:] = False
    return stance.astype(np.uint8), known.astype(np.uint8), h

def resample(labels, known, output_frames):
    if output_frames < 2:
        raise ValueError('At least two output frames')
    # Native preserves source time span; final stretch uses normalized phase.
    phase = np.linspace(0, len(labels)-1, output_frames)
    lo = np.floor(phase).astype(int)
    hi = np.ceil(phase).astype(int)
    mask = known[lo] & known[hi] & (labels[lo] == labels[hi])
    return labels[lo], mask.astype(np.uint8)


def from_features(features, joints):
    """Preserve official-layout contact bits; geometry only masks disagreements."""
    a = np.asarray(features)
    if a.shape != (len(joints),263) or not np.isfinite(a).all():
        raise ValueError('Expected aligned, finite, unnormalized [T,263] features')
    raw = a[:, -4:]
    if not np.isin(raw, [0.,1.]).all():
        raise ValueError('Contact bits must be binary; do not threshold normalized features')
    raw = raw.astype(np.uint8)
    contact = np.stack([raw[:,:2].any(1),raw[:,2:].any(1)],axis=1).astype(np.uint8)
    geometric, known, height = infer(joints)
    known &= (contact == geometric)
    # Preserve source transitions as unknown even when geometric phase is stable.
    for foot in range(2):
        for t in np.flatnonzero(contact[1:,foot] != contact[:-1,foot])+1:
            known[max(0,t-1):min(len(joints),t+2),foot] = False
    return contact, known, height, raw
