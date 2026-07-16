# finger_identity.py
# -----------------------------------------------------------------------------
# Shared, deterministic FINGER IDENTITY assignment.
#
# The problem it fixes: every finger engine was re-deriving "which tube/tip is
# thumb/index/middle/ring/pinky" from cues that don't hold on real hands -
# collinearity of a LINE fit (4 fingers aren't collinear) and LENGTH ("middle is
# longest", "index longer than pinky"). On straight/gapless hands those misfire
# and shift EVERY label, and they even resolve DIFFERENTLY on mirror-identical L
# vs R hands (why the symmetry pass kept mirroring). See the geo/template logs.
#
# This module assigns identity from signals that ARE reliable and LENGTH-FREE:
#   * THUMB = the one item that is simultaneously (a) OFF the knuckle row, while
#     the other four ARE collinear; (b) the most PROXIMAL (lowest geodesic reach,
#     or base nearest the wrist when geodesics aren't available); (c) the most
#     ISOLATED tip (the four fingers are adjacent; the thumb tip stands alone).
#     Four independent votes - far more stable than a single line residual.
#   * The other four are ordered along the knuckle row, oriented so INDEX is the
#     row end nearest the thumb (anatomical: the thumb is adjacent to the index).
#     No length is used to decide identity or orientation - length is only ever a
#     soft warning in the caller.
#
# Pure numpy + plain dicts so BOTH the geometric engine and the template engine
# (and anything else) can call it, and it's unit-testable without Blender.
# -----------------------------------------------------------------------------

import numpy as np

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _line_fit(pts):
    """SVD line fit -> (centroid, unit dir, RMS perpendicular residual)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    perp = (pts - c) - np.outer((pts - c) @ d, d)
    resid = float(np.sqrt((perp ** 2).sum(axis=1).mean()))
    return c, d, resid


def _norm01(a):
    a = np.asarray(a, dtype=float)
    rng = float(a.max() - a.min())
    return (a - a.min()) / rng if rng > 1e-12 else np.zeros_like(a)


def assign_finger_identity(items, arp_side, wrist, forward, tag=None):
    """Label exactly five finger items as thumb/index/middle/ring/pinky.

    items : list of >=5 dicts, each with
              'tip'  : (3,) world coord of the fingertip
              'base' : (3,) world coord of the palm entry / MCP
              'g'    : geodesic reach from the wrist (optional; lower = thumb-ish)
              'ref'  : anything the caller wants back (the source tube/label)
            The caller SELECTS the five (this module does not); only the first
            five are used if more are passed.
    arp_side : 'L' or 'R' (used only for logging / future handedness checks).
    wrist    : (3,) wrist position (proximity fallback when 'g' is absent).
    forward  : (3,) wrist->fingers direction (reserved; not required here).
    tag      : optional log prefix; prints a one-line identity summary if given.

    Returns {finger: item} for all five, or None if it can't decide.
    """
    if items is None or len(items) < 5:
        return None
    items = list(items)[:5]

    tips  = np.array([np.asarray(it["tip"],  dtype=float) for it in items])
    bases = np.array([np.asarray(it["base"], dtype=float) for it in items])
    wrist = np.asarray(wrist, dtype=float)

    gs = [it.get("g") for it in items]
    have_g = all(g is not None for g in gs)
    prox = (np.array([float(g) for g in gs]) if have_g
            else np.linalg.norm(bases - wrist, axis=1))   # lower = more thumb-like
    prox_n = _norm01(prox)                                 # 0 = most proximal

    # Base proximity to the wrist: the thumb's BASE (thenar/CMC) sits proximal,
    # near the wrist, while a finger's base is out on the knuckle ROW. This is the
    # clean discriminator between the real thumb and a short pinky-side STUB, which
    # off-row + tip-isolation alone couldn't tell apart (the stub got labelled
    # thumb, so the thumb marker landed on the pinky). Separate from `prox` above,
    # which for the geo path is TIP geodesic reach (a pinky stub's tip reach can
    # match the thumb's; its BASE position does not).
    base_wrist  = np.linalg.norm(bases - wrist, axis=1)
    base_prox_n = _norm01(base_wrist)                      # 0 = base closest to wrist

    # Tip isolation: distance to the NEAREST other tip (thumb tip stands alone).
    iso = np.array([min(np.linalg.norm(tips[i] - tips[j])
                        for j in range(5) if j != i) for i in range(5)])
    iso_n = _norm01(iso)

    # -- Thumb = best combined vote ------------------------------------------
    best_i, best_score, diag = None, -1e18, {}
    for i in range(5):
        others = np.array([bases[j] for j in range(5) if j != i])
        c, d, resid = _line_fit(others)
        row_len = float(np.ptp((others - c) @ d))
        if row_len < 1e-9:
            continue
        off = float(np.linalg.norm((bases[i] - c) - ((bases[i] - c) @ d) * d))
        off_n   = off / row_len            # thumb sits OFF the row of the other 4
        col_pen = resid / row_len          # the other 4 should be collinear (low)
        score = (1.0 * off_n) + (0.8 * (1.0 - prox_n[i])) \
                + (0.5 * iso_n[i]) + (0.7 * (1.0 - base_prox_n[i])) \
                - (1.0 * col_pen)
        diag[i] = score
        if score > best_score:
            best_score, best_i = score, i
    if best_i is None:
        return None

    # -- Order the remaining four along the knuckle row ----------------------
    rest = [j for j in range(5) if j != best_i]
    rc, rd, _ = _line_fit(np.array([bases[j] for j in rest]))
    rest.sort(key=lambda j: float((bases[j] - rc) @ rd))
    # Orient: INDEX is the row end nearest the thumb base (thumb ~ index side).
    tb = bases[best_i]
    if np.linalg.norm(bases[rest[-1]] - tb) < np.linalg.norm(bases[rest[0]] - tb):
        rest.reverse()

    # Orientation SANITY (length used ONLY to orient an already-ordered row, never
    # to pick identity). The pinky is the shortest finger and the index is clearly
    # longer than it on EVERY hand - unlike "middle is longest", this is ironclad.
    # If nearest-thumb put the longer end at "pinky", the orientation is reversed
    # -> flip. 0.88 hysteresis leaves genuinely equal-length stylised hands to the
    # structural (nearest-thumb) signal. Uses tip->base length from the items.
    def _flen(j):
        return float(np.linalg.norm(tips[j] - bases[j]))
    if _flen(rest[0]) < _flen(rest[-1]) * 0.88:
        rest.reverse()
        if tag:
            print(f"{tag} row re-oriented: index end shorter than pinky end "
                  f"({_flen(rest[-1])*1000:.0f}mm < {_flen(rest[0])*1000:.0f}mm)")

    named = {"thumb": items[best_i],
             "index": items[rest[0]], "middle": items[rest[1]],
             "ring":  items[rest[2]], "pinky":  items[rest[3]]}
    # Thumb-vote confidence, exposed for callers (weak margin = the hand parse
    # is ambiguous; the geo engine uses it to gate its multi-scale retry).
    margin = best_score - max((s for i, s in diag.items() if i != best_i),
                              default=best_score)
    named["_margin"] = margin

    if tag:
        print(f"{tag} identity: thumb={items[best_i].get('ref')}  "
              f"score={best_score:.2f} margin={margin:.2f}  "
              f"(g={'yes' if have_g else 'no'})")
    return named
