from .constants import dbg
# ai_detect_template.py
# -----------------------------------------------------------------------------
# TEMPLATE finger engine (Phase 1) - "constrained hand builder".
#
# A SEPARATE, additive engine. It does NOT replace or modify the NEURAL or
# GEOMETRIC paths - those stay fully active and unchanged. Selectable via the
# Finger Engine enum ('TEMPLATE').
#
# Idea (ARP-style): instead of trusting a detector to place 20 joints freely
# (which lets fingers collapse / cross / drift), we take only the RELIABLE
# evidence from an existing detector - the fingertips and the two OUTER knuckles
# (index + pinky MCP, the least-confusable ones) - and REBUILD the hand from a
# template that is valid by construction:
#
#   * the 4 finger MCPs are forced onto a constrained knuckle arc (index & pinky
#     anchored to evidence; middle & ring INTERPOLATED between them), so the
#     knuckle row is always ordered and evenly spaced - middle can't drift onto
#     ring, ring can't drift onto pinky;
#   * each finger's phalanges are laid along its own MCP->TIP direction at fixed
#     anatomical length ratios - the chain is always straight, ordered, and
#     correctly proportioned;
#   * the fingertip stays where the detector found it (the one thing detectors
#     get right), so the hand still matches THIS mesh.
#
# The operator then runs its normal cleanup passes (depth-centring, inside-snap,
# lateral straighten) which project the template onto the mesh surface. Because
# the template already guarantees which tube each joint belongs to, those passes
# can no longer cross fingers.
#
# Thumb: NOT templated in Phase 1 - the thumb is anatomically separate (thenar
# base, not a knuckle-row finger). Its evidence markers are passed through
# unchanged so we never make the thumb worse. (Phase 2 can add a thumb template.)
#
# CALIBRATION: the fractions below (_MCP_LAT_FRAC, _ARC_DEPTH, _PHALANX_FRAC) are
# reasonable anatomical defaults, NOT tuned to your hands yet. Eyeball the
# `[template]` log lines against a couple of known-good hands and adjust.
# -----------------------------------------------------------------------------

import math
from mathutils import Vector

from .ai_detect_lvt import _FINGER_TO_ARP

_FOUR = ("index", "middle", "ring", "pinky")

# Lateral position of each finger's MCP measured from the PINKY (0.0) to the
# INDEX (1.0) along the knuckle chord. index & pinky are anchored to evidence;
# middle & ring are interpolated at these fractions -> always ordered & spaced.
_MCP_LAT_FRAC = {"index": 1.0, "middle": 0.65, "ring": 0.34, "pinky": 0.0}

# Forward bump of each MCP off the index-pinky chord toward the fingertips, as a
# fraction of the chord (row) width. Models the knuckle ARC (middle protrudes
# most; pinky sits back). Gentle by default so a wrong guess can't dominate.
_ARC_DEPTH = {"index": 0.05, "middle": 0.12, "ring": 0.07, "pinky": -0.04}

# Along-finger position of PIP / DIP as a fraction of MCP->TIP (MCP=0, TIP=1).
# Anatomical proximal:middle:distal ~ 0.44 : 0.29 : 0.27.
_PHALANX_FRAC = {"phal2": 0.44, "phal3": 0.73}   # phal1=MCP(0.0), tip=1.0


def _key(arp_name, side):
    return f"{arp_name}_{side}"


def _walk_centerline(bvh, tip, mcp_guess, flen):
    """March from the trusted TIP down the finger's OWN tube by MOMENTUM and
    return (MCP, PIP, DIP) at anatomical arc positions of the walked path.

    The MCP target here comes from the TIPS-ANCHORED row (correct lane by
    construction), so the march direction is trustworthy; the per-step
    cross-section recentring only bends the path to follow the finger's
    curve. (A fully momentum-driven walk that also derived the MCP from the
    walk end was tried and REVERTED: cross-section centring hops lanes on
    chubby/webbed meshes and L/R walked differently.) Returns None when the
    walk doesn't lock onto a tube; the caller keeps the straight chain then."""
    from .ai_detect_lvt import _local_cross_section_center
    n = 14
    step = flen / n
    sr = min(0.035, max(0.010, 0.30 * flen))
    cur = Vector(tip)
    dirv = Vector(mcp_guess) - cur
    if dirv.length < 1e-6:
        return None
    dirv = dirv.normalized()
    pts = [cur.copy()]
    centred = 0
    for _i in range(2 * n):
        d = Vector(mcp_guess) - cur
        rem = d.length
        if rem <= step * 1.25:
            break
        dirv = d / rem
        nxt = cur + dirv * step
        cen = _local_cross_section_center(nxt, dirv, bvh, search_radius=sr)
        # Reject a far-away "centre" (rays caught a neighbouring finger);
        # blend partially so mesh noise doesn't zigzag the path.
        if cen is not None and (cen - nxt).length < 0.75 * step:
            nxt = nxt + (cen - nxt) * 0.7
            centred += 1
        pts.append(nxt.copy())
        cur = nxt
    pts.append(Vector(mcp_guess))
    if centred < max(2, (len(pts) - 2) // 4):
        return None                # didn't lock onto a tube
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    arc = cum[-1]
    if arc < 1e-6:
        return None

    def _at(s):
        s = max(0.0, min(s, arc))
        for i in range(1, len(cum)):
            if cum[i] >= s:
                seg = cum[i] - cum[i - 1]
                t = (s - cum[i - 1]) / seg if seg > 1e-12 else 0.0
                return pts[i - 1] + (pts[i] - pts[i - 1]) * t
        return pts[-1]

    # _PHALANX_FRAC measures from the MCP end; the walked arc runs TIP -> MCP.
    pip = _at(arc * (1.0 - _PHALANX_FRAC["phal2"]))
    dip = _at(arc * (1.0 - _PHALANX_FRAC["phal3"]))
    return pip, dip


def _relabel_by_identity(ev, side, hw):
    """Correct mis-labelled evidence via the SHARED finger-identity module, so a
    template built on a mis-identified detection (esp. the geometric fallback,
    which swaps thumb/finger labels on straight hands) gets the right fingers.

    Returns a possibly-relabelled {finger: {'mcp','tip'}} dict - unchanged when
    there isn't a full 5-finger set, the module can't decide, or the labels
    already agree (the common neural case -> no-op)."""
    from .finger_identity import assign_finger_identity
    items = []
    for f in ("thumb",) + _FOUR:
        mcp, tip = ev[f]["mcp"], ev[f]["tip"]
        if mcp is not None and tip is not None:
            items.append({"tip": tip, "base": mcp, "g": None, "ref": f})
    # COMPRESSED evidence guard: a finger whose chain is bunched at the tip
    # (base ~ tip, the truncated-tube hand) gives the identity module a
    # degenerate base -- its votes (off-row, base-proximity) become noise and
    # can relabel a hand that only needed rebuilding. Judge scale from the tip
    # spread itself so this works at any character size.
    if len(items) >= 2:
        spread = max((a["tip"] - b["tip"]).length
                     for a in items for b in items)
        items = [it for it in items
                 if (it["tip"] - it["base"]).length > 0.10 * spread]
    if len(items) < 5:
        return ev
    tip_c = sum((it["tip"] for it in items), Vector()) / len(items)
    named = assign_finger_identity(items, side, hw, tip_c - hw,
                                   tag=f"[template {side}]")
    if named is None:
        return ev
    remap = {canon: named[canon]["ref"] for canon in ("thumb",) + _FOUR}
    if all(remap[c] == c for c in remap):
        return ev                       # already correct -> no change
    dbg(f"[template {side}] identity RELABEL: "
          + ", ".join(f"{c}<-{remap[c]}" for c in remap if remap[c] != c))
    return {c: ev[remap[c]] for c in ("thumb",) + _FOUR}


def _evidence_points(evidence, side):
    """Pull {finger: {'mcp','p2','p3','tip'}} (Vectors or None) from a
    detector's marker dict (ARP-keyed, e.g. FINGER_INDEX_1_L)."""
    out = {}
    for f in ("thumb",) + _FOUR:
        arp = _FINGER_TO_ARP[f]
        get = evidence.get if evidence else (lambda _k: None)
        out[f] = {"mcp": get(_key(arp["phal1"], side)),
                  "p2":  get(_key(arp["phal2"], side)),
                  "p3":  get(_key(arp["phal3"], side)),
                  "tip": get(_key(arp["tip"],   side))}
    return out


def detect_fingers_template(mesh_obj, hw, ew, side, evidence, verbose=True):
    """Rebuild a valid hand from an existing detector's evidence.

    mesh_obj : hand mesh (unused in Phase 1 build - the operator's cleanup does
               the mesh projection; kept in the signature for Phase 2).
    hw, ew   : HAND (wrist) and ELBOW marker positions (Vectors).
    side     : 'L' or 'R'.
    evidence : ARP-keyed marker dict from NEURAL/GEOMETRIC detection for THIS
               side (its tips + outer MCPs are the only inputs trusted).

    Returns an ARP-keyed marker dict for this side, or None if there isn't
    enough evidence (caller then keeps the original detector result).
    """
    if not evidence:
        if verbose:
            dbg(f"  [template {side}]: no evidence - declined")
        return None

    ev = _evidence_points(evidence, side)
    ev = _relabel_by_identity(ev, side, hw)     # shared identity: fix mislabeled evidence
    tips = {f: ev[f]["tip"] for f in _FOUR if ev[f]["tip"] is not None}
    if len(tips) < 3:
        if verbose:
            dbg(f"  [template {side}]: only {len(tips)} tips - declined")
        return None

    # -- Anatomical length expectations (from tips + wrist) --------------------
    # hand_len = wrist -> farthest fingertip. Expected finger lengths as loose
    # anthropometric fractions of it (middle ~0.46, pinky ~0.33). Used ONLY to
    # judge whether evidence is PLAUSIBLE (wide 0.45x-2.0x acceptance band, so
    # stylised chibi/cartoon proportions pass) and to rebuild what clearly
    # isn't -- never to override plausible evidence.
    hw_v = Vector(hw)
    hand_len = max((t - hw_v).length for t in tips.values())
    _EXP_RATIO = {"index": 0.42, "middle": 0.46, "ring": 0.43, "pinky": 0.33}
    exp = {f: _EXP_RATIO[f] * hand_len for f in _FOUR}

    def _anchor(f):
        """Evidence MCP for an outer finger -- or an anatomical reconstruction
        when the evidence chain is clearly compressed (all joints bunched at
        the fingertip, the truncated-tube hand) or the MCP is missing."""
        m, tip = ev[f]["mcp"], tips.get(f)
        if tip is None:
            return Vector(m) if m is not None else None    # can't judge
        if m is not None and 0.45 * exp[f] < (tip - m).length < 2.0 * exp[f]:
            return Vector(m)                                # plausible evidence
        d = tip - hw_v
        if d.length < 1e-4:
            return Vector(m) if m is not None else None
        if verbose:
            why = ("missing" if m is None else
                   f"implausible ({(tip - m).length*1000:.0f}mm vs "
                   f"expected ~{exp[f]*1000:.0f}mm)")
            dbg(f"  [template {side}]: {f} MCP evidence {why} -- "
                  f"rebuilt from tip + anatomical length")
        return tip - d.normalized() * exp[f]

    i_mcp = _anchor("index")
    p_mcp = _anchor("pinky")
    if i_mcp is None or p_mcp is None or (i_mcp - p_mcp).length < 1e-4:
        if verbose:
            dbg(f"  [template {side}]: missing/degenerate index|pinky MCP - declined")
        return None

    # -- Palm frame -----------------------------------------------------------
    tip_centroid = sum(tips.values(), Vector()) / len(tips)
    chord   = i_mcp - p_mcp                       # index<-pinky lateral chord
    row_w   = chord.length
    fwd_raw = tip_centroid - hw_v
    fwd_raw = fwd_raw.normalized() if fwd_raw.length > 1e-4 \
        else (hw_v - Vector(ew)).normalized() if ew is not None else Vector((0, 0, 1))
    # ACROSS is LEVELED: strip the forward component from the chord. The two
    # anchors sit at different depths along the fingers (the pinky knuckle is
    # proximal of the index one; rebuilt anchors use differing anatomical
    # lengths), so the raw chord TILTS toward the tips -- and a tilted lane
    # axis inflates every tip's across-offset and false-fires the drift
    # straightening on perfectly straight fingers.
    across = chord - fwd_raw * chord.dot(fwd_raw)
    across = across.normalized() if across.length > 1e-4 else chord.normalized()
    # forward, re-orthogonalised so the arc bump is purely "toward the tips".
    forward = (fwd_raw - across * fwd_raw.dot(across))
    forward = forward.normalized() if forward.length > 1e-4 else fwd_raw

    # -- Constrained MCP row --------------------------------------------------
    # index & pinky anchored to evidence; middle & ring interpolated + arc bump.
    mcp = {"index": Vector(i_mcp), "pinky": Vector(p_mcp)}
    for f in ("middle", "ring"):
        base = p_mcp.lerp(i_mcp, _MCP_LAT_FRAC[f])          # on the chord
        mcp[f] = base + forward * (_ARC_DEPTH[f] * row_w)   # bumped onto the arc
    # index & pinky also get a (small) arc bump so the row is a smooth curve.
    mcp["index"] = mcp["index"] + forward * (_ARC_DEPTH["index"] * row_w)
    mcp["pinky"] = mcp["pinky"] + forward * (_ARC_DEPTH["pinky"] * row_w)

    # -- Row RE-CENTRING onto the tips ----------------------------------------
    # On the kid/chubby hand class the TIPS are right but the MCP evidence is
    # laterally biased, and anchoring the row on it shifted EVERY knuckle one
    # lane over (index MCP/PIP toward middle ... ring toward pinky, tips
    # fine). Full tips-anchored rows over-correct splayed hands (broke the
    # approved long-nail row) and mesh walks hop lanes on chubby meshes - so
    # correct only the DOF that is demonstrably wrong: translate the whole
    # row along the across-axis so its lateral centre matches the TIPS'
    # centre. Gated: rows already centred under the tips (all approved hands)
    # move 0mm and are untouched.
    if all(f in tips for f in _FOUR):
        a_cen_tips = sum(tips[f].dot(across) for f in _FOUR) / 4.0
        a_cen_row  = sum(mcp[f].dot(across) for f in _FOUR) / 4.0
        shift = a_cen_tips - a_cen_row
        if abs(shift) > 0.10 * row_w:
            for f in _FOUR:
                mcp[f] = mcp[f] + across * shift
            if verbose:
                dbg(f"  [template {side}]: knuckle row re-centred "
                      f"{shift*1000:+.0f}mm onto the fingertips (evidence row "
                      f"was laterally biased)")

    # -- EVIDENCE MCP passthrough (ALL four fingers) --------------------------
    # The row construction guarantees ORDER/SPACING but pins the knuckles to
    # the straight index-pinky chord: middle/ring are interpolated onto it,
    # and even the index/pinky anchors get a heuristic forward arc bump (or a
    # full anatomical rebuild when a strong CURL shrinks their tip-to-knuckle
    # straight-line distance under the length gate). On a curved hand the
    # kept PIP/DIP/TIP follow the curl and the MCPs visibly "don't adjust
    # with the rest". Keep the detector's MCP when it is plausibly that
    # finger's knuckle - sane length to its own tip and laterally inside its
    # own row slot (DEPTH is deliberately free: that's the adjustment). The
    # chord/across/row structure above still comes from the validated
    # anchors; only the final knuckle POSITIONS follow the evidence.
    kept_mcp = []
    for f in _FOUR:
        e_m, t_f = ev[f]["mcp"], tips.get(f)
        if e_m is None or t_f is None:
            continue
        if not (0.45 * exp[f] < (t_f - Vector(e_m)).length < 2.0 * exp[f]):
            continue
        gaps_f = [(mcp[f] - mcp[g]).length for g in _FOUR if g != f]
        if gaps_f and abs((Vector(e_m) - mcp[f]).dot(across)) \
                < 0.55 * min(gaps_f):
            mcp[f] = Vector(e_m)
            kept_mcp.append(f)
    if kept_mcp and verbose:
        dbg(f"  [template {side}]: evidence MCP kept for "
              f"{','.join(kept_mcp)} (depth follows the detector)")

    # Median finger length (fallback for a finger whose tip is missing).
    lens = [(tips[f] - mcp[f]).length for f in _FOUR if f in tips]
    med_len = sorted(lens)[len(lens) // 2] if lens else row_w

    # -- Tip-lane validation --------------------------------------------------
    # The MCP row is now regularised, but the template still trusts the detector's
    # TIP for each finger's DIRECTION. If a tip has drifted onto a NEIGHBOUR (the
    # classic middle->ring / ring->pinky failure), the chain still leans over that
    # neighbour even with a correct MCP. Detect it the same way as the knuckle row:
    # a tip whose across-coord sits nearer a DIFFERENT finger's MCP lane than its
    # own has crossed lanes -> it's drifted. Rebuild that finger STRAIGHT along the
    # hand forward (keeps its length, drops the bad direction). Edge fingers fan
    # OUTWARD (away from neighbours) so they never trip this; only a genuine
    # cross-lane drift fires, so natural splay on index/pinky is preserved.
    a_mcp = {f: mcp[f].dot(across) for f in _FOUR}

    def _tip_drifted(f, tip):
        a_t = tip.dot(across)
        own = abs(a_t - a_mcp[f])
        others = [g for g in _FOUR if g != f and abs(a_t - a_mcp[g]) < own]
        if not others:
            return False
        # Natural SPLAY also leans a straight finger into a neighbour's lane
        # (long-nail hand: the ring tip at -25mm of a ~31mm MCP gap got
        # straightened, and the rebuilt chain drifted onto the MIDDLE finger).
        # A leaning tip is NOT a drifted tip: a genuinely drifted tip has
        # LANDED ON the neighbour, so confirm by tip-to-tip proximity before
        # overriding the detector's direction (trust the model - no lateral
        # manipulation without positive evidence).
        for g in others:
            nt = tips.get(g)
            if nt is not None and \
                    (tip - nt).length < 0.45 * (mcp[f] - mcp[g]).length:
                return True
        return False

    # -- Build each finger chain ---------------------------------------------
    # PIP/DIP prefer the mesh-anchored CENTERLINE WALK over the straight
    # chord: the chord is only correct on straight, flat-splayed fingers.
    bvh = None
    if mesh_obj is not None:
        from .ai_detect_lvt import _build_bvh
        bvh = _build_bvh(mesh_obj)
    out = {}
    straightened = []
    walked = []
    kept = []                                      # evidence PIP/DIP passed gates
    tip_off = {}                                   # lateral offset of tip vs MCP (diag)
    for f in _FOUR:
        arp = _FINGER_TO_ARP[f]
        m   = mcp[f]
        tip = tips.get(f)
        if tip is not None and (tip - m).length > 1e-4:
            flen = (tip - m).length
            tip_off[f] = (tip - m).dot(across)      # signed across offset (diag)
            if _tip_drifted(f, tip):
                # Tip crossed into a neighbour lane -> straighten this finger.
                fdir = forward
                tip  = m + fdir * flen
                straightened.append(f)
            else:
                fdir = (tip - m).normalized()
        else:
            fdir = forward
            flen = med_len
            tip  = m + fdir * flen
        p2 = m + fdir * (_PHALANX_FRAC["phal2"] * flen)
        p3 = m + fdir * (_PHALANX_FRAC["phal3"] * flen)
        # EVIDENCE PIP/DIP passthrough (validation-gated): the detector's
        # interior joints carry the finger's CURL, which the straight chord
        # throws away - the template used to lose to plain NEURAL on curved
        # fingers because it rebuilt a curled chain as a straight one (joints
        # off-centre / pressed under the palmar surface). Keep the evidence
        # joints when they are plausibly THIS finger's: ordered along the
        # chord at sane fractions, and bounded in the ACROSS direction (curl
        # bends dorsal/palmar; cross-finger drift is lateral). Rebuild on the
        # chord otherwise - and always when this finger's tip was
        # straightened (the whole evidence chain is suspect then).
        if f not in straightened and tips.get(f) is not None:
            e2, e3 = ev[f].get("p2"), ev[f].get("p3")
            if e2 is not None and e3 is not None and flen > 1e-5:
                t2 = (Vector(e2) - m).dot(fdir) / flen
                t3 = (Vector(e3) - m).dot(fdir) / flen
                lat2 = abs((Vector(e2) - m).dot(across)
                           - (Vector(tip) - m).dot(across) * t2)
                lat3 = abs((Vector(e3) - m).dot(across)
                           - (Vector(tip) - m).dot(across) * t3)
                gaps_f = [(mcp[f] - mcp[g]).length
                          for g in _FOUR if g != f]
                lat_cap = min(0.22 * flen, 0.55 * min(gaps_f)) if gaps_f \
                    else 0.22 * flen
                if (0.15 <= t2 <= 0.75 and t2 + 0.05 <= t3 <= 0.95
                        and lat2 < lat_cap and lat3 < lat_cap):
                    p2, p3 = Vector(e2), Vector(e3)
                    kept.append(f)
        # Centerline walk DISABLED: its cross-section recentring catches the
        # neighbour finger in webbed/pressed-together regions and drags the
        # PIP into the neighbour's lane (lane report: Hand_A L index PIP
        # -13mm into the middle lane, L/R asymmetric; Kid PIPs -4..-5mm with
        # lanes only ~8mm apart). A chord PIP/DIP is laterally IN-LANE by
        # construction (lerp of an in-lane MCP and an in-lane tip); depth
        # centring is the [depth] pass's job and is depth-only. Keep the walk
        # code for potential future use on clean meshes.
        if False and bvh is not None:
            w = _walk_centerline(bvh, tip, m, flen)
            if w is not None:
                p2, p3 = w
                walked.append(f)
        out[_key(arp["phal1"], side)] = m
        out[_key(arp["phal2"], side)] = p2
        out[_key(arp["phal3"], side)] = p3
        out[_key(arp["tip"],   side)] = tip

    # -- Thumb: pass PLAUSIBLE evidence through unchanged; template only a
    # clearly COMPRESSED chain (Phase 2, validation-gated). THUMB_1 is the CMC
    # near the wrist, so on every real hand the CMC->tip evidence span is a
    # large fraction of the wrist->thumbtip distance; joints bunched at the
    # fingertip (the truncated-tube hand) give a tiny fraction instead. Only
    # then rebuild the chain along wrist->thumbtip at anatomical fractions
    # (CMC 0.30 / MCP 0.58 / IP 0.80), keeping the evidence tip -- a good
    # thumb is never made worse.
    thumb = _FINGER_TO_ARP["thumb"]
    t_tip = ev["thumb"]["tip"]
    t_cmc = ev["thumb"]["mcp"]          # phal1 = THUMB_1 = CMC
    rebuilt_thumb = False
    if t_tip is not None:
        span = (t_tip - hw_v).length
        ch = (t_tip - t_cmc).length if t_cmc is not None else 0.0
        if span > 1e-4 and ch < 0.45 * span:
            d = (t_tip - hw_v).normalized()
            out[_key(thumb["phal1"], side)] = hw_v + d * (0.30 * span)
            out[_key(thumb["phal2"], side)] = hw_v + d * (0.58 * span)
            out[_key(thumb["phal3"], side)] = hw_v + d * (0.80 * span)
            out[_key(thumb["tip"],   side)] = Vector(t_tip)
            rebuilt_thumb = True
            if verbose:
                dbg(f"  [template {side}]: thumb chain compressed "
                      f"(CMC->tip {ch*1000:.0f}mm vs wrist->tip "
                      f"{span*1000:.0f}mm) -- rebuilt along wrist->thumbtip")
    if not rebuilt_thumb:
        for part in ("phal1", "phal2", "phal3", "tip"):
            k = _key(thumb[part], side)
            if evidence.get(k) is not None:
                out[k] = evidence[k]

    if verbose:
        gaps = [ (mcp["index"] - mcp["middle"]).length,
                 (mcp["middle"] - mcp["ring"]).length,
                 (mcp["ring"] - mcp["pinky"]).length ]
        # tip lateral offset from own MCP, index->pinky order (mm, +ve = toward index)
        offs = '/'.join('%+.0f' % (tip_off.get(f, 0.0) * 1000) for f in _FOUR)
        dbg(f"  [template {side}]: rebuilt from {len(tips)}/4 tips  "
              f"row_w={row_w*1000:.0f}mm  MCP gaps="
              f"{'/'.join('%.0f' % (g*1000) for g in gaps)}mm  "
              f"lens={'/'.join('%.0f' % ((tips[f]-mcp[f]).length*1000) for f in _FOUR if f in tips)}mm")
        dbg(f"  [template {side}]: tip across-offset (i/m/r/p) = {offs}mm"
              + (f"  STRAIGHTENED {','.join(straightened)}" if straightened else "")
              + (f"  curve-walked {len(walked)}/4" if walked else "")
              + (f"  evidence PIP/DIP kept: {','.join(kept)}" if kept else ""))
    return out
