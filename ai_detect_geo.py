"""
Geometric (no-ML) finger detection — geodesic-tube engine.

Works purely from the mesh: no renders, no onnxruntime. Selected via the
finger_detection_engine scene property (GEOMETRIC) in the Finger Markers panel.

Method (few stages, all thresholds relative to the hand's own geometry):
  1. Isolate the hand (two passes — see detect_fingers_geo).
  2. Geodesic distance field g() over the mesh edge graph, multi-source seeded
     from a proximal SLAB (not a single rim vertex) so a wrist cut that lands
     slightly into the palm still produces a stable field.
  3. Fingertips = geodesic maxima of g with geodesic non-max suppression.
     Touching-but-unwelded fingers don't shortcut (no shared edges).
  4. Per-finger tube march from each tip: iso-distance band centroids trace the
     medial axis (inside the volume by construction); the band radius measures
     that finger's own thickness. MCP = where the radius jumps (palm entry),
     relative to the finger's own base radius — thin/fat/thick fingers alike.
  5. PIP/DIP at anatomical fractions of the measured medial ARC length, so
     curled fingers are followed, not straight-lined.
  6. The thumb runs the SAME pipeline; identity is assigned afterwards:
     thumb = the candidate whose exclusion leaves the other four palm-entry
     points most collinear (the knuckle row); index = the row end nearest the
     thumb. No world-axis or handedness assumptions.

Fails closed: fewer than 5 validated finger tubes -> returns None so the1
caller can fall back instead of guessing.
"""

import bpy
from .constants import dbg
import heapq
import numpy as np
from mathutils import Vector

_INF = float("inf")

# Last accepted hand-plausibility per side, for the operator's confidence gate.
# NOTE: a WEAK absolute signal (the ranking terms inflate it), so it only catches
# a hand where NO set looked like a hand. The per-invariant checks in
# finger_quality_report (thumb-too-long, MCP order, tips merged) are the primary
# protection; this floor is a backstop for degenerate parses.
_LAST_PLAUSIBILITY = {}
# 3.3, not the original 2.5: the thumb-straightness term (+0.8 for a straight
# thumb) shifted every score up by ~0.8 (good hands now run 4.2-4.5), so the
# floor moves with it to keep the same effective strictness.
_PLAUSIBILITY_FLOOR = 3.3


def _dijkstra(adj, seeds, cap=_INF):
    """Multi-source Dijkstra over the vertex adjacency list, truncated at cap."""
    dist = {s: 0.0 for s in seeds}
    pq = [(0.0, s) for s in seeds]
    heapq.heapify(pq)
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist.get(v, _INF):
            continue
        for o, w in adj[v]:
            nd = d + w
            if nd <= cap and nd < dist.get(o, _INF):
                dist[o] = nd
                heapq.heappush(pq, (nd, o))
    return dist


def _sample_path(pts, cum, s):
    """Point at arc length s along polyline pts (cum = cumulative lengths)."""
    s = max(0.0, min(s, cum[-1]))
    for i in range(1, len(cum)):
        if cum[i] >= s:
            seg = cum[i] - cum[i - 1]
            t = (s - cum[i - 1]) / seg if seg > 1e-12 else 0.0
            return pts[i - 1] + (pts[i] - pts[i - 1]) * t
    return pts[-1]


def _tube_from_tip(adj, co, tip_idx, g_max, median_edge, g):
    """
    March iso-distance bands from a tip candidate. Returns dict with the medial
    path (tip -> palm entry), arc length, and base radius — or None if the
    candidate isn't finger-shaped (never enters a widening palm, too short,
    or not elongated).
    g_max here is the HAND-length reference (hand_ref), not the raw field max:
    coefficients are calibrated for it (ref ~= 0.65x of the old stub-inflated
    g_max, so 0.85/0.0155/0.09 reproduce the old 0.55/0.010/0.06 behaviour).
    """
    cap = 0.85 * g_max
    d = _dijkstra(adj, [tip_idx], cap=cap)
    w = max(2.0 * median_edge, 0.0155 * g_max)
    nb = max(4, int(cap / w))
    bands = [[] for _ in range(nb)]
    for v, dv in d.items():
        k = int(dv / w)
        if k < nb:
            bands[k].append(v)

    cents, radii = [], []
    for k in range(nb):
        if not bands[k]:
            break
        pts = co[bands[k]]
        c = pts.mean(axis=0)
        cents.append(c)
        if len(bands[k]) >= 2:
            radii.append(float(np.linalg.norm(pts - c, axis=1).mean()))
        else:
            # Single-vert band on a low-poly finger: carry the previous radius
            # for continuity instead of aborting the march (a break here made
            # REAL fingertips fail validation on sparse meshes).
            radii.append(radii[-1] if radii else 0.0)
    if len(cents) < 4:
        return None

    # Base radius from the first bands past the tip cap; the knuckle/palm entry
    # is the first SUSTAINED jump above it (2 consecutive bands, or the last).
    base = float(np.median(radii[1:4]))
    if base < 1e-9:
        return None
    k_end = None
    for k in range(3, len(radii)):
        if radii[k] > 1.75 * base:
            nxt_wide = (k + 1 >= len(radii)) or (radii[k + 1] > 1.60 * base)
            if nxt_wide:
                k_end = k
                break
    if k_end is None:
        return None                      # never widened into a palm: not a finger
    # Honest tube length = the CENTROID arc, not bands-marched * band-width.
    # When the geodesic front bleeds past the web into the PALM, the band
    # centroids stall and bunch up (successive steps shrink), so k_end * w
    # overestimates the length and a fat palm blob (radius ~ its own length)
    # scrapes past this gate -- it then wins the thumb slot and puts the whole
    # thumb chain inside the palm (Cartoon hand: arc 74mm, base_r 59mm, cum
    # steps 33/19/10/11mm). A real finger's centroids advance ~w per band, so
    # for it this is the same number.
    tube_len = sum(float(np.linalg.norm(np.asarray(cents[i + 1])
                                        - np.asarray(cents[i])))
                   for i in range(1, k_end - 1)) + w
    if tube_len < 2.2 * base or tube_len < 0.09 * g_max:
        return None                      # stubby bump, not an elongated finger
                                         # (2.2x, not 2.5x: chubby baby/toon
                                         # fingers are nearly as thick as long)

    # Fingertip = centroid of the distal CAP: verts whose seed-field g is within
    # one band of the tip's. The single argmax vertex lands wherever the meshing
    # put it (often the SIDE of a rounded tip); the seed field flows up through
    # the finger symmetrically, so its distal level set is centred on the tip.
    # Band is 1.5w deep so a coarse low-poly tip still collects verts from ALL
    # sides of the fingertip -- a 1-2 vert cap re-creates the sideways drift.
    g_tip = g.get(tip_idx, 0.0)
    cap_verts = [v for v, dv in d.items()
                 if dv <= 3.0 * w and g.get(v, -_INF) >= g_tip - 1.5 * w]
    if len(cap_verts) >= 3:
        cpts = co[cap_verts]
        tip_pos = cpts.mean(axis=0)
        # The cap centroid sits ~w PROXIMAL of the apex ("marker on the
        # finger, not the tip"). Project the cap onto the MEDIAL-AXIS distal
        # direction (cents[1]->cents[3] reversed — stable on curled and
        # splayed fingers, unlike a centroid-to-band chord, and unlike the
        # geodesic argmax, which lands off-apex on curls because the dorsal
        # surface path is longer) and take the centroid of the verts in the
        # last half-band: laterally centred AND at the fingertip apex.
        if len(cents) >= 4:
            a_dir = np.asarray(cents[1]) - np.asarray(cents[3])
            dl = float(np.linalg.norm(a_dir))
            if dl > 1e-9:
                a_dir = a_dir / dl
                proj = (cpts - tip_pos) @ a_dir
                apex = cpts[proj >= float(proj.max()) - 0.5 * w]
                if len(apex):
                    tip_pos = apex.mean(axis=0)
    else:
        tip_pos = co[tip_idx]

    # Medial path: tip cap centroid + band centroids, lightly smoothed
    # (endpoints kept). Stop BEFORE the radius-jump band: its centroid is
    # contaminated by palm / neighbour-finger verts and bends every finger's
    # proximal direction toward the palm centre (MCPs pinch together).
    raw = [tip_pos] + cents[1:k_end]
    path = [raw[0]]
    for i in range(1, len(raw) - 1):
        path.append((raw[i - 1] + raw[i] + raw[i + 1]) / 3.0)
    path.append(raw[-1])
    path = [np.asarray(p, dtype=np.float64) for p in path]
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + float(np.linalg.norm(path[i] - path[i - 1])))
    if cum[-1] < 1e-9:
        return None
    return {"tip_idx": tip_idx, "path": path, "cum": cum,
            "arc": cum[-1], "base_r": base}


def _drop_overlapping(tubes, arp_side):
    """
    A bump/fold on a finger can validate as its own small tube RIDING that
    finger's volume (seen as thumb markers landing on the middle finger). A
    real finger never runs inside another finger, so drop the shorter of any
    two tubes whose medial paths COINCIDE. Coincide is the key word: a rider's
    bands wrap the host finger itself, so beyond its bump the centroids land ON
    the host's axis (distance ~0). Real fingers -- even thick ones pressed
    together -- keep their axes about one radius-SUM apart, several times the
    half-radius threshold here (a 0.6x radius-sum test dropped real fingers on
    a chunky low-poly hand).
    """
    tubes = sorted(tubes, key=lambda t: t["arc"], reverse=True)
    kept = []
    for t in tubes:
        pts = np.array(t["path"])
        dup = None
        for k in kept:
            kp = np.array(k["path"])
            dmin = np.sqrt(((pts[:, None, :] - kp[None, :, :]) ** 2).sum(-1)).min(axis=1)
            near = dmin < 0.5 * max(t["base_r"], k["base_r"])
            if near.mean() < 0.45:
                continue                       # shafts don't overlap
            # Overlapping shaft. Distinguish a RIDER / SHELL-DUPLICATE (drop) from
            # a TOUCHING NEIGHBOUR FINGER (keep). A real neighbour has its own
            # knuckle AND its own fingertip: even fingers pressed flesh-to-flesh
            # keep base and tip centres about a radius-SUM apart. A rider's base
            # coincides with the host's base; a DOUBLE-SHELL duplicate (second
            # surface of the same finger) traces the SAME fingertip, so its TIP
            # coincides (~one shell offset) even when its base clears the base
            # test. Requiring BOTH separations fixes the double-shell regression
            # (dupes kept -> selection picked a dupe -> MCP row out of order)
            # while still keeping genuinely touching neighbours.
            base_sep = float(np.linalg.norm(pts[-1] - kp[-1]))
            tip_sep  = float(np.linalg.norm(pts[0] - kp[0]))
            _sep_ref = 0.6 * (t["base_r"] + k["base_r"])
            if base_sep < _sep_ref or tip_sep < _sep_ref:
                dup = k
                break              # base OR tip coincides -> rider / shell dupe
            # else: own knuckle + own tip -> touching neighbour finger, keep both
        if dup is None:
            kept.append(t)
        else:
            dbg(f"[geo {arp_side}] dropped {t['arc']*1000:.0f}mm tube riding "
                  f"the {dup['arc']*1000:.0f}mm tube's volume")
    return kept


def _pick_five(tubes, g_max, top_k=16, hand_ref=None):
    """
    Return up to top_k structurally-plausible 5-tube SETS (each a list of 5
    tubes), best structural score first, for the identity + hand-plausibility
    step to choose among.

    Structure ALONE is not enough: a clean collinear row of SHORT BUMPS
    out-scores the real (longer, slightly-off-row) fingers on row geometry, so
    _pick_five kept selecting the wrong five (ignored 112/114mm tubes, grabbed
    54/64mm bumps). We therefore hand the caller several passing sets and let it
    re-rank by how anatomical the LABELLED hand looks (see _hand_plausibility).

    Per-set gates (evaluated at the best thumb position):
      - the four finger palm entries roughly collinear (knuckle row) and spread;
      - one entry clearly OFF the row near a row END (thumb), not a nub.
    Returns [] when no set passes.
    """
    from itertools import combinations
    scored = []
    for sub in combinations(range(len(tubes)), 5):
        cand = [tubes[i] for i in sub]
        g_sum = sum(t["g"] for t in cand)
        # A low-reach candidate (admitted below the normal tip floor to catch a
        # DOWN-pointing thumb) may only occupy the THUMB slot: its low reach is
        # exactly why it was excluded as a finger, and letting palm bumps from
        # the lowered floor compete as row fingers would pollute selection.
        low = [j for j, t in enumerate(cand) if t.get("low_reach")]
        if len(low) > 1:
            continue
        best_score = None
        for ti in (low if low else range(5)):
            rest = [cand[j] for j in range(5) if j != ti]
            E = np.array([t["path"][-1] for t in rest])
            c = E.mean(axis=0)
            _, _, vt = np.linalg.svd(E - c)
            d = vt[0]
            proj = np.sort((E - c) @ d)
            row_len = float(proj[-1] - proj[0])
            if row_len < 1e-6:
                continue
            perp = (E - c) - np.outer((E - c) @ d, d)
            resid = float(np.sqrt((perp ** 2).sum(axis=1).mean()))
            if resid > 0.22 * row_len:
                continue                    # knuckle row not collinear
            if float(np.diff(proj).min()) < 0.15 * row_len:
                continue                    # two "fingers" share one knuckle
            tv = np.asarray(cand[ti]["path"][-1]) - c
            tp = float(tv @ d)
            toff = float(np.linalg.norm(tv - tp * d))
            if toff < 0.10 * row_len:
                continue                    # thumb entry sits ON the row
            if proj[0] + 0.30 * row_len < tp < proj[-1] - 0.30 * row_len:
                continue                    # thumb hangs over mid-row
            max_rest = max(t2["arc"] for t2 in rest)
            if max_rest > 1e-9 and cand[ti]["arc"] / max_rest < 0.35:
                continue                    # nub can't be a thumb
            rest_arcs = sorted(t2["arc"] for t2 in rest)
            if rest_arcs[0] < 0.45 * float(np.median(rest_arcs)):
                continue                    # bump posing as a row finger
            # POSITIONAL gate: the four MCP/web bases all sit at about the same
            # geodesic DEPTH (the knuckle row). A tube whose path dives across
            # the PALM interior ends much deeper -- its base geodesic (tip g
            # minus arc; the tube marches down iso-g bands, so arc ~ geodesic
            # descent) is far below its siblings' -- even when its base happens
            # to LINE UP with the row in 3D (the Cartoon-hand bug: a
            # finger-length palm tube out-scored the real short pinky on the
            # length priors). Reject the set structurally instead of piling on
            # more scalar priors.
            if hand_ref is not None:
                bd = [t2["g"] - t2["arc"] for t2 in rest]
                if max(bd) - min(bd) > 0.30 * hand_ref:
                    continue                # a "finger" base dives into the palm
            score = resid / row_len - 0.15 * (g_sum / (5.0 * g_max))
            if best_score is None or score < best_score:
                best_score = score
        if best_score is not None:
            scored.append((best_score, cand))
    scored.sort(key=lambda x: x[0])
    return [cand for _, cand in scored[:top_k]]


def _hand_plausibility(ident, g_max):
    """Rank candidate 5-tube SETS by how anatomical the LABELLED hand looks.

    Length is used to rank SETS, NEVER to label fingers within a set (that stays
    structural, in finger_identity). Real fingers: middle near-longest, lengths
    consistent (bumps mixed with fingers blow the min/max ratio up), reaching far.
    Higher = more hand-like.
    """
    four = ("index", "middle", "ring", "pinky")
    L = [ident[f]["ref"]["arc"] for f in four]
    mx = max(L)
    if mx < 1e-9:
        return -1e18
    s_consist = min(L) / mx                                   # bumps -> low
    s_mid     = ident["middle"]["ref"]["arc"] / mx            # middle near-longest
    s_reach   = (sum(L) / 4.0) / g_max if g_max > 1e-9 else 0.0
    # STRAIGHTNESS: a real rest-pose finger runs roughly straight (arc ~ tip->base
    # span). A BENT tube whose chain curls toward a neighbour reads as a finger by
    # LENGTH but lands its markers on the wrong finger (the "pinky is on the middle
    # /ring" bug: a bent 90mm tube beat the straight pinky on length-consistency
    # alone). Prefer the SET whose finger tubes are straighter. Relative across
    # sets of the SAME hand, so a genuinely curled hand (all tubes bent) isn't
    # penalised - it just can't do better.
    def _straightness(t):
        p, a = t["path"], t["arc"]
        if a < 1e-9:
            return 1.0
        return float(np.linalg.norm(np.asarray(p[0]) - np.asarray(p[-1]))) / a
    s_straight = sum(_straightness(ident[f]["ref"]) for f in four) / 4.0
    # THUMB LENGTH sanity: the thumb is one of the SHORTEST digits. A "thumb" as
    # long as the fingers is a mislabeled finger (a long bent tube whose curl put
    # its base near the wrist and fooled the thumb pick), and "index = nearest
    # thumb" then ROTATES the whole hand. Plausibility otherwise scores only the
    # four fingers, so a set with a bogus long thumb wins as easily as the real
    # one. Penalize a thumb approaching the finger length. Full credit below 0.75x
    # the longest finger (real thumbs ~0.55-0.70x); 0 credit at >=1.0x.
    t_ratio = ident["thumb"]["ref"]["arc"] / mx
    s_thumb  = max(0.0, 1.0 - max(0.0, t_ratio - 0.75) / 0.25)
    # THUMB STRAIGHTNESS: s_straight above scores only the four fingers, so a
    # folded thenar/palm tube (straightness ~0.5) could win the thumb slot
    # over the real, straighter thumb on length terms alone (Adam: tip marker
    # at the thumb base, chain folded into the palm). Every confirmed-good
    # thumb pick runs 0.75-1.0. Relative across sets of the same hand, so a
    # genuinely bent thumb (all thumb candidates bent) isn't penalised.
    s_tstraight = _straightness(ident["thumb"]["ref"])
    return (s_consist + 0.6 * s_mid + 0.6 * s_reach
            + 1.2 * s_straight + 1.0 * s_thumb + 0.8 * s_tstraight)


def body_extremities_geodesic(mesh_obj, max_verts=12000):
    """
    Head / hands / feet (+ wrist, elbow, ankle) estimated from GEODESIC
    EXTREMITIES: the farthest-point-sampling extremities of a humanoid surface
    are the head, two hands and two feet — for ANY pose (T, A, arms-down) and
    ANY proportions. Replaces the fixed-height lateral-edge template that
    assumed a T-pose (wrists found at 78% body height parked the arm markers
    on the SHOULDERS for arms-down characters).

    Wrist/elbow/ankle come from the finger-engine iso-band trick: the centroid
    of the surface band at hand-length geodesic distance from the fingertip IS
    the wrist ring's centre — inside the arm, pose-independent.

    Returns {'head','hand_l','hand_r','wrist_l',...} world Vectors, or None
    (caller keeps its template) when extremities can't be classified.
    """
    ob = None
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = mesh_obj.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
        ob = bpy.data.objects.new("_er_bodygeo", me)
        bpy.context.collection.objects.link(ob)
        ob.matrix_world = mesh_obj.matrix_world.copy()
        if len(me.vertices) > max_verts:
            mod = ob.modifiers.new("dec", 'DECIMATE')
            mod.ratio = max_verts / len(me.vertices)
            bpy.context.view_layer.update()
            dg2 = bpy.context.evaluated_depsgraph_get()
            ev2 = ob.evaluated_get(dg2)
            me2 = bpy.data.meshes.new_from_object(ev2, depsgraph=dg2)
            old = ob.data
            ob.data = me2
            bpy.data.meshes.remove(old)
            me = me2

        n = len(me.vertices)
        if n < 100 or len(me.edges) < 100:
            return None
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(ob.matrix_world, dtype=np.float64)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        ev_ = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev_)
        ev_ = ev_.reshape(-1, 2)
        el = np.linalg.norm(co[ev_[:, 0]] - co[ev_[:, 1]], axis=1)
        adj = [[] for _ in range(n)]
        for (a, b), l in zip(ev_.tolist(), el.tolist()):
            adj[a].append((b, l))
            adj[b].append((a, l))

        h = float(co[:, 2].max() - co[:, 2].min())
        if h < 1e-4:
            return None
        cx = float((co[:, 0].max() + co[:, 0].min()) * 0.5)

        # Largest connected piece = the body.
        seen = np.zeros(n, dtype=bool)
        best_comp = []
        for i0 in range(n):
            if seen[i0]:
                continue
            stack, comp = [i0], [i0]
            seen[i0] = True
            while stack:
                v = stack.pop()
                for o, _w in adj[v]:
                    if not seen[o]:
                        seen[o] = True
                        comp.append(o)
                        stack.append(o)
            if len(comp) > len(best_comp):
                best_comp = comp
        comp_set = best_comp
        cen_np = co[comp_set].mean(axis=0)
        v0 = min(comp_set,
                 key=lambda i: float(((co[i] - cen_np) ** 2).sum()))

        # Farthest-point sampling: 6 geodesic extremities.
        dmin = _dijkstra(adj, [v0])
        pts = []
        for _k in range(6):
            far = max(dmin, key=dmin.get)
            pts.append(far)
            d_new = _dijkstra(adj, [far])
            for v, dv in d_new.items():
                if dv < dmin.get(v, _INF):
                    dmin[v] = dv

        # Classify: head = highest, feet = two lowest, hands = the remaining
        # pair with opposite X signs (a tail/prop extremity fails that test).
        zs = [(i, float(co[i][2])) for i in pts]
        head_i = max(zs, key=lambda t: t[1])[0]
        rest = [i for i in pts if i != head_i]
        rest.sort(key=lambda i: float(co[i][2]))
        feet = rest[:2]
        cand = rest[2:]
        hands = None
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                xa = float(co[cand[a]][0]) - cx
                xb = float(co[cand[b]][0]) - cx
                if xa * xb < 0:
                    hands = (cand[a], cand[b])
                    break
            if hands:
                break
        if hands is None or float(co[feet[0]][0] - cx) * float(co[feet[1]][0] - cx) > 0:
            dbg("[body-geo] extremities unclassifiable -- template keeps arms")
            return None

        def _lr(pair):
            a, b = pair
            return (a, b) if co[a][0] >= cx else (b, a)

        hand_l, hand_r = _lr(hands)
        foot_l, foot_r = _lr(tuple(feet))

        med_e = float(np.median(el))

        def _band_centroid(tip, lo, hi):
            d = _dijkstra(adj, [tip], cap=hi * 1.05)
            band = [v for v, dv in d.items() if lo <= dv <= hi]
            if len(band) < 3:
                return None
            p = co[band].mean(axis=0)
            return Vector((float(p[0]), float(p[1]), float(p[2])))

        def _arm_chain(tip):
            """Wrist/elbow/shoulder from the arm's RADIUS PROFILE, not fixed
            height fractions (a fixed 0.11h wrist landed on the FOREARM of
            big-handed characters). Wrist = first band-radius local minimum
            past the palm (the wrist narrowing); shoulder = where the radius
            explodes into the torso (armpit entry — the finger-MCP trick at
            body scale); elbow = surface midpoint of the two. Pose-proof."""
            cap = 0.60 * h
            d = _dijkstra(adj, [tip], cap=cap)
            w = max(2.0 * med_e, 0.010 * h)
            nb = max(6, int(cap / w))
            bands = [[] for _ in range(nb)]
            for v, dv in d.items():
                k = int(dv / w)
                if k < nb:
                    bands[k].append(v)
            cents, radii = [], []
            for k in range(nb):
                if len(bands[k]) < 2:
                    break
                pts = co[bands[k]]
                c = pts.mean(axis=0)
                cents.append(c)
                radii.append(float(np.linalg.norm(pts - c, axis=1).mean()))
            if len(cents) < 6:
                return None, None, None
            k_lo = max(3, int(0.05 * h / w))
            k_hi = min(len(radii) - 2, int(0.20 * h / w))
            k_w = None
            for k in range(k_lo, k_hi):
                if radii[k] <= radii[k - 1] and radii[k] < radii[k + 1]:
                    k_w = k
                    break                      # wrist narrowing
            if k_w is None:
                k_w = min(int(0.11 * h / w), len(cents) - 1)
            k_s = None
            for k in range(k_w + 3, len(radii)):
                med_r = float(np.median(radii[k_w:k]))
                if med_r > 1e-9 and radii[k] > 2.0 * med_r and (
                        k + 1 >= len(radii) or radii[k + 1] > 1.8 * med_r):
                    k_s = k
                    break                      # torso explosion = armpit
            V = lambda p: Vector((float(p[0]), float(p[1]), float(p[2])))
            wrist = V(cents[k_w])
            if k_s is None:
                return wrist, None, None       # arm never widened: keep template
            shoulder = V(cents[max(k_s - 1, k_w + 1)])
            elbow = V(cents[(k_w + k_s) // 2])
            return wrist, elbow, shoulder

        def _v(i):
            return Vector((float(co[i][0]), float(co[i][1]), float(co[i][2])))

        out = {"head": _v(head_i),
               "hand_l": _v(hand_l), "hand_r": _v(hand_r),
               "foot_l": _v(foot_l), "foot_r": _v(foot_r)}
        for side, hnd, ft in (("l", hand_l, foot_l), ("r", hand_r, foot_r)):
            wr, e, sh = _arm_chain(hnd)
            a = _band_centroid(ft, h * 0.055, h * 0.085)
            if wr is not None:
                out[f"wrist_{side}"] = wr
            if e is not None:
                out[f"elbow_{side}"] = e
            if sh is not None:
                out[f"shoulder_{side}"] = sh
            if a is not None:
                out[f"ankle_{side}"] = a
        dbg(f"[body-geo] extremities: head z={out['head'].z:.2f}  "
              f"hands=({out['hand_l'].x:.2f},{out['hand_r'].x:.2f})  "
              f"wrists={'yes' if 'wrist_l' in out and 'wrist_r' in out else 'partial'}  "
              f"shoulders={'yes' if 'shoulder_l' in out and 'shoulder_r' in out else 'no'}")
        return out
    except Exception as e:
        dbg(f"[body-geo] failed: {e}")
        return None
    finally:
        if ob is not None:
            try:
                _me = ob.data
                bpy.data.objects.remove(ob, do_unlink=True)
                bpy.data.meshes.remove(_me)
            except Exception:
                pass


def estimate_hand_tip_geodesic(mesh_obj, hw, ew, arp_side):
    """
    Fingertip estimate for the body-detect HAND_TIP marker: the point farthest
    from the wrist ALONG THE SURFACE (geodesic), laterally centred over the
    distal cap. Unlike the straight-line forearm projection (_estimate_hand_tip
    in ai_detect.py), this lands on the true middle fingertip on CURLED hands
    (where knuckles reach farther forward than the curled-back tips) and SPREAD
    hands (where the longest finger points off the forearm axis).
    Returns a world-space Vector, or None so the caller can fall back.
    """
    from .ai_detect_lvt import _isolate_hand_mesh

    hw_v = Vector(hw)
    ew_v = Vector(ew) if ew is not None else None
    if ew_v is None or (hw_v - ew_v).length <= 1e-4:
        return None
    arm_len = (hw_v - ew_v).length
    arm_dir = (hw_v - ew_v) / arm_len

    iso = None
    try:
        iso = _isolate_hand_mesh(mesh_obj, hw_v - arm_dir * (arm_len * 0.10),
                                 ew, tip=None, wrist_island=True)
        if iso is None:
            return None
        me = iso.data
        n = len(me.vertices)
        if n < 50 or len(me.edges) < 50:
            return None
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        ev = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        el = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1)
        median_edge = float(np.median(el))
        adj = [[] for _ in range(n)]
        for (a, b), l in zip(ev.tolist(), el.tolist()):
            adj[a].append((b, l))
            adj[b].append((a, l))

        arm_np = np.array(arm_dir, dtype=np.float64)
        s = co @ arm_np
        s_min = float(s.min())
        extent = float(s.max()) - s_min
        if extent < arm_len * 0.5:
            # Isolation kept a FRAGMENT (unwelded hand: a 318-vert wrist stub
            # was seen) — a geodesic estimate on a stub lands anywhere.
            # Fall back to the caller's projection estimate.
            dbg(f"[geo-tip {arp_side}] isolation fragment "
                  f"(extent {extent*1000:.0f}mm vs arm {arm_len*1000:.0f}mm) "
                  f"-- falling back")
            return None
        slab = max(extent * 0.06, median_edge * 2.0)
        seeds = np.nonzero(s <= s_min + slab)[0].tolist()
        g = _dijkstra(adj, seeds)
        if not g:
            return None
        tip_idx = max(g, key=g.get)
        g_max = g[tip_idx]
        if g_max < 1e-5:
            return None
        # Distal cap: verts near the max BOTH geodesically and in space (the
        # spatial bound keeps a same-length neighbouring fingertip out).
        w = max(2.0 * median_edge, 0.010 * g_max)
        t0 = co[tip_idx]
        cap = [v for v, dv in g.items()
               if dv >= g_max - 1.5 * w
               and float(np.linalg.norm(co[v] - t0)) <= 4.0 * w]
        if len(cap) < 3:
            return Vector((float(t0[0]), float(t0[1]), float(t0[2])))
        # Apex centring (same trick as the finger engine): the raw cap
        # centroid sits to the SIDE of a curled/rounded tip because the
        # geodesic argmax does. Local distal direction from a ring below the
        # cap; tip = centroid of the cap verts at the farthest level.
        p = co[cap].mean(axis=0)
        ring = [v for v, dv in g.items()
                if g_max - 4.0 * w <= dv <= g_max - 2.0 * w
                and float(np.linalg.norm(co[v] - t0)) <= 6.0 * w]
        if len(ring) >= 3:
            d_ax = p - co[ring].mean(axis=0)
            dl = float(np.linalg.norm(d_ax))
            if dl > 1e-9:
                d_ax = d_ax / dl
                cpts = co[cap]
                proj = (cpts - p) @ d_ax
                apex = cpts[proj >= float(proj.max()) - 0.5 * w]
                if len(apex):
                    p = apex.mean(axis=0)
        return Vector((float(p[0]), float(p[1]), float(p[2])))
    except Exception as _e:
        dbg(f"[geo-tip {arp_side}] geodesic hand-tip estimate failed: {_e}")
        return None
    finally:
        if iso is not None:
            try:
                _me = iso.data
                bpy.data.objects.remove(iso, do_unlink=True)
                bpy.data.meshes.remove(_me)
            except Exception:
                pass


def detect_fingers_geo(mesh_obj, hw, ew, arp_side,
                       knuckle_depth=0.22, thumb_depth=0.45, min_finger=0.30):
    """
    Geometric finger detection. Returns {marker_name_SIDE: Vector} for all 20
    finger markers, or None when no pass finds 5 valid finger tubes.
    hw/ew only steer the isolation cut and the proximal seed direction — all
    finger geometry downstream is mesh-derived (wrist-marker independent).

    Two isolation passes:
      1. ISLAND (default): weld + wrist/tip flood-fill keep. Right for most
         hands; drops slab junk (legs, props) for free.
      2. BRIDGE (retry, only when pass 1 fails): no flood-keep; disconnected
         pieces are bridged into the wrist component by synthetic edges.
         Rescues hands whose fingers are separate unwelded geometry — but on
         layered characters (armor shells, clothing) it drags in junk tubes,
         which is why it is NEVER used when the island pass succeeds.
    """
    from .ai_detect_lvt import _isolate_hand_mesh, _FINGER_TO_ARP

    hw_v = Vector(hw)
    ew_v = Vector(ew) if ew is not None else None
    if ew_v is not None and (hw_v - ew_v).length > 1e-4:
        arm_len = (hw_v - ew_v).length
        arm_dir = (hw_v - ew_v) / arm_len
    else:
        arm_dir = Vector((1.0 if arp_side == 'L' else -1.0, 0.0, 0.0))
        arm_len = 0.25

    # Back the wrist anchor off toward the elbow: a HAND marker placed slightly
    # INTO the palm (short palm) then still yields a cut on the forearm, and the
    # extra forearm stub is harmless (the seed slab + relative tube thresholds
    # below don't care how much forearm is included).
    hw_iso = hw_v - arm_dir * (arm_len * 0.10)

    # HAND_TIP marker (when plausibly placed) tightens the front cut; without it
    # the isolation falls back to a generous arm-length-based front cut.
    # Guard is generous (2.5x): a huge-handed, short-armed character (Hulk-like)
    # has a valid HAND_TIP well beyond 1.2x arm length, and rejecting it made
    # the arm-based front cut clip the fingers.
    _tip_obj = bpy.data.objects.get(f"MARKER_HAND_TIP_{arp_side}")
    tip_pt = None
    if _tip_obj:
        _tp = _tip_obj.location.copy()
        if (_tp - hw_v).length <= arm_len * 2.5:
            tip_pt = _tp

    for do_bridge in (False, True):
        iso = None
        try:
            iso = _isolate_hand_mesh(mesh_obj, hw_iso, ew, tip=tip_pt,
                                     wrist_island=not do_bridge)
            if iso is None:
                result = None
            else:
                result = _detect_from_iso(iso, hw_v, arm_dir, arm_len,
                                          arp_side, do_bridge, _FINGER_TO_ARP,
                                          knuckle_depth, thumb_depth, min_finger)
        finally:
            if iso is not None:
                try:
                    _me = iso.data
                    bpy.data.objects.remove(iso, do_unlink=True)
                    bpy.data.meshes.remove(_me)
                except Exception:
                    pass
        if result:
            return result
        if not do_bridge:
            dbg(f"[geo {arp_side}] island pass failed -- retrying with "
                  f"piece-bridging (separate-geometry hand?)")
    return None


def _detect_from_iso(iso, hw_v, arm_dir, arm_len, arp_side, do_bridge,
                     _FINGER_TO_ARP, knuckle_depth=0.22, thumb_depth=0.45,
                     min_finger=0.30):
    """One detection pass over an isolated hand mesh. Returns markers or None."""
    me = iso.data
    n = len(me.vertices)
    if n < 50 or len(me.edges) < 50:
        dbg(f"[geo {arp_side}] isolated island too small ({n} verts) -- abort")
        return None

    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)              # world space (isolation bakes matrix_world)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    el = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1)
    median_edge = float(np.median(el))
    adj = [[] for _ in range(n)]
    for (a, b), l in zip(ev.tolist(), el.tolist()):
        adj[a].append((b, l))
        adj[b].append((a, l))

    # -- Connected pieces; main = the one holding the wrist ---------------------
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i0 in range(n):
        if seen[i0]:
            continue
        stack, comp = [i0], [i0]
        seen[i0] = True
        while stack:
            v = stack.pop()
            for o, _w in adj[v]:
                if not seen[o]:
                    seen[o] = True
                    comp.append(o)
                    stack.append(o)
        comps.append(comp)
    hw_np = np.array(hw_v, dtype=np.float64)
    d2hw = [float(((co[c] - hw_np) ** 2).sum(axis=1).min()) for c in comps]
    main = comps[int(np.argmin(d2hw))]

    # -- Bridge disconnected pieces (BRIDGE pass only) ---------------------------
    # Hands modelled as separate unwelded geometry split the edge graph, and
    # the geodesic field can't enter the detached pieces. Connect each piece
    # to the wrist component with a synthetic edge when the gap is tiny
    # (touching / interpenetrating parts); pieces genuinely far away stay
    # unbridged and invisible.
    if do_bridge and len(comps) > 1:
        from mathutils import kdtree as _kdt
        others = [c for c in comps if c is not main and len(c) >= 8]
        bridge_gap = max(4.0 * median_edge, arm_len * 0.02)
        n_bridge, merged = 0, True
        while merged and others:
            merged = False
            kd = _kdt.KDTree(len(main))
            for idx in main:
                kd.insert(Vector(co[idx]), idx)
            kd.balance()
            still = []
            for c in others:
                step = max(1, len(c) // 200)
                ba, bb, bd = None, None, _INF
                for idx in c[::step]:
                    _co, mi, dist = kd.find(Vector(co[idx]))
                    if dist is not None and dist < bd:
                        ba, bb, bd = idx, mi, dist
                if bb is not None and bd <= bridge_gap:
                    w_b = max(bd, 1e-6)
                    adj[ba].append((bb, w_b))
                    adj[bb].append((ba, w_b))
                    main.extend(c)
                    n_bridge += 1
                    merged = True
                else:
                    still.append(c)
            others = still
        if n_bridge:
            dbg(f"[geo {arp_side}] bridged {n_bridge} separate piece(s) "
                  f"into the hand ({len(main)}/{n} verts reachable)")

    # -- Seed SLAB at the proximal end -------------------------------------
    # All verts within a band of the minimal along-arm projection seed at
    # distance 0. A whole cross-section slab (not one rim vertex) makes the
    # field robust to WHERE the cut landed — a diagonal cut, a cut slightly
    # into the palm, or a ragged rim all produce the same smooth front.
    # Restricted to the main component so a stray fragment can't hijack the
    # proximal extreme.
    arm_np = np.array(arm_dir, dtype=np.float64)
    midx = np.array(main, dtype=np.int64)
    s = co[midx] @ arm_np
    s_min = float(s.min())
    extent = float(s.max()) - s_min
    if extent < 1e-5:
        return None
    slab = max(extent * 0.06, median_edge * 2.0)
    seeds = midx[s <= s_min + slab].tolist()
    g = _dijkstra(adj, seeds)
    if not g:
        return None
    g_max = max(g.values())
    if g_max < 1e-5:
        return None

    # -- Hand-length reference (wrist-marker slide invariant) ----------------
    # g_max includes the forearm STUB between the cut and the wrist marker, so
    # thresholds tied to g_max changed whenever the user slid the HAND marker
    # along the forearm — the thumb (lowest-g fingertip) was the first
    # casualty of the rising candidate floor. hand_ref = g_max - g(wrist)
    # cancels the stub: sliding the marker no longer moves the thresholds.
    items = list(g.items())
    gi = np.array([i for i, _ in items], dtype=np.int64)
    gv = np.array([dv for _, dv in items], dtype=np.float64)
    d2w = ((co[gi] - hw_np) ** 2).sum(axis=1)
    ball = max(4.0 * median_edge, arm_len * 0.04)
    in_ball = d2w <= ball * ball
    g_wrist = (float(np.median(gv[in_ball])) if in_ball.any()
               else float(gv[int(np.argmin(d2w))]))
    hand_ref = min(max(g_max - g_wrist, 0.45 * g_max), g_max)
    dbg(f"[geo {arp_side}] island {n} verts, seeds {len(seeds)}, "
          f"g_max {g_max*1000:.0f}mm, hand_ref {hand_ref*1000:.0f}mm")

    # -- Tube stage (candidates -> tubes -> selection), scale-parameterized --
    # Every gate below (candidate floor, NMS radius, tube march cap, length
    # gates, band width) is proportional to href. On double-shell / layered
    # meshes the CORRECT native scale can resolve each shell as its own tube
    # (dupes everywhere, structure search fails, thumb only via the low-reach
    # rescue) while ~1.75x the scale merges the shells into clean tubes and the
    # real thumb clears the candidate floor -- the user discovered this by
    # dragging the wrist marker up the forearm (which inflates hand_ref) and
    # getting a BETTER hand than the correctly-snapped wrist. The multi-scale
    # retry below automates that workaround with the marker left in place.
    from .finger_identity import assign_finger_identity

    def _items_of(five):
        return [{"tip":  np.asarray(t["path"][0],  dtype=float),
                 "base": np.asarray(t["path"][-1], dtype=float),
                 "g":    float(t.get("g", 0.0)),
                 "ref":  t} for t in five]

    order = sorted(g, key=g.get, reverse=True)

    def _tube_stage(href):
        """One candidates -> tubes -> selection pass with all gates scaled by
        href. Returns (tier, plausibility, ident, five, fell_back, lost_tips)
        or None when no usable hand was found at this scale."""
        # -- Fingertip candidates: geodesic maxima with geodesic NMS --------
        # Coefficients recalibrated for hand_ref (~0.65x the old stub-inflated
        # g_max): 0.28/0.46 reproduce the old 0.18/0.30 on a typical hand.
        r_nms = 0.28 * href
        floor = 0.46 * href
        suppressed = set()
        cands = []
        # Cap 12, not 8: a double-shell mesh (solidify-style duplicated
        # surface) yields TWO tips per finger on disconnected shells that
        # geodesic NMS cannot suppress across -- 5 fingers then need ~10
        # slots, and an 8-slot cap crowded out a real finger before dedup
        # could pair them.
        for v in order:
            if g[v] < floor or len(cands) >= 12:
                break
            if v in suppressed:
                continue
            cands.append(v)
            suppressed |= _dijkstra(adj, [v], cap=r_nms).keys()

        # -- SECOND candidate pass at a LOWER floor: proximal (down-pointing)
        # thumb. A Disney-style thumb pointing DOWN has a LOW geodesic reach,
        # so the 0.46 floor never admits its tip as a candidate -- identity
        # then crowns a real finger as "thumb" (something must win) and
        # "index = nearest thumb" rotates the WHOLE hand (thumb label on the
        # pinky, pinky on the index). Admit a few extra low-reach candidates
        # here; they are MARKED and may only compete for the THUMB slot in
        # the set search (never as a row finger), so palm bumps admitted by
        # the lower floor cannot displace real fingers.
        low_cands = []
        low_floor = 0.24 * href
        for v in order:
            if g[v] < low_floor or len(low_cands) >= 3:
                break
            if v in suppressed or g[v] >= floor:
                continue
            low_cands.append(v)
            suppressed |= _dijkstra(adj, [v], cap=r_nms).keys()
        if low_cands:
            dbg(f"[geo {arp_side}] {len(low_cands)} low-reach thumb candidate(s) "
                  f"(g {', '.join(f'{g[v]*1000:.0f}mm' for v in low_cands)}; "
                  f"floor {floor*1000:.0f} -> {low_floor*1000:.0f}mm)")
        low_set = set(low_cands)

        # -- Tube validation -------------------------------------------------
        tubes = []
        lost_tips = []
        for v in cands + low_cands:
            t = _tube_from_tip(adj, co, v, href, median_edge, g)
            if t is not None:
                t["g"] = g[v]
                t["low_reach"] = v in low_set
                tubes.append(t)
            elif g[v] > 0.85 * g_max:
                # A near-global-max region IS a fingertip on any hand; its tube
                # failing means a finger is about to go missing (seen as a
                # straight middle finger replaced by a knuckle bump). Keep the
                # position: the rebuild rescue snaps a rebuilt tip onto it.
                lost_tips.append(co[v].copy())
                dbg(f"[geo {arp_side}] WARNING: top tip candidate "
                      f"(g={g[v]*1000:.0f}mm) has no valid tube -- "
                      f"a finger may be missing from this hand")
        _arcs = ", ".join(f"{t['arc']*1000:.0f}mm" for t in tubes)
        dbg(f"[geo {arp_side}] {len(cands)} tip candidates -> "
              f"{len(tubes)} validated tubes ({_arcs})")
        # Drop FOLDED tubes: a valid finger runs roughly straight from tip to
        # palm, so its arc ~ its tip->base span. A mis-traced tube can LOOP
        # BACK -- 85mm of arc folding into a 15mm tip->base span -- and then
        # evicts the real straight finger in _drop_overlapping (the "ring tip
        # went short into the finger" bug, where every OTHER finger was dead
        # straight). Drop such a tube only when it is a clear OUTLIER on a
        # mostly-straight hand, so a genuine FIST (all fingers fold -> low
        # median straightness -> no outlier) is left completely untouched.
        if len(tubes) >= 5:
            def _straightness(t):
                p = t["path"]
                s = float(np.linalg.norm(np.asarray(p[0]) - np.asarray(p[-1])))
                return s / t["arc"] if t["arc"] > 1e-6 else 1.0
            _sr  = {id(t): _straightness(t) for t in tubes}
            _med = float(np.median(list(_sr.values())))
            if _med > 0.70:                      # hand is mostly straight
                # Only EXTREME loops (arc > ~3.3x the tip->base span,
                # straightness < 0.30) are mis-traced artifacts. A real angled
                # THUMB (~0.6-0.8) or a curled finger (~0.4-0.5) must survive
                # -- dropping those shifts the selection and mislabels the
                # thumb. Tightened from 0.45 after that over-reach put the
                # thumb on the pinky.
                folded = [t for t in tubes
                          if _sr[id(t)] < 0.30 and _sr[id(t)] < 0.50 * _med]
                if folded and len(tubes) - len(folded) >= 5:
                    _fids = {id(t) for t in folded}
                    _fa = ", ".join(f"{t2['arc']*1000:.0f}mm(s={_sr[id(t2)]:.2f})"
                                    for t2 in folded)
                    tubes = [t for t in tubes if id(t) not in _fids]
                    dbg(f"[geo {arp_side}] dropped {len(folded)} FOLDED tube(s) "
                          f"({_fa}) -- arc >> tip->base span, mis-traced not a finger")
        # Drop imposter tubes far shorter than the hand's own fingers (knuckle
        # bumps / folds validate as tiny tubes with a HIGH tip g, and
        # top-5-by-g would then evict the real thumb — whose tip g is the
        # LOWEST). Floor is relative to the MEDIAN of the 5 longest tubes, not
        # the single longest: one junk mega-tube (a bridged prop) must not
        # raise the bar over a real pinky/thumb.
        if len(tubes) >= 5:
            ref = float(np.median(sorted((t["arc"] for t in tubes),
                                         reverse=True)[:5]))
            # low-reach tubes are exempt: a down-pointing thumb's tube can be
            # short AND low-g, and it only ever competes for the thumb slot
            # anyway (the nub gate in _pick_five still rejects genuine nubs
            # there).
            short = [t for t in tubes
                     if t["arc"] < min_finger * ref and not t.get("low_reach")]
            if short and len(tubes) - len(short) >= 5:
                _sids = {id(t) for t in short}
                _short_arcs = ", ".join(f"{t2['arc']*1000:.0f}mm" for t2 in short)
                tubes = [t for t in tubes if id(t) not in _sids]
                dbg(f"[geo {arp_side}] dropped {len(short)} imposter tube(s) "
                      f"({_short_arcs}) vs reference {ref*1000:.0f}mm")
        # Bump tubes riding a real finger's volume masquerade as fingers (and
        # as thumbs) -- drop them before choosing the five. If the dedup was
        # over-eager (pressed/interpenetrating fingers can coincide too),
        # retry the structure search on the FULL set: its min-gap constraint
        # already rejects any subset holding two tubes on one knuckle, so
        # duplicates can't be co-selected anyway.
        tubes_all = list(tubes)
        tubes = _drop_overlapping(tubes, arp_side)
        if len(tubes) < 5 <= len(tubes_all):
            dbg(f"[geo {arp_side}] dedup left {len(tubes)} -- retrying "
                  f"structure search on all {len(tubes_all)} tubes")
            tubes = tubes_all
        if len(tubes) < 5:
            dbg(f"[geo {arp_side}] fewer than 5 finger tubes -- giving up")
            return None

        # -- Select the 5 tubes + assign identity ---------------------------
        # _pick_five returns SEVERAL structurally-passing 5-sets; the
        # length-free finger_identity module LABELS each, and
        # _hand_plausibility picks the set whose labelled hand looks most
        # anatomical. This fixes selection grabbing a clean row of SHORT
        # BUMPS over the real (longer) fingers - structure alone scored the
        # bump row higher.
        fell_back = False
        candidates = _pick_five(tubes, g_max, hand_ref=href)
        if not candidates:
            # No set passes the structure gates: farthest-reaching FOUR
            # NORMAL tubes as fingers + EACH proximal leftover as a thumb
            # candidate (one set per leftover). A single min-g pick chose the
            # DEEPEST tube, and once low-reach candidates existed that was
            # always a thenar/palm fold, never the thumb (Adam: tip marker at
            # the thumb base, chain in the palm). Multiple sets let identity
            # + plausibility + the low-reach tier decide, same as the
            # structured path.
            dbg(f"[geo {arp_side}] no 5-tube combo passes hand-structure "
                  f"checks -- falling back to reach + proximal thumb")
            fell_back = True
            _byg = sorted(tubes, key=lambda t: t["g"], reverse=True)
            _fingers4 = [t for t in _byg if not t.get("low_reach")][:4]
            if len(_fingers4) == 4 and len(tubes) > 4:
                _left = sorted((t for t in tubes if t not in _fingers4),
                               key=lambda t: t["g"])
                candidates = [_fingers4 + [t] for t in _left[:6]]
            else:
                candidates = [_byg[:5]]

        best = None
        for five in candidates:
            ident = assign_finger_identity(
                _items_of(five), arp_side, np.asarray(hw_v, dtype=float),
                np.asarray(arm_dir, dtype=float))
            if ident is None:
                continue
            pl = _hand_plausibility(ident, g_max)
            # A LOW-REACH thumb is a RESCUE for hands whose real thumb never
            # reaches the normal candidate floor (down-pointing Disney
            # thumbs). On a hand whose thumb IS normally reachable, a
            # low-reach tube is a thenar/palm fold with an unbeatable
            # proximity vote in the identity scoring -- it hijacked Adam's
            # previously-correct thumb (tip marker at the thumb base, chain
            # folded into the palm). Tier the ranking: any set whose LABELLED
            # thumb is a normal candidate beats every set with a low-reach
            # thumb; low-reach only wins when nothing else does.
            tier = 0 if ident["thumb"]["ref"].get("low_reach") else 1
            if best is None or (tier, pl) > (best[0], best[1]):
                best = (tier, pl, ident, five)
        if best is None:
            dbg(f"[geo {arp_side}] identity assignment failed -- "
                  f"rejecting this pass")
            return None
        return best + (fell_back, lost_tips, tubes_all)

    stage = _tube_stage(hand_ref)
    # -- MULTI-SCALE RETRY: when the native scale produced no hand at all,
    # one whose thumb only came from the low-reach rescue (the double-shell
    # signature: shells trace duplicate tubes, structure search fails, a
    # thenar tube gets crowned), or a fallback whose identity is AMBIGUOUS
    # (weak thumb-vote margin -- the long-nail hand: nail creases truncate
    # three tubes at ~60mm, margins 0.19-0.32 vs >=2.7 on every approved
    # fallback hand), retry once at 1.75x. Self-gating: a hand that parsed
    # structurally, or fell back but with a CONFIDENT identity, never reaches
    # the retry -- Adam's approved fallback hand (margin 2.75) stays put. The
    # retry is only ADOPTED when it is strictly better -- a structured
    # (non-fallback) hand with a normally-reachable thumb -- or when the
    # native scale produced nothing.
    _weak = (stage is not None and stage[4]
             and stage[2].get("_margin", 1e9) < 0.8)
    if stage is None or stage[0] == 0 or _weak:
        dbg(f"[geo {arp_side}] "
              + ("no usable hand" if stage is None else
                 "thumb only via low-reach rescue" if stage[0] == 0 else
                 f"ambiguous fallback (identity margin "
                 f"{stage[2].get('_margin', 0.0):.2f})")
              + f" at native scale -- retrying tube stage at hand_ref x1.75 "
              f"({1.75 * hand_ref * 1000:.0f}mm)")
        retry = _tube_stage(1.75 * hand_ref)
        if retry is not None and not retry[4] and retry[0] == 1:
            dbg(f"[geo {arp_side}] x1.75 retry found a structured hand with "
                  f"a normally-reachable thumb -- using it")
            stage = retry
        elif stage is None and retry is not None:
            dbg(f"[geo {arp_side}] using x1.75 retry result "
                  f"(native scale had none)")
            stage = retry
        elif stage is not None:
            dbg(f"[geo {arp_side}] x1.75 retry not better -- "
                  f"keeping native-scale result")
    if stage is None:
        return None
    _tier, _plaus, _ident, _five, _fell_back, lost_tips, tubes_all = stage
    if _tier == 0:
        dbg(f"[geo {arp_side}] thumb from LOW-REACH rescue candidate "
              f"(no set with a normally-reachable thumb passed)")
    # Re-run identity on the WINNING set with a tag so the thumb score/margin
    # diagnostic prints (it was lost when selection went multi-set; pure
    # function, same result).
    assign_finger_identity(_items_of(_five), arp_side,
                           np.asarray(hw_v, dtype=float),
                           np.asarray(arm_dir, dtype=float),
                           tag=f"[geo {arp_side}]")
    _LAST_PLAUSIBILITY[arp_side] = _plaus     # backstop confidence signal for the operator
    named = {f: _ident[f]["ref"] for f in ("thumb", "index", "middle", "ring", "pinky")}
    _four = ("index", "middle", "ring", "pinky")
    if max(_four, key=lambda f: named[f]["arc"]) != "middle":
        dbg(f"[geo {arp_side}] note: middle is not the longest tube "
              f"(soft check - identity kept from structure, not length)")
    dbg(f"[geo {arp_side}] plausibility={_plaus:.2f}  "
          + "  ".join(f"{f}={named[f]['arc']*1000:.0f}mm" for f in
                      ("thumb", "index", "middle", "ring", "pinky")))

    # Structures the joint-placement code below still expects (rebuilt from the
    # shared identity result instead of the old thumb-pick/ordering).
    thumb = named["thumb"]
    rest  = [named["index"], named["middle"], named["ring"], named["pinky"]]
    _rb   = np.array([t["path"][-1] for t in rest])
    _, _, _rvt = np.linalg.svd(_rb - _rb.mean(axis=0))
    row_dir = _rvt[0]

    # -- Plausibility gates ---------------------------------------------------
    # ONE short finger is rescuable (garment rim / broken tube). TWO OR MORE
    # wildly-off finger lengths mean the tubes aren't fingers at all (bumps on
    # a fingerless palm island) -- reject the pass so the caller can try the
    # bridge pass / bail to the neural engine instead of placing garbage.
    four = ("index", "middle", "ring", "pinky")
    arcs4 = [named[f]["arc"] for f in four]
    med4 = float(np.median(arcs4))
    n_off = sum(1 for a in arcs4 if a < 0.42 * med4 or a > 1.9 * med4)
    if n_off >= 2:
        dbg(f"[geo {arp_side}] implausible finger pattern "
              f"({n_off}/4 lengths far off the median) -- rejecting this pass")
        return None
    # The bridge pass runs on exotic layered/separate geometry where a
    # structurally-plausible but WRONG pick is common (index at a tip shell,
    # pinky on the thumb). Only accept a canonical-looking hand from it;
    # anything else is better handled by the neural engine.
    if do_bridge and named["middle"]["arc"] < 0.95 * max(arcs4):
        dbg(f"[geo {arp_side}] bridge-pass result not canonical "
              f"(middle is not the longest finger) -- rejecting")
        return None

    # -- Joints along the medial arc ----------------------------------------
    # The band-radius jump fires at the WEB between fingers, which sits
    # distal of the true MCP knuckle: extend the medial path proximally for
    # non-thumb MCPs. Fractions are anatomical phalange proportions of the
    # FULL (extended) finger arc.
    result = {}
    for fname, t in named.items():
        path, cum, arc = t["path"], t["cum"], t["arc"]
        # Proximal direction from the MID-TUBE window (0.55-0.85 of the
        # arc): per-finger and straight. The old path-end direction was
        # already bent toward the palm centre by the near-web bands, which
        # drove all four MCPs to pinch together.
        a = _sample_path(path, cum, arc * 0.55)
        b = _sample_path(path, cum, arc * 0.85)
        dir_prox = b - a
        dl = float(np.linalg.norm(dir_prox))
        dir_prox = dir_prox / dl if dl > 1e-9 else np.zeros(3)
        arp = _FINGER_TO_ARP[fname]
        if fname == "thumb":
            # Tube covers distal+proximal phalanx; entry ~ THUMB_2 (MCP at
            # web level). THUMB_1 (CMC) continues into the thenar mound --
            # shallow by default so it stays in the mound, not deep in the
            # palm. thumb_depth = user slider.
            j = {"tip":   path[0],
                 "phal3": _sample_path(path, cum, arc * 0.45),
                 "phal2": path[-1],
                 "phal1": path[-1] + dir_prox * (arc * thumb_depth)}
        else:
            # Knuckle sits proximal of the web where the tube ends: extend
            # each finger along its own mid-tube direction, ACROSS-STRIPPED
            # (row component removed). Finger axes anatomically converge
            # toward the palm centre, so a raw per-finger extension drags
            # every MCP toward the middle; the web entry is already
            # laterally correct, so the extension must not move sideways.
            # knuckle_depth = user slider.
            dir_ext = dir_prox - float(dir_prox @ row_dir) * row_dir
            dln = float(np.linalg.norm(dir_ext))
            dir_ext = dir_ext / dln if dln > 1e-9 else dir_prox
            mcp = path[-1] + dir_ext * (arc * knuckle_depth)
            S = arc * (1.0 + knuckle_depth)
            j = {"tip":   path[0],
                 "phal3": _sample_path(path, cum, S * 0.23),
                 "phal2": _sample_path(path, cum, S * 0.52),
                 "phal1": mcp}
        for key, p in j.items():
            result[f"{arp[key]}_{arp_side}"] = Vector((float(p[0]),
                                                       float(p[1]),
                                                       float(p[2])))

    # -- DIAGNOSTIC: tube arc vs straight tip->base vs placed MCP->TIP --------
    # A finger whose PLACED chain (MCP->TIP euclidean) is far shorter than its own
    # tube ARC has a tip that folded back short of the fingertip (the curled-ring
    # "tip went short into the finger" case). straight = tube path[0]->path[-1]
    # euclidean; a curled finger has straight << arc even when correct, so the
    # tell is chain << straight (the placed tip didn't reach the tube's own end).
    for _f in ("thumb", "index", "middle", "ring", "pinky"):
        _t = named[_f]
        _p = _t["path"]
        _arc = _t["arc"]
        _straight = float(np.linalg.norm(np.asarray(_p[0]) - np.asarray(_p[-1])))
        _arp = _FINGER_TO_ARP[_f]
        _tp = result.get(f"{_arp['tip']}_{arp_side}")
        _mc = result.get(f"{_arp['phal1']}_{arp_side}")
        _chain = (_tp - _mc).length if (_tp and _mc) else 0.0
        # Healthy finger: chain >= arc (MCP is extended proximally past the tube).
        # chain << arc => the placed tip folded back short of the tube's end.
        # Also show whether the TUBE ITSELF folds (straight << arc = path[0] near
        # path[-1], a malformed/over-curled tube) vs the PLACEMENT losing the tip.
        _flag = ""
        if _arc > 1e-6 and _chain < 0.60 * _arc:
            _cause = "tube folds back" if _straight < 0.55 * _arc else "placement lost tip"
            _flag = f"  <-- TIP SHORT (chain {_chain*1000:.0f} << arc {_arc*1000:.0f}; {_cause})"
        dbg(f"  [geo-diag {arp_side}] {_f}: arc={_arc*1000:.0f}mm  "
              f"straight={_straight*1000:.0f}mm  chain={_chain*1000:.0f}mm{_flag}")

    # Knuckle-row anchor: the four finger MCPs sit on the metacarpal-head
    # row (the same row the identity step relies on).
    # A finger whose tube is FAR shorter than its siblings hit a garment
    # rim (half glove / sleeve edge), not the palm -- its MCP stopped at
    # the rim. Rescue: fit the row on the trustworthy fingers only, snap
    # the suspect's MCP fully onto that row, and respace its PIP/DIP
    # between tip and the rescued MCP. Trustworthy MCPs get the usual 50%
    # blend (keep along-row spread, remove per-finger wobble).
    mcp_key = {f: f"{_FINGER_TO_ARP[f]['phal1']}_{arp_side}" for f in four}
    suspects = [f for f in four if named[f]["arc"] < 0.60 * med4]
    good = [f for f in four if f not in suspects]
    if len(good) < 2:
        suspects, good = [], list(four)   # too many shorties: no safe row
    P = np.array([[result[mcp_key[f]].x, result[mcp_key[f]].y,
                   result[mcp_key[f]].z] for f in good])
    rc = P.mean(axis=0)
    _, _, rvt = np.linalg.svd(P - rc)
    rd = rvt[0]
    for f, p in zip(good, P):
        proj = rc + float((p - rc) @ rd) * rd
        bl = 0.5 * p + 0.5 * proj
        result[mcp_key[f]] = Vector((float(bl[0]), float(bl[1]), float(bl[2])))
    # Pool of orphan tips a rebuilt finger can claim: tips of validated tubes
    # the selection did NOT use, plus near-max candidates whose tube failed.
    # One of these is usually the rebuilt finger's REAL fingertip.
    _sel = [thumb] + rest
    orphan_tips = [t["path"][0] for t in tubes_all
                   if all(t is not s_ for s_ in _sel)] + lost_tips
    _nbr = {"middle": ("index", "ring"), "ring": ("middle", "pinky")}
    for f in suspects:
        arp = _FINGER_TO_ARP[f]
        nb = _nbr.get(f)
        if (nb and named[f]["arc"] < 0.50 * med4
                and all(x in good for x in nb)):
            # SEVERELY short interior finger: its tube (often a knuckle bump
            # that stole the slot from a failed real tube) can't be trusted --
            # not even its tip. Rebuild the chain by interpolating the two
            # good neighbours, then claim the nearest orphan tip (the real
            # fingertip, when detection saw it) or push slightly distal.
            a1, a2 = _FINGER_TO_ARP[nb[0]], _FINGER_TO_ARP[nb[1]]
            for part in ("phal1", "phal2", "phal3", "tip"):
                ka = f"{a1[part]}_{arp_side}"
                kb = f"{a2[part]}_{arp_side}"
                result[f"{arp[part]}_{arp_side}"] = (result[ka] + result[kb]) * 0.5
            tip_k = f"{arp['tip']}_{arp_side}"
            mcp_v = result[f"{arp['phal1']}_{arp_side}"]
            tip_v = result[tip_k]
            t_np = np.array([tip_v.x, tip_v.y, tip_v.z])
            snap, snap_d = None, 0.6 * med4
            for p_ in orphan_tips:
                dd = float(np.linalg.norm(np.asarray(p_) - t_np))
                if dd < snap_d:
                    snap_d, snap = dd, p_
            if snap is not None:
                result[tip_k] = Vector((float(snap[0]), float(snap[1]),
                                        float(snap[2])))
                src = "orphan tip"
            else:
                result[tip_k] = mcp_v + (tip_v - mcp_v) * 1.08
                src = "neighbour extrapolation"
            tip_v = result[tip_k]
            result[f"{arp['phal3']}_{arp_side}"] = tip_v.lerp(mcp_v, 0.23)
            result[f"{arp['phal2']}_{arp_side}"] = tip_v.lerp(mcp_v, 0.52)
            dbg(f"[geo {arp_side}] {f} tube bogus "
                  f"({named[f]['arc']*1000:.0f}mm vs median {med4*1000:.0f}mm) "
                  f"-- chain rebuilt from neighbours, tip from {src}")
            continue
        # Mildly short (garment rim): tip is real, only the MCP stopped at the
        # rim -- snap it onto the knuckle row and respace PIP/DIP.
        k = mcp_key[f]
        p = np.array([result[k].x, result[k].y, result[k].z])
        proj = rc + float((p - rc) @ rd) * rd
        result[k] = Vector((float(proj[0]), float(proj[1]), float(proj[2])))
        tip_v = result[f"{arp['tip']}_{arp_side}"]
        mcp_v = result[k]
        result[f"{arp['phal3']}_{arp_side}"] = tip_v.lerp(mcp_v, 0.23)
        result[f"{arp['phal2']}_{arp_side}"] = tip_v.lerp(mcp_v, 0.52)
        dbg(f"[geo {arp_side}] {f} tube short "
              f"({named[f]['arc']*1000:.0f}mm vs median {med4*1000:.0f}mm) "
              f"-- MCP rescued onto the knuckle row")
    return result
