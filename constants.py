# constants.py — Marker definitions, bone maps, and size configurations for Easy Rigify.
from mathutils import Vector, Matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BODY_SIZE    = 0.026
ARM_SIZE     = 0.026
FINGER_SIZE  = 0.008

FACE_SIZE    = 0.0085
TIP_EXTEND   = 0.011

ARM_MARKER_BASES = {"SHOULDER", "ARM", "ELBOW", "HAND", "HAND_TIP"}

FOOT_Y = 0.05
HEEL_X = 0.07
HEEL_Y = 0.14

FINGER_PREFIXES = ("f_index.", "f_middle.", "f_ring.", "f_pinky.", "thumb.", "palm.")

# Finger .01 bones that need primary_rotation_axis = 'X' before generate.
# Defined once here — used in AlignRig, FixBoneRoll, and GenerateRig.
FINGER_01_BONES = [
    "thumb.01.L",    "thumb.01.R",
    "f_index.01.L",  "f_index.01.R",
    "f_middle.01.L", "f_middle.01.R",
    "f_ring.01.L",   "f_ring.01.R",
    "f_pinky.01.L",  "f_pinky.01.R",
]

# ─────────────────────────────────────────────────────────────────────────────
# BODY MARKER DEFINITIONS
# Positions match Rigify Human Metarig default A-pose bone head positions.
# A-pose: arms at ~45° down from horizontal, character facing -Y.
# ─────────────────────────────────────────────────────────────────────────────
SINGLE_MARKERS = [
    # Centre spine — match Rigify metarig bone head positions
    ("PELVIS",    "CUBE",   ( 0.000,  0.000,  1.000)),  # pelvis head
    ("SPINE_001", "CUBE",   ( 0.000,  0.000,  1.160)),  # spine.001 head
    ("SPINE_002", "CUBE",   ( 0.000,  0.000,  1.298)),  # spine.002 head
    ("CHEST",     "CUBE",   ( 0.000,  0.000,  1.434)),  # spine.003 head
    ("NECK",      "SPHERE", ( 0.000,  0.000,  1.582)),  # spine.004 head
    ("HEAD",      "SPHERE", ( 0.000,  0.000,  1.700)),  # spine.006 head
]

# Rigify metarig A-pose: arms angled ~45° down from shoulder.
# X = lateral distance, Y = slight back offset for arm droop, Z = height
BILATERAL_MARKERS = [
    # Arm chain
    ("SHOULDER",         "SPHERE", ( 0.156,  0.000,  1.434)),
    ("ARM",              "SPHERE", ( 0.284,  0.000,  1.327)),
    ("ELBOW",            "SPHERE", ( 0.544,  0.000,  1.090)),
    ("HAND",             "SPHERE", ( 0.764,  0.000,  0.900)),
    ("HAND_TIP",         "SPHERE", ( 0.900,  0.000,  0.800)),  # middle fingertip — camera framing ref

    # Leg chain
    ("THIGH",            "SPHERE", ( 0.098,  0.000,  1.000)),
    ("SHIN",             "SPHERE", ( 0.098,  0.000,  0.530)),
    ("FOOT",             "SPHERE", ( 0.098,  FOOT_Y, 0.085)),
    ("TOES",             "SPHERE", ( 0.098, -0.105,  0.031)),
    ("HEEL",             "SPHERE", ( HEEL_X, HEEL_Y, 0.040)),

    # Thumb
    ("THUMB_1",          "SPHERE", ( 0.7720, -0.0295,  0.8972)),
    ("THUMB_2",          "SPHERE", ( 0.8069, -0.0506,  0.8705)),
    ("THUMB_3",          "SPHERE", ( 0.8308, -0.0661,  0.8534)),
    ("THUMB_TIP",        "SPHERE", ( 0.8510, -0.0793,  0.8395)),

    # Index
    ("FINGER_INDEX_1",   "SPHERE", ( 0.7906, -0.0236,  0.8718)),
    ("FINGER_INDEX_2",   "SPHERE", ( 0.8297, -0.0256,  0.8448)),
    ("FINGER_INDEX_3",   "SPHERE", ( 0.8573, -0.0271,  0.8214)),
    ("FINGER_INDEX_TIP", "SPHERE", ( 0.8761, -0.0279,  0.8065)),

    # Middle
    ("FINGER_MIDDLE_1",  "SPHERE", ( 0.7975,  0.0000,  0.8693)),
    ("FINGER_MIDDLE_2",  "SPHERE", ( 0.8421,  0.0000,  0.8404)),
    ("FINGER_MIDDLE_3",  "SPHERE", ( 0.8771,  0.0000,  0.8165)),
    ("FINGER_MIDDLE_TIP","SPHERE", ( 0.9002,  0.0000,  0.8000)),

    # Ring
    ("FINGER_RING_1",    "SPHERE", ( 0.7906,  0.0236,  0.8718)),
    ("FINGER_RING_2",    "SPHERE", ( 0.8297,  0.0256,  0.8448)),
    ("FINGER_RING_3",    "SPHERE", ( 0.8573,  0.0271,  0.8214)),
    ("FINGER_RING_TIP",  "SPHERE", ( 0.8761,  0.0279,  0.8065)),

    # Pinky
    ("FINGER_PINKY_1",   "SPHERE", ( 0.7810,  0.0459,  0.8735)),
    ("FINGER_PINKY_2",   "SPHERE", ( 0.8131,  0.0498,  0.8495)),
    ("FINGER_PINKY_3",   "SPHERE", ( 0.8365,  0.0529,  0.8312)),
    ("FINGER_PINKY_TIP", "SPHERE", ( 0.8537,  0.0552,  0.8180)),
]

FINGER_BASE_NAMES = {
    "THUMB_1","THUMB_2","THUMB_3","THUMB_TIP",
    "FINGER_INDEX_1","FINGER_INDEX_2","FINGER_INDEX_3","FINGER_INDEX_TIP",
    "FINGER_MIDDLE_1","FINGER_MIDDLE_2","FINGER_MIDDLE_3","FINGER_MIDDLE_TIP",
    "FINGER_RING_1","FINGER_RING_2","FINGER_RING_3","FINGER_RING_TIP",
    "FINGER_PINKY_1","FINGER_PINKY_2","FINGER_PINKY_3","FINGER_PINKY_TIP",
}

ALL_MARKERS = list(SINGLE_MARKERS)
for _base, _dtype, (_lx, _ly, _lz) in BILATERAL_MARKERS:
    ALL_MARKERS.append((f"{_base}_L", _dtype, ( _lx, _ly, _lz)))
    ALL_MARKERS.append((f"{_base}_R", _dtype, (-_lx, _ly, _lz)))

# ─────────────────────────────────────────────────────────────────────────────
# FACE MARKER DEFINITIONS
# Covers every bone chain in Rigify's face rig.
# Single face markers (centre line) + bilateral (L/R)
# Each bilateral entry: (base_name, display_type, left_xyz)
# ─────────────────────────────────────────────────────────────────────────────
# Z reference: HEAD marker at 1.78, face spans ~1.65–1.92

FACE_SINGLE = [
    # Jaw / chin / mouth bottom
    ("FACE_JAW",          "SPHERE", ( 0.000, -0.070,  1.665)),  # jaw — under chin (menton)
    ("FACE_CHIN",         "SPHERE", ( 0.000, -0.095,  1.675)),  # chin — chin tip (pogonion)
    ("FACE_LIP_T",        "SPHERE", ( 0.000, -0.115,  1.710)),  # lips.T
    ("FACE_LIP_B",        "SPHERE", ( 0.000, -0.108,  1.700)),  # lips.B
    # Nose centre
    ("FACE_NOSE_BRIDGE",         "SPHERE", ( 0.000, -0.138,  1.758)),  # nose
    ("FACE_NOSE_TIP",     "SPHERE", ( 0.000, -0.143,  1.746)),  # nose.001 (tip detail)
    ("FACE_NOSE_BOT",     "SPHERE", ( 0.000, -0.127,  1.744)),  # nose_glue.004
    # Teeth and tongue
    ("FACE_TEETH_T",      "CUBE",   ( 0.000, -0.100,  1.705)),  # teeth.T
    ("FACE_TEETH_B",      "CUBE",   ( 0.000, -0.098,  1.697)),  # teeth.B
    ("FACE_TONGUE_1",     "PLAIN_AXES", ( 0.000, -0.075,  1.692)),  # tongue
    ("FACE_TONGUE_2",     "PLAIN_AXES", ( 0.000, -0.058,  1.692)),  # tongue.001
    ("FACE_TONGUE_3",     "PLAIN_AXES", ( 0.000, -0.042,  1.692)),  # tongue.002
    # Chin glue
    ("FACE_LIP_BOT",    "SPHERE", ( 0.000, -0.110,  1.702)),  # chin_end_glue.001 tail
    # Centre forehead / brow reference
    ("FACE_FOREHEAD",     "SPHERE", ( 0.000, -0.090,  1.848)),  # between forehead_side L/R
    ("FACE_BROW",         "SPHERE", ( 0.000, -0.112,  1.815)),  # between brow_inner L/R
]

FACE_BILATERAL = [
    # ── MOUTH ANCHORS (4 anchors drive entire lip loop) ────────────────────
    # corner, top-mid, bottom-mid drive the rest via interpolation
    ("FACE_MOUTH_CORNER",  "SPHERE", ( 0.030, -0.108,  1.707)),  # lips.L/R corner
    ("FACE_MOUTH_TOP",     "SPHERE", ( 0.013, -0.116,  1.713)),  # lips.T.L.001
    ("FACE_MOUTH_BOT",     "SPHERE", ( 0.013, -0.109,  1.703)),  # lips.B.L.001

    # ── BROW ANCHORS (4 markers drive the 3-bone brow arc) ─────────────────
    ("FACE_BROW_1",    "SPHERE", ( 0.015, -0.112,  1.815)),  # brow.T.L.004 tail
    ("FACE_BROW_2",    "SPHERE", ( 0.023, -0.115,  1.817)),  # brow.T.L.002/004 joint
    ("FACE_BROW_3",      "SPHERE", ( 0.032, -0.119,  1.819)),  # brow.T.L.001/002 joint
    ("FACE_BROW_OUTER",    "SPHERE", ( 0.050, -0.115,  1.812)),  # brow.T.L.001 head

    # ── BROW BOTTOM (under-brow bones) ─────────────────────────────────────
    ("FACE_CREASE_INNER","SPHERE", ( 0.046, -0.109,  1.800)),  # reference marker (near eye outer)
    ("FACE_CREASE_OUTER",  "SPHERE", ( 0.018, -0.106,  1.796)),  # brow.B.L chain start
    ("FACE_BROW_BOT_OUTER","SPHERE", ( 0.042, -0.110,  1.793)),  # brow.B.L.003 tail (outer reference)
    ("FACE_LID_CREASE_T",   "SPHERE", ( 0.028, -0.114,  1.820)),  # brow.B.L.002 tail / .003 head junction — sits above EYE_TOP (1.811)

    # ── EYE ANCHOR (single sphere = eye pivot + lid loop centre) ───────────
    ("FACE_EYE_CENTER",    "SPHERE", ( 0.032, -0.105,  1.800)),  # eye.L pivot
    # Top/bottom of eyelid opening (2 anchors drive 4+4 lid bones each)
    ("FACE_EYE_TOP",       "SPHERE", ( 0.032, -0.112,  1.811)),  # lid.T.L apex
    ("FACE_EYE_BOT",       "SPHERE", ( 0.032, -0.108,  1.791)),  # lid.B.L apex
    # Inner/outer eye corners
    ("FACE_EYE_INNER",     "SPHERE", ( 0.018, -0.110,  1.800)),  # lid inner corner
    ("FACE_EYE_OUTER",     "SPHERE", ( 0.048, -0.109,  1.800)),  # lid outer corner

    # ── CHEEK ─────────────────────────────────────────────────────────────
    ("FACE_CHEEK",         "SPHERE", ( 0.050, -0.107,  1.757)),  # cheek.B.L.001
    ("FACE_CHEEK_TOP",     "SPHERE", ( 0.042, -0.108,  1.772)),  # cheek_glue.T.L.001 / cheek.T.L.001 head

    # ── NOSE ──────────────────────────────────────────────────────────────
    ("FACE_NOSE_WING",     "SPHERE", ( 0.016, -0.132,  1.750)),  # nose.L / nose.L.001
    ("FACE_NOSE_BRIDGE",   "SPHERE", ( 0.012, -0.120,  1.768)),  # nose.L chain start

    # ── EAR ───────────────────────────────────────────────────────────────
    ("FACE_EAR",           "SPHERE", ( 0.080,  0.010,  1.775)),  # ear chain root

    # ── JAW SIDE ──────────────────────────────────────────────────────────
    ("FACE_JAW_SIDE",      "SPHERE", ( 0.060, -0.030,  1.735)),  # jaw.L — jaw angle, between chin and ear
    ("FACE_CHIN_SIDE",     "SPHERE", ( 0.018, -0.095,  1.660)),  # chin.L / chin.R

    # ── TEMPLE ────────────────────────────────────────────────────────────
    ("FACE_TEMPLE",        "SPHERE", ( 0.062, -0.085,  1.842)),  # temple.L

    # ── FOREHEAD SIDE (4 markers, one per forehead bone head) ─────────────
    ("FACE_FOREHEAD_SIDE",   "SPHERE", ( 0.030, -0.090,  1.848)),  # forehead.L head
    ("FACE_FOREHEAD_SIDE_1", "SPHERE", ( 0.031, -0.095,  1.840)),  # forehead.L.001 head
    ("FACE_FOREHEAD_SIDE_2", "SPHERE", ( 0.030, -0.100,  1.831)),  # forehead.L.002 head
    ("FACE_FOREHEAD_SIDE_3",    "SPHERE", ( 0.028, -0.103,  1.820)),  # forehead.L.003 head

]

ALL_FACE_MARKERS = list(FACE_SINGLE)
for _base, _dtype, (_lx, _ly, _lz) in FACE_BILATERAL:
    ALL_FACE_MARKERS.append((f"{_base}_L", _dtype, ( _lx, _ly, _lz)))
    ALL_FACE_MARKERS.append((f"{_base}_R", _dtype, (-_lx, _ly, _lz)))

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL FACE DETECTOR — core landmark schema
#
# The subset of face markers the neural model predicts. Teeth and tongue are
# EXCLUDED (placed geometrically by the user). Midline markers stay as-is;
# bilateral bases expand to _L / _R. This ordered list is the contract shared by
# the data generator (gen_face_training_data.py), the trainer (train_face_detector.py)
# and inference (_detect_face_onnx) — its order defines the heatmap channel order,
# so DO NOT reorder without regenerating the dataset and retraining.
# ─────────────────────────────────────────────────────────────────────────────
CORE_FACE_MIDLINE = [
    "FACE_LIP_T", "FACE_LIP_B", "FACE_BROW",
    "FACE_NOSE_TIP", "FACE_NOSE_BRIDGE",
    "FACE_CHIN", "FACE_JAW", "FACE_FOREHEAD",
]
CORE_FACE_BILATERAL = [
    "FACE_MOUTH_CORNER", "FACE_BROW_OUTER",
    "FACE_EYE_INNER", "FACE_EYE_OUTER", "FACE_EYE_TOP", "FACE_EYE_BOT",
    "FACE_NOSE_WING", "FACE_JAW_SIDE", "FACE_EAR",
    "FACE_CHEEK", "FACE_LID_CREASE_T",
]

# Fully-resolved marker names (30): 8 midline + 11 bilateral × 2.
CORE_FACE_LANDMARKS = list(CORE_FACE_MIDLINE)
for _cb in CORE_FACE_BILATERAL:
    CORE_FACE_LANDMARKS.append(f"{_cb}_L")
    CORE_FACE_LANDMARKS.append(f"{_cb}_R")

# L<->R swap map for horizontal-flip augmentation (mirror a sample to double data).
CORE_FACE_FLIP_SWAP = {f"{_cb}_L": f"{_cb}_R" for _cb in CORE_FACE_BILATERAL}
CORE_FACE_FLIP_SWAP.update({f"{_cb}_R": f"{_cb}_L" for _cb in CORE_FACE_BILATERAL})

# FULL neural marker set for the mesh-native (point-cloud) face model: EVERY face
# skin marker EXCEPT the ones placed geometrically — teeth, tongue, and the eyeball
# centres (EYE_CENTER, which come from the eye mesh). Order follows ALL_FACE_MARKERS.
_FACE_NEURAL_EXCLUDE = ("TEETH", "TONGUE", "EYE_CENTER")
FULL_FACE_LANDMARKS = [_nm for (_nm, _dt, _p) in ALL_FACE_MARKERS
                       if not any(_x in _nm for _x in _FACE_NEURAL_EXCLUDE)]

# ─────────────────────────────────────────────────────────────────────────────
# FACE BONE MAP — anchor → bone(s)
#
# Strategy:
#  • ANCHOR markers snap the PRIMARY bone (head) directly.
#  • CHAIN bones between anchors are placed by linear interpolation between
#    the two surrounding anchor positions (handled in the align function).
#  • Eye lid chains: 4 bones each, interpolated around a half-ellipse from
#    inner-corner → apex → outer-corner anchors.
# ─────────────────────────────────────────────────────────────────────────────

# Direct snap: marker → bone head (tail keeps original length/direction)
FACE_DIRECT_MAP = {
    # Centre single
    "FACE_LIP_T":         ["lip.T"],
    "FACE_LIP_B":         ["lip.B"],
    "FACE_TONGUE_1":      ["tongue"],
    "FACE_TONGUE_2":      ["tongue.001"],
    "FACE_TONGUE_3":      ["tongue.002"],
    "FACE_NOSE_BOT":      ["nose_glue.004"],
    "FACE_LIP_BOT":     ["chin_end_glue.001"],
    # Bilateral direct snaps
    "FACE_MOUTH_CORNER_L":["lip.L", "cheek.B.L"],
    "FACE_MOUTH_CORNER_R":["lip.R", "cheek.B.R"],
    "FACE_BROW_BOT_OUTER_L":  ["cheek.T.L"],
    "FACE_BROW_BOT_OUTER_R":  ["cheek.T.R"],
    "FACE_EYE_CENTER_L":  ["eye.L"],
    "FACE_EYE_CENTER_R":  ["eye.R"],
    # Lid bones are now positioned entirely by FACE_CHAIN_MAP (see below)
    "FACE_CHEEK_L":          ["cheek.B.L.001"],
    "FACE_CHEEK_R":          ["cheek.B.R.001"],
    "FACE_NOSE_WING_L":   ["nose_glue.L.001"],
    "FACE_NOSE_WING_R":   ["nose_glue.R.001"],
    # brow_glue.B.L.002 head (tail is still snapped to BROW_MID_1 via FACE_TAIL_MAP)
    "FACE_LID_CREASE_T_L":     ["brow_glue.B.L.002"],
    "FACE_LID_CREASE_T_R":     ["brow_glue.B.R.002"],
    # Forehead — base bone head snap only; .001/.002/.003 heads+tails handled in _align_face
    "FACE_FOREHEAD_SIDE_L":   ["forehead.L"],
    "FACE_FOREHEAD_SIDE_R":   ["forehead.R"],
    # face bone position handled separately in _align_face (matches spine.006)
}

# Direct snaps where the tail should face +Y (into mouth) rather than +Z.
# Head is placed at the marker; tail extends in local +Y direction.
FACE_DIRECT_Y_MAP = {
    "FACE_TEETH_T": ["teeth.T"],
    "FACE_TEETH_B": ["teeth.B"],
}

# Tail-only snaps: marker → bone tail (head/length unchanged)
FACE_TAIL_MAP = {
    "FACE_LIP_B":             ["chin_end_glue.001"],
    "FACE_LIP_T":             ["nose_glue.004"],
    "FACE_NOSE_WING_L":       ["lip.T.L.003", "nose.L.002"],
    "FACE_NOSE_WING_R":       ["lip.T.R.003", "nose.R.002"],
    "FACE_BROW_1_L":      ["forehead.L"],
    "FACE_BROW_1_R":      ["forehead.R"],
    "FACE_MOUTH_CORNER_L":    ["chin.L.003"],
    "FACE_MOUTH_CORNER_R":    ["chin.R.003"],
    "FACE_MOUTH_TOP_L":       ["nose_glue.L.001"],
    "FACE_MOUTH_TOP_R":       ["nose_glue.R.001"],
    # brow_glue.B.L.002 tail is conditional (BROW_2 custom / BROW_3 default) — handled in _align_face Step 9
    "FACE_BROW_BOT_OUTER_L":  ["cheek.B.L.003"],
    "FACE_BROW_BOT_OUTER_R":  ["cheek.B.R.003"],
    # cheek.T.L tail to CHEEK_TOP
    "FACE_CHEEK_TOP_L":       ["cheek.T.L"],
    "FACE_CHEEK_TOP_R":       ["cheek.T.R"],
}

# Aimed-tail snaps: tail is DIRECTED toward a marker, but bone length is preserved.
# Use case: cheek.B.L head is at MOUTH_CORNER, tail should aim toward CHEEK_L.
FACE_AIM_TAIL_MAP = {
    "FACE_CHEEK_L": ["cheek.B.L"],
    "FACE_CHEEK_R": ["cheek.B.R"],
}

# Equal-length pair chains — intentionally empty.
# nose.L / nose.L.002 / brow.B.L.004 are a connected chain; positioning is
# handled by FACE_TAIL_MAP (brow.B.L.004 tail → NOSE_BRIDGE_L) so that
# nose.L head follows via use_connect without any disconnection.
FACE_EQUAL_PAIR_CHAINS = []

# Bone-to-bone tail snaps: snap target bone's tail to source bone's head.
# Runs AFTER FACE_CHAIN_MAP so brow bones are already in position.
FACE_BONE_TAIL_FROM_BONE_HEAD = [
    # brow.T and forehead .001/.002/.003 are handled by conditional code in _align_face
]

# Rigid translate: entire bone chain shifts so first bone's head hits the marker.
# Bone shapes (direction, length) are preserved exactly.
FACE_TRANSLATE_CHAINS = [
    (["ear.L", "ear.L.001", "ear.L.002", "ear.L.003", "ear.L.004"], "FACE_EAR_L"),
    (["ear.R", "ear.R.001", "ear.R.002", "ear.R.003", "ear.R.004"], "FACE_EAR_R"),
]

# Interpolated chains: (anchor_A, anchor_B, [(bone_name, t), ...])
# t=0.0 → fully at A position, t=1.0 → fully at B position
# These bones sit BETWEEN two anchor markers
FACE_INTERP_CHAINS = []  # Lid bones are now handled entirely by FACE_CHAIN_MAP

# ─────────────────────────────────────────────────────────────────────────────
# FACE CHAIN MAP — bone chains stretched between marker pairs
#
# Each entry: ([bone_names], [marker_names])
# • len(markers) == 2            → start/end only; N bones distributed by
#                                  proportional original lengths
# • len(markers) == len(bones)+1 → every joint position given explicitly
# ─────────────────────────────────────────────────────────────────────────────
FACE_CHAIN_MAP = [
    # ── JAW MASTER (head computed in execute as centre of JAW_SIDE_L/R) ──
    (["jaw"], ["FACE_JAW", "FACE_CHIN"]),
    # jaw_master is handled by special code in AlignRig.execute

    # ── CHIN CENTER CHAIN — chin head=CHIN, chin.001 tail=CHIN_GLUE ──────
    (["chin", "chin.001"], ["FACE_CHIN", "FACE_LIP_BOT"], "even"),

    # ── JAW SIDE CHAIN — jaw.L head=JAW_SIDE, jaw.L.001 tail=CHIN_SIDE ───
    (["jaw.L", "jaw.L.001"], ["FACE_JAW_SIDE_L", "FACE_CHIN_SIDE_L"], "even"),
    (["jaw.R", "jaw.R.001"], ["FACE_JAW_SIDE_R", "FACE_CHIN_SIDE_R"], "even"),

    # ── CHIN CHAIN — chin.L head=CHIN_SIDE, chin.L.003 tail=MOUTH_CORNER ─
    (["chin.L", "chin.L.003"], ["FACE_CHIN_SIDE_L", "FACE_MOUTH_CORNER_L"], "even"),
    (["chin.R", "chin.R.003"], ["FACE_CHIN_SIDE_R", "FACE_MOUTH_CORNER_R"], "even"),

    # ── TEMPLE ────────────────────────────────────────────────────────────
    (["temple.L"], ["FACE_TEMPLE_L", "FACE_JAW_SIDE_L"]),
    (["temple.R"], ["FACE_TEMPLE_R", "FACE_JAW_SIDE_R"]),

    # ── BROW UPPER ARC — handled by conditional code in _align_face (Step 9) ────
    # Custom rig (.004 exists): base=BOT_OUTER→OUTER, .001=OUTER→BROW_3,
    #   .002=BROW_3→BROW_2, .004=BROW_2→BROW_1
    # Default rig (no .004): base=BOT_OUTER→OUTER, .001=OUTER→BROW_3, .002=BROW_3→BROW_1

    # ── LOWER BROW (.000–.003) — split arc; .002.tail / .003.head pin to LID_CREASE_T
    (["brow.B.L", "brow.B.L.001", "brow.B.L.002"], ["FACE_CREASE_OUTER_L", "FACE_LID_CREASE_T_L"]),
    (["brow.B.L.003"],                              ["FACE_LID_CREASE_T_L", "FACE_CREASE_INNER_L"]),
    (["brow.B.R", "brow.B.R.001", "brow.B.R.002"], ["FACE_CREASE_OUTER_R", "FACE_LID_CREASE_T_R"]),
    (["brow.B.R.003"],                              ["FACE_LID_CREASE_T_R", "FACE_CREASE_INNER_R"]),
    # brow.B.L.004: exact single bone head=CREASE_INNER, tail=NOSE_BRIDGE
    (["brow.B.L.004"], ["FACE_CREASE_INNER_L", "FACE_NOSE_BRIDGE_L"]),
    (["brow.B.R.004"], ["FACE_CREASE_INNER_R", "FACE_NOSE_BRIDGE_R"]),

    # ── LOWER BROW EXTENDED (.005–.008) — arc-preserved, 4 bones ────────────
    (["brow.B.L.005", "brow.B.L.006", "brow.B.L.007", "brow.B.L.008"],
     ["FACE_CREASE_OUTER_L", "FACE_NOSE_BRIDGE_L"]),
    (["brow.B.R.005", "brow.B.R.006", "brow.B.R.007", "brow.B.R.008"],
     ["FACE_CREASE_OUTER_R", "FACE_NOSE_BRIDGE_R"]),

    # ── NOSE ──────────────────────────────────────────────────────────────
    (["nose",    "nose.001"],    ["FACE_NOSE_BRIDGE",    "FACE_NOSE_TIP"]),
    # nose.002 head=NOSE_TIP, nose.003 tail=NOSE_BOT (connected pair)
    (["nose.002", "nose.003"],   ["FACE_NOSE_TIP", "FACE_NOSE_BOT"]),
    # nose.004 head=NOSE_BOT, nose_end_glue.004 tail=LIP_T (equal lengths, even)
    (["nose.004", "nose_end_glue.004"], ["FACE_NOSE_BOT", "FACE_LIP_T"], "even"),
    # nose.L / nose.L.002 — even between NOSE_BRIDGE and NOSE_WING
    (["nose.L", "nose.L.002"], ["FACE_NOSE_BRIDGE_L", "FACE_NOSE_WING_L"], "even"),
    (["nose.R", "nose.R.002"], ["FACE_NOSE_BRIDGE_R", "FACE_NOSE_WING_R"], "even"),
    # nose.L.001 / nose.R.001 — independent: head=NOSE_WING, tail=NOSE_TIP
    (["nose.L.001"], ["FACE_NOSE_WING_L", "FACE_NOSE_TIP"]),
    (["nose.R.001"], ["FACE_NOSE_WING_R", "FACE_NOSE_TIP"]),

    # ── CHEEK BONES ───────────────────────────────────────────────────────
    (["cheek_glue.T.L.001"], ["FACE_CHEEK_TOP_L", "FACE_CHEEK_L"]),
    (["cheek_glue.T.R.001"], ["FACE_CHEEK_TOP_R", "FACE_CHEEK_R"]),
    (["cheek.T.L.001"],      ["FACE_CHEEK_TOP_L", "FACE_NOSE_BRIDGE_L"]),
    (["cheek.T.R.001"],      ["FACE_CHEEK_TOP_R", "FACE_NOSE_BRIDGE_R"]),

    # ── FOREHEAD / BROW CENTRE BONES ─────────────────────────────────────
    (["forehead.T.004"], ["FACE_FOREHEAD", "FACE_BROW"]),
    (["brow.T.005"],     ["FACE_BROW",     "FACE_NOSE_BRIDGE"]),
    (["brow.T.L.005"],   ["FACE_BROW",     "FACE_BROW_1_L"]),
    (["brow.T.R.005"],   ["FACE_BROW",     "FACE_BROW_1_R"]),

    # ── LID GLUE BONES ───────────────────────────────────────────────────
    (["lid_glue.B.L.002"], ["FACE_EYE_BOT_L", "FACE_CHEEK_TOP_L"]),
    (["lid_glue.B.R.002"], ["FACE_EYE_BOT_R", "FACE_CHEEK_TOP_R"]),
    (["lid_glue.B.L.003"], ["FACE_EYE_BOT_L", "FACE_CHEEK_TOP_L"]),
    (["lid_glue.B.R.003"], ["FACE_EYE_BOT_R", "FACE_CHEEK_TOP_R"]),

    # ── UPPER LID — two arc-preserved 2-bone halves; apex joint pins to EYE_TOP ─
    (["lid.T.L", "lid.T.L.001"],    ["FACE_EYE_OUTER_L", "FACE_EYE_TOP_L"  ]),
    (["lid.T.L.002", "lid.T.L.003"],["FACE_EYE_TOP_L",   "FACE_EYE_INNER_L"]),
    (["lid.T.R", "lid.T.R.001"],    ["FACE_EYE_OUTER_R", "FACE_EYE_TOP_R"  ]),
    (["lid.T.R.002", "lid.T.R.003"],["FACE_EYE_TOP_R",   "FACE_EYE_INNER_R"]),
    # ── LOWER LID — two arc-preserved 2-bone halves; apex joint pins to EYE_BOT ─
    (["lid.B.L", "lid.B.L.001"],    ["FACE_EYE_INNER_L", "FACE_EYE_BOT_L"  ]),
    (["lid.B.L.002", "lid.B.L.003"],["FACE_EYE_BOT_L",   "FACE_EYE_OUTER_L"]),
    (["lid.B.R", "lid.B.R.001"],    ["FACE_EYE_INNER_R", "FACE_EYE_BOT_R"  ]),
    (["lid.B.R.002", "lid.B.R.003"],["FACE_EYE_BOT_R",   "FACE_EYE_OUTER_R"]),

    # ── UPPER LIP — 3 explicit joints per side ────────────────────────────
    (["lip.T.L", "lip.T.L.001"], ["FACE_LIP_T", "FACE_MOUTH_TOP_L", "FACE_MOUTH_CORNER_L"]),
    (["lip.T.R", "lip.T.R.001"], ["FACE_LIP_T", "FACE_MOUTH_TOP_R", "FACE_MOUTH_CORNER_R"]),

    # ── LOWER LIP — 3 explicit joints per side ────────────────────────────
    (["lip.B.L", "lip.B.L.001"], ["FACE_LIP_B", "FACE_MOUTH_BOT_L", "FACE_MOUTH_CORNER_L"]),
    (["lip.B.R", "lip.B.R.001"], ["FACE_LIP_B", "FACE_MOUTH_BOT_R", "FACE_MOUTH_CORNER_R"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# FINGER BONE TABLE
# ─────────────────────────────────────────────────────────────────────────────
FINGER_BONES_L = [
    ("thumb.01.L",    "THUMB_1",          "THUMB_2"),
    ("thumb.02.L",    "THUMB_2",          "THUMB_3"),
    ("thumb.03.L",    "THUMB_3",          "THUMB_TIP"),
    ("f_index.01.L",  "FINGER_INDEX_1",   "FINGER_INDEX_2"),
    ("f_index.02.L",  "FINGER_INDEX_2",   "FINGER_INDEX_3"),
    ("f_index.03.L",  "FINGER_INDEX_3",   "FINGER_INDEX_TIP"),
    ("f_middle.01.L", "FINGER_MIDDLE_1",  "FINGER_MIDDLE_2"),
    ("f_middle.02.L", "FINGER_MIDDLE_2",  "FINGER_MIDDLE_3"),
    ("f_middle.03.L", "FINGER_MIDDLE_3",  "FINGER_MIDDLE_TIP"),
    ("f_ring.01.L",   "FINGER_RING_1",    "FINGER_RING_2"),
    ("f_ring.02.L",   "FINGER_RING_2",    "FINGER_RING_3"),
    ("f_ring.03.L",   "FINGER_RING_3",    "FINGER_RING_TIP"),
    ("f_pinky.01.L",  "FINGER_PINKY_1",   "FINGER_PINKY_2"),
    ("f_pinky.02.L",  "FINGER_PINKY_2",   "FINGER_PINKY_3"),
    ("f_pinky.03.L",  "FINGER_PINKY_3",   "FINGER_PINKY_TIP"),
]

PALM_BONES_L = [
    ("palm.01.L", "FINGER_INDEX_1"),
    ("palm.02.L", "FINGER_MIDDLE_1"),
    ("palm.03.L", "FINGER_RING_1"),
    ("palm.04.L", "FINGER_PINKY_1"),
]

# ─────────────────────────────────────────────────────────────────────────────
# BONE ROLL RULES
# Each entry: (prefixes_tuple, align_roll_vector, extra_radians)
#
# align_roll(vec) sets the bone's local Z as close to `vec` as possible
# while keeping local Y fixed (head→tail direction).
#
# Rigify A-pose conventions:
#   Arm chain   (upper_arm, forearm, hand): align -Y → Z forward, X down
#   Leg chain   (thigh, shin):              align +Y → Z backward
#   Foot, Toe:                              align +Z → Z up
#   Spine/body:                             align +Z → Z up
#   Fingers/palms: NOT in this table — handled separately in AlignRig
#
# NOTE: foot is NOT in the thigh/shin group. Those are vertical bones, so Z→+Y
# (backward) is well-defined. The foot bone points forward/down toward the toes
# (~along -Y), so aligning its Z to +Y is (anti)parallel to the bone direction →
# align_roll degenerates and the foot twists. A flat foot wants Z up (+Z), same
# as the toe.
# ─────────────────────────────────────────────────────────────────────────────
ROLL_RULES = [
    (("upper_arm.", "forearm.", "hand."), Vector((0, -1, 0)),  0.0),
    (("shoulder.",),                      Vector((0,  0, 1)),  0.0),
    (("thigh.", "shin."),                 Vector((0,  1, 0)),  0.0),
    (("toe.", "foot."),                   Vector((0,  0, 1)),  0.0),
]
