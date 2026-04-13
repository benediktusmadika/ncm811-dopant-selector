from __future__ import annotations

import re

LOGGER_NAME = "ncm811_dopant_selector"

HOST_ELEMENTS = ("Li", "Ni", "Co", "Mn", "O")
HOST_SET = set(HOST_ELEMENTS)

_ELEMENT_SYMBOLS = {
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og",
}

_ATOMIC_WEIGHT = {
    "H": 1.00794, "He": 4.002602,
    "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.0983, "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293,
    "Cs": 132.90545196, "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93033,
    "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668,
    "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040,
    "Th": 232.0377, "U": 238.02891,
}

_ELEMENT_TOKEN_RE = re.compile(r"([A-Z][a-z]?)")

DEFAULT_SEED = 42
SUPPORTED_REASONING_EFFORTS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_JSON_OUTPUT_TOKENS_CAP = 8192
JSON_REPAIR_TRIM_WINDOW_CHARS = 2048
JSON_REPAIR_MAX_TRIM_CANDIDATES = 72
JSON_COMPACT_RETRY_SUFFIX = (
    "\n\nKeep the JSON concise to fit the token budget: use short phrases, avoid repetition, "
    "and keep free-text fields compact while preserving the same meaning."
)

KNOWN_MODEL_PREFIXES = (
    "gpt-5.4-pro",
    "gpt-5-pro",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
    "o3-mini",
    "o3",
    "o1-mini",
    "o1-preview",
    "o1",
)

MECH_CATEGORIES = [
    "cation_mixing_reduction",
    "oxygen_release_suppression",
    "surface_reconstruction_suppression",
    "interfacial_stability",
    "electronic_conductivity",
    "ionic_conductivity",
    "lattice_stabilization",
    "particle_morphology_control",
]
