import functools
import itertools
import re
from typing import Any, List, Optional, Tuple

from group_selfies.constants import (
    ELEMENTS,
    INDEX_ALPHABET,
    INDEX_CODE,
    ORGANIC_SUBSET
)
from group_selfies.group_mol_graph import Atom


def process_atom_symbol(symbol: str) -> Optional[Tuple[Any, Atom]]:
    try:
        output = _PROCESS_ATOM_CACHE[symbol]
    except KeyError:
        output = _process_atom_symbol_no_cache(symbol)
        if output is None:
            return None
        _PROCESS_ATOM_CACHE[symbol] = output
    return output


def process_branch_symbol(symbol: str) -> Optional[Tuple[int, int]]:
    try:
        return _PROCESS_BRANCH_CACHE[symbol]
    except KeyError:
        return None


def process_ring_symbol(symbol: str) -> Optional[Tuple[int, int, Any]]:
    try:
        return _PROCESS_RING_CACHE[symbol]
    except KeyError:
        return None


def next_atom_state(
        bond_order: int, bond_cap: int, state: int
) -> Tuple[int, Optional[int]]:
    if state == 0:
        bond_order = 0
    bond_order = min(bond_order, state, bond_cap)
    bonds_left = bond_cap - bond_order
    next_state = None if (bonds_left == 0) else bonds_left
    return bond_order, next_state


def next_branch_state(
        branch_type: int, state: int
) -> Tuple[int, Optional[int]]:
    assert 1 <= branch_type <= 3
    assert state > 1

    branch_init_state = min(state - 1, branch_type)
    next_state = state - branch_init_state
    return branch_init_state, next_state


def next_ring_state(
        ring_type: int, state: int
) -> Tuple[int, Optional[int]]:
    assert state > 0

    bond_order = min(ring_type, state)
    bonds_left = state - bond_order
    next_state = None if (bonds_left == 0) else bonds_left
    return bond_order, next_state


def get_index_from_selfies(*symbols: List[str]) -> int:
    index = 0
    for i, c in enumerate(reversed(symbols)):
        index += INDEX_CODE.get(c, 0) * (len(INDEX_CODE) ** i)
    return index


def get_selfies_from_index(index: int) -> List[str]:
    if index < 0:
        raise IndexError()
    elif index == 0:
        return [INDEX_ALPHABET[0]]

    symbols = []
    base = len(INDEX_ALPHABET)
    while index:
        symbols.append(INDEX_ALPHABET[index % base])
        index //= base
    return symbols[::-1]


# =============================================================================
# Caches (for computational speed)
# =============================================================================


SELFIES_ATOM_PATTERN = re.compile(
    r"^[\[]"  # opening square bracket [
    r"([=#/\\]?)"  # bond char
    r"(\d*)"  # isotope number (optional, e.g. 123, 26)
    r"([A-Z][a-z]?)"  # element symbol
    r"([@]{0,2})"  # chiral_tag (optional, only @ and @@ supported)
    r"((?:[H]\d)?)"  # H count (optional, e.g. H1, H3)
    r"((?:[+-][1-9]+)?)"  # charge (optional, e.g. +1)
    r"((?:[R]\d)?)" #radical count (optional)
    r"[]]$"  # closing square bracket ]
)

bond_char_to_order = {
    "": 1,
    "=": 2,
    "#": 3,
}

def _process_atom_symbol_no_cache(symbol):
    m = SELFIES_ATOM_PATTERN.match(symbol)
    assert m is not None, f"{symbol} did not match any atom type"
    bond_char, isotope, element, chirality, h_count, charge, radical = m.groups()
    bond_dir = None
    dirs = ['\\', '/']
    for dire in dirs:
        if dire in bond_char:
            bond_dir = dire
            bond_char = bond_char.replace(dire, '')
            break
    bond_order = bond_char_to_order[bond_char]
    atom = Atom(element, charge)
    if symbol[1 + len(bond_char) + (len(bond_dir) if bond_dir is not None else 0): -1] in ORGANIC_SUBSET:
        atom = Atom(element=element)
        return bond_order, bond_dir, atom
    isotope = None if (isotope == "") else int(isotope)
    if element not in ELEMENTS:
        return None
    chirality = None if (chirality == "") else chirality
    s = h_count
    if s == "":
        h_count = None
    else:
        h_count = int(s[1:])
    s = charge
    if s == "":
        charge = 0
    else:
        charge = int(s[1:])
        charge *= 1 if (s[0] == "+") else -1
    if radical == "":
        radical = 0
    else:
        radical = int(radical[1:])
    atom = Atom(
        element=element,
        isotope=isotope,
        h_count=h_count,
        charge=charge,
        radical=radical,
    )
    return bond_order, bond_dir, atom

def _build_atom_cache():
    cache = dict()
    common_symbols = [
        "[#C+1]", "[#C-1]", "[#C]", "[#N+1]", "[#N]", "[#O+1]", "[#P+1]",
        "[#P-1]", "[#P]", "[#S+1]", "[#S-1]", "[#S]", "[=C+1]", "[=C-1]",
        "[=C]", "[=N+1]", "[=N-1]", "[=N]", "[=O+1]", "[=O]", "[=P+1]",
        "[=P-1]", "[=P]", "[=S+1]", "[=S-1]", "[=S]", "[Br]", "[C+1]", "[C-1]",
        "[C]", "[Cl]", "[F]", "[H]", "[I]", "[N+1]", "[N-1]", "[N]", "[O+1]",
        "[O-1]", "[O]", "[P+1]", "[P-1]", "[P]", "[S+1]", "[S-1]", "[S]"
    ]

    for symbol in common_symbols:
        cache[symbol] = _process_atom_symbol_no_cache(symbol)
    return cache


def _build_branch_cache():
    cache = dict()
    for bond_char in ["", "=", "#"]:
        symbol = "[{}Branch]".format(bond_char)
        cache[symbol] = bond_char_to_order[bond_char]
    return cache


def _build_ring_cache():
    cache = dict()
    for L in range(1, 4):
        # [RingL], [=RingL], [#RingL]
        for bond_char in ["", "=", "#"]:
            symbol = "[{}Ring{}]".format(bond_char, L)
            order = bond_char_to_order[bond_char]
            cache[symbol] = (order, L, (None, None))

        # [-/RingL], [\/RingL], [\-RingL], ...
        for stereochar in ['/', '\\']:
            symbol = "[{}Ring{}]".format(stereochar, L)
            cache[symbol] = (1, L, (stereochar, stereochar))
    return cache


_PROCESS_ATOM_CACHE = _build_atom_cache()

_PROCESS_BRANCH_CACHE = _build_branch_cache()

_PROCESS_RING_CACHE = _build_ring_cache()
