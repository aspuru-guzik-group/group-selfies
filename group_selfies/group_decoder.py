from itertools import chain
from group_selfies.grammar_rules import (
    next_atom_state,
    next_branch_state,
    next_ring_state,
    process_branch_symbol,
    process_ring_symbol,
    get_index_from_selfies,
    process_atom_symbol,
    bond_char_to_order
)

class DecoderError(Exception):
    pass

from group_selfies.utils.selfies_utils import split_selfies
def _tokenize_selfies(selfies):
    if isinstance(selfies, str):
        symbol_iter = split_selfies(selfies)
    elif isinstance(selfies, list):
        symbol_iter = selfies
    else:
        raise ValueError()  # should not happen

    try:
        for symbol in symbol_iter:
            if symbol == "[nop]":
                continue
            yield symbol
    except ValueError as err:
        raise DecoderError(str(err)) from None

def _raise_decoder_error(selfies, invalid_symbol):
    err_msg = "invalid symbol '{}'\n\tSELFIES: {}".format(
        invalid_symbol, selfies
    )
    raise DecoderError(err_msg)

from group_selfies.group_mol_graph import Atom, MolecularGraph
import re
from rdkit import Chem, RDLogger
from group_selfies.constants import POP_SYMBOL, NEGATIVE_RING_SYMBOL, BOND_DIR, INVERT_DIR
from group_selfies.bond_constraints import get_bonding_capacity
from collections import defaultdict

def _read_index_from_selfies(symbol_iter, n_symbols):
    index_symbols = []
    for _ in range(n_symbols):
        try:
            next_one = next(symbol_iter)
            if next_one == POP_SYMBOL:
                return None
            index_symbols.append(next_one)
        except StopIteration:
            index_symbols.append(None)
    return get_index_from_selfies(*index_symbols)

def parse_ring_length(symbol_iter, n_symbols):
    try:
        next_one = next(symbol_iter)
        if next_one == POP_SYMBOL:
            return None
        
        negative_ring = False
        if next_one == NEGATIVE_RING_SYMBOL:
            Q = _read_index_from_selfies(symbol_iter, n_symbols)
            if Q is None:
                return None
            negative_ring = True
        else:
            Q = _read_index_from_selfies(
                chain([next_one], symbol_iter), n_symbols)
            if Q is None:
                return None
        return Q, negative_ring
    except StopIteration:
        return None

def update_member_idxs(member_idxs, attachment_point_idx):
    new_member_idxs = []
    for i, idx in enumerate(member_idxs):
        if idx is None:
            new_member_idxs.append(None)
        elif i == attachment_point_idx:
            new_member_idxs.append(None)
        elif i > attachment_point_idx:
            new_member_idxs.append(idx-1)
        else:
            new_member_idxs.append(idx)
    return new_member_idxs

class Counter:
    def __init__(self, value):
        self.value = value

    def get(self):
        old = self.value
        self.value += 1
        return old

def find_next_available(current_idx, rel_idx, member_idxs, attachment_points):
    current = (current_idx + rel_idx) % len(attachment_points)
    for shift in range(len(attachment_points)):
        inner = (current + shift) % len(attachment_points)
        if member_idxs[attachment_points[inner]] is not None:
            return inner
    return None

bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }

def set_id(atom, idx):
    if atom.HasProp('id'):
        atom.SetProp('id', f"{atom.GetProp('id')},{idx}")
    else:
        atom.SetProp('id', str(idx))


_exhausted = object()
def selfies_to_graph_iterative(grammar, 
                                symbol_iter, 
                                selfies=None,
                                rings=None, 
                                dummy_counter=None, 
                                place_from_idx=None, 
                                inverse_place=None,
                                previous_att_idx=None, 
                                group_atom=None,
                                verbose=False):
    mol = Chem.RWMol()
    stack = [(None, 0, False)]
    while (symbol := next(symbol_iter, _exhausted)) != _exhausted:
        if not len(stack):
            break
        merged_state = stack.pop()
        if symbol == POP_SYMBOL:
            if not len(stack):
                stack.append(merged_state)
            continue
        prev, state, att, *additional_info = merged_state
        if state is None:
            symbol_iter = chain([symbol], symbol_iter)
            continue
        new_state = merged_state
        for _ in range(1):
            if prev != -1:
                # Case 0: Normal state
                if ":" == symbol[1]:
                    # Subcase 0.0: Group symbol (e.g. [:benzene]) 
                    absolute = False
                    group_idx = -1
                    start_idx = 2
                    bond_order = 1
                    bond_dir = Chem.BondDir.NONE

                    # probably inefficient
                    # maybe do this manually instead of using regex
                    remaining = symbol[2:]
                    idx_match = re.search(r'\d+', remaining)
                    dir_match = re.search(r'[\\/]', remaining)
                    order_match = re.search(r'[#\=\-]', remaining)
                    if idx_match:
                        group_idx = int(res := idx_match.group(0))
                        start_idx += len(res)
                    else:
                        group_idx = 0
                    if dir_match:
                        bond_dir = dir_match.group(0)
                        start_idx += 1
                    if order_match:
                        bond_order = bond_char_to_order[order_match.group(0)]
                        start_idx += 1
                    
                    ####### add group
                    group_name = symbol[start_idx:-1]
                    group = grammar.get_group(group_name)

                    if prev is None:
                        absolute = True

                    # place a dummy atom on the main chain
                    if not absolute:
                        prev_atom = mol.GetAtomWithIdx(prev)
                        prev_parent = prev_atom
                        if (bond_order := min(state, bond_order, group.attachment_valency(group_idx))): #add group only if possible
                            atom_map_num = dummy_counter.get()
                            if not att:
                                dummy = Chem.Atom('*')
                                dummy.SetAtomMapNum(atom_map_num)
                                mol.AddAtom(dummy)
                                b_idx = mol.AddBond(prev, mol.GetNumAtoms()-1, bond_types[bond_order])
                            else:
                                prev_parent = prev_atom.GetNeighbors()[0]
                                prev_atom.SetAtomMapNum(atom_map_num)
                                prev_atom.GetBonds()[0].SetBondType(bond_types[bond_order])
                        else:
                            continue
                        set_id(prev_parent, atom_map_num)
                        prev_parent.SetProp(f'bond_dir{atom_map_num}', str(bond_dir))

                    # === add the group atoms and bonds ===
                    previous_size = mol.GetNumAtoms()
                    mol.InsertMol(group.mol)
                    if not absolute:
                        mol.GetAtomWithIdx(previous_size + group.attachment_points[group_idx]).SetAtomMapNum(atom_map_num)
                        att_global_idx = previous_size + group.attachment_points[group_idx]
                        parent_global_idx = previous_size + group.parent_map[group.attachment_points[group_idx]]
                        
                        att_parent = mol.GetAtomWithIdx(parent_global_idx)
                        set_id(att_parent, atom_map_num)
                        att_parent.SetProp(f'bond_dir{atom_map_num}', str(INVERT_DIR[bond_dir]))

                        att_bond = mol.GetBondBetweenAtoms(previous_size + group.attachment_points[group_idx], previous_size + group.parent_map[group.attachment_points[group_idx]])
                        att_bond.SetBondType(bond_types[bond_order])

                    member_idxs = [i+previous_size for i in range(group.mol.GetNumAtoms())]
                    for i, att in enumerate(group.attachment_points):
                        if member_idxs[att] not in place_from_idx:
                            place_from_idx[member_idxs[att]] = len(place_from_idx)
                            inverse_place.append(member_idxs[att])
                            group_atom[member_idxs[att]] = (group, i)
                    if not absolute:
                        member_idxs[group.attachment_points[group_idx]] = None
                    else:
                        group_idx = 0
                    new_state = (-1, group, False, member_idxs, group_idx)

                elif "ch" == symbol[-3:-1]:
                    # Subcase 0.1: Branch symbol (e.g. [Branch])
                    if (att or state <= 1): 
                        continue
                    btype = process_branch_symbol(symbol)
                    if btype is None:
                        _raise_decoder_error(selfies, symbol)
                    
                    binit_state, next_state = next_branch_state(btype, state)
                    stack.append((prev, next_state, att, *additional_info))
                    new_state = (prev, binit_state, att, *additional_info)
                    
                elif "ng" == symbol[-4:-2]:
                    # Subcase 0.2: Ring symbol (e.g. [Ring2])
                    output = process_ring_symbol(symbol)
                    if output is None:
                        _raise_decoder_error(selfies, symbol)
                    ring_type, n, stereo = output

                    if state == 0:
                        next_state = state
                    else:
                        ring_order, next_state = next_ring_state(ring_type, state)
                        bond_info = (ring_order, stereo)
                        output = parse_ring_length(symbol_iter, n_symbols=n)
                        if output is None:
                            # end of string, or pop
                            if not len(stack):
                                stack.append(merged_state)
                            break
                        Q, negative_ring = output

                        if negative_ring:
                            lookup = -(Q + 1)
                        else:
                            lookup = Q + 1
                        ridx = prev
                        lplace = min(max(place_from_idx[ridx] - lookup, 0), len(place_from_idx)-1)
                        lidx = inverse_place[lplace]
                        # ridx and lidx are atom indices
                        # ring bond is from lidx <- ridx
                        rings.append((prev, lidx, bond_info))
                        new_state = (prev, next_state, att, *additional_info)
                        
                elif "[epsilon]" == symbol:
                    # Subcase 0.3: [epsilon]
                    pass
                elif symbol == NEGATIVE_RING_SYMBOL:
                    pass
                else:
                    bond_order, bond_dir, atom = process_atom_symbol(symbol)
                    cap = get_bonding_capacity(atom.element, atom.charge)
                    cap -= 0 if (atom.h_count is None) else atom.h_count
                    bond_order, next_state = next_atom_state(bond_order, cap, state)
                    new_at = Chem.Atom(atom.element)
                    new_at.SetFormalCharge(atom.charge) 
                    if atom.isotope:
                        new_at.SetIsotope(atom.isotope)
                    if atom.radical:
                        new_at.SetNumRadicalElectrons(atom.radical)
                    added_atom = mol.AddAtom(new_at)
                    
                    if atom.h_count is not None:
                        mol.GetAtomWithIdx(added_atom).SetNumExplicitHs(atom.h_count)
                        mol.GetAtomWithIdx(added_atom).SetNoImplicit(True)

                    place_from_idx[added_atom] = len(place_from_idx)
                    inverse_place.append(added_atom)
                    
                    if bond_order != 0:
                        atom_map_num = dummy_counter.get()
                        set_id(mol.GetAtomWithIdx(added_atom), atom_map_num)
                        mol.GetAtomWithIdx(added_atom).SetProp(f'bond_dir{atom_map_num}', str(INVERT_DIR[bond_dir]))
                        prev_parent = mol.GetAtomWithIdx(prev)
                        # not the starting atom, so a previous bond exists
                        if not att:
                            b_idx = mol.AddBond(prev, added_atom, bond_types[bond_order])
                        else:
                            prev_atom = mol.GetAtomWithIdx(prev)
                            prev_atom.SetAtomMapNum(atom_map_num)

                            prev_parent = prev_atom.GetNeighbors()[0]

                            dummy = Chem.Atom('*')
                            dummy.SetAtomMapNum(atom_map_num)
                            dummy_idx = mol.AddAtom(dummy)
                            b_idx = mol.AddBond(dummy_idx, added_atom, bond_types[bond_order])
                            b_prev = prev_atom.GetBonds()[0]
                            b_prev.SetBondType(bond_types[bond_order])

                        set_id(prev_parent, atom_map_num)
                        prev_parent.SetProp(f'bond_dir{atom_map_num}', str(bond_dir))
                    new_state = (added_atom, next_state, False)
            else:
                # Case 1: Group
                group = state
                rel_idx = grammar.get_index_from_selfies(symbol)
                member_idxs, inner_att = additional_info
                possible_inner_att = find_next_available(inner_att, rel_idx, member_idxs, group.attachment_points)
                if possible_inner_att is None:
                    break
                inner_att = possible_inner_att
                next_state = group.attachment_valency(inner_att)
                prev = member_idxs[group.attachment_points[inner_att]]
                member_idxs[group.attachment_points[inner_att]] = None
                stack.append((-1, group, False, member_idxs, inner_att))
                new_state = (prev, next_state, True)
        else:
            stack.append(new_state)
    return mol

def available_bonds(atom):
    atom.UpdatePropertyCache()
    return get_bonding_capacity(atom.GetSymbol(), atom.GetFormalCharge()) - atom.GetExplicitValence()

def form_rings_bilocally_iterative(mol, rings, place_from_idx, inverse_place, dummy_counter, group_atom, verbose=False):
    for ridx, lidx, bond_info in rings:
        # clamp lidx to inside the graph
        lplace = min(place_from_idx[lidx], len(place_from_idx)-1)
        lidx = min(lidx, inverse_place[lplace])
        if lidx == ridx:  # ring to the same atom forbidden
            continue
        order, (lstereo, rstereo) = bond_info

        latom = mol.GetAtomWithIdx(lidx)
        ratom = mol.GetAtomWithIdx(ridx)
        cont = False
        for lr_atom, lr_idx in [(latom, lidx), (ratom, ridx)]:
            if (lr_atom.GetSymbol() == '*' and lr_atom.GetAtomMapNum() != 0):
                cont = True
                break
        if cont:
            continue
        if latom.GetSymbol() != '*' and ratom.GetSymbol() != '*':
            # atom to atom
            lfree = available_bonds(latom)
            rfree = available_bonds(ratom)

            if lfree <= 0 or rfree <= 0:
                continue  # no room for ring bond

            order = min(order, lfree, rfree)
            lr_bond = mol.GetBondBetweenAtoms(lidx, ridx)

            if lr_bond is None:
                formed = mol.AddBond(ridx, lidx, bond_types[order])
            else:
                new_order = min(order + int(lr_bond.GetBondTypeAsDouble()), 3)
                lr_bond.SetBondType(bond_types[new_order])
            
            id_num = dummy_counter.get()
            set_id(latom, id_num)
            set_id(ratom, id_num)
            latom.SetProp(f'bond_dir{id_num}', str(INVERT_DIR[lstereo]))
            ratom.SetProp(f'bond_dir{id_num}', str(lstereo))

        elif [latom.GetSymbol(), ratom.GetSymbol()].count('*') == 1:
            free = order
            attachment = 0
            for i, lr_atom in enumerate([latom, ratom]):
                if lr_atom.GetSymbol() == '*':
                    attachment = i
                    if lr_atom.GetIdx() in group_atom:
                        group_t, att_idx = group_atom[lr_atom.GetIdx()]
                        free = min(free, group_t.attachment_valency(att_idx))
                else:
                    free = min(available_bonds(lr_atom), free)
            if free <= 0:
                continue
            dummy = Chem.Atom('*')
            atom_map_num = dummy_counter.get()
            dummy.SetAtomMapNum(atom_map_num)
            mol.AddAtom(dummy)
            att_atom = [latom, ratom][attachment]
            normal_atom = [latom, ratom][1-attachment]
            att_parent = att_atom.GetNeighbors()[0]
            
            set_id(att_parent, atom_map_num)
            att_parent.SetProp(f'bond_dir{atom_map_num}', str(INVERT_DIR[lstereo] if attachment == 0 else lstereo))
            set_id(normal_atom, atom_map_num)
            normal_atom.SetProp(f'bond_dir{atom_map_num}', str(INVERT_DIR[lstereo] if attachment == 1 else lstereo))

            if attachment == 0:
                formed = mol.AddBond(ridx, mol.GetNumAtoms()-1, bond_types[free])
            else:
                formed = mol.AddBond(mol.GetNumAtoms()-1, lidx, bond_types[free])
            
            att_atom.GetBonds()[0].SetBondType(bond_types[free])
            att_atom.SetAtomMapNum(atom_map_num)
            
        else:            
            free = order
            for lr_atom in [latom, ratom]:
                if lr_atom.GetIdx() in group_atom:
                    group_t, att_idx = group_atom[lr_atom.GetIdx()]
                    free = min(free, group_t.attachment_valency(att_idx))
            atom_map_num = dummy_counter.get()

            l_parent, r_parent = [lr_atom.GetNeighbors()[0] for lr_atom in [latom, ratom]]
            set_id(l_parent, atom_map_num)
            set_id(r_parent, atom_map_num)

            l_parent.SetProp(f'bond_dir{atom_map_num}', str(INVERT_DIR[lstereo]))
            r_parent.SetProp(f'bond_dir{atom_map_num}', str(lstereo))

            latom.SetAtomMapNum(atom_map_num)
            ratom.SetAtomMapNum(atom_map_num)
            latom.GetBonds()[0].SetBondType(bond_types[free])
            ratom.GetBonds()[0].SetBondType(bond_types[free])
            
def group_decoder(grammar, selfies, verbose=False):
    if not selfies:
        return Chem.Mol()
    try:
        rings = []
        place_from_idx = dict()
        inverse_place = []
        dummy_counter = Counter(1)
        group_atom = dict()
        mol = selfies_to_graph_iterative(
            grammar=grammar,
            symbol_iter=_tokenize_selfies(selfies),
            selfies=selfies,
            rings=rings,
            dummy_counter=dummy_counter,
            place_from_idx=place_from_idx,
            inverse_place=inverse_place,
            verbose=verbose,
            group_atom=group_atom
        )

        form_rings_bilocally_iterative(mol, rings, place_from_idx, inverse_place, dummy_counter, group_atom, verbose=verbose)
        try:
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol, force=True)
        except:
            raise ValueError('Issue assigning double bond stereochem')

        remove_maps = []
        atom_pair = set()
        map_idx = dict()
        # remove multiple bonds between the same pairs of atoms
        # assumes that mapped atoms are attachment points + map indices only used once
        at_map = defaultdict(int) # cache of atom maps
        for at_idx, atom in enumerate(mol.GetAtoms()):
            if map_num := atom.GetAtomMapNum():
                at_map[at_idx] = map_num
                neighbor = atom.GetNeighbors()[0].GetIdx()
                if map_num in map_idx:
                    att_idx, parent_idx = map_idx[map_num]
                    to_add = tuple(sorted([parent_idx, neighbor]))
                    if (to_add in atom_pair) or (mol.GetBondBetweenAtoms(parent_idx, neighbor) is not None) or (parent_idx == neighbor):
                        remove_maps.extend([att_idx, at_idx])
                    else:
                        atom_pair.add(to_add)
                else:
                    map_idx[map_num] = (at_idx, neighbor)

        for idx in remove_maps:
            at_map[idx] = 0
            mol.GetAtomWithIdx(idx).SetAtomMapNum(0) 

        # zip molecule together sequentially to preserve stereochemistry

        frag_atom = []
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=frag_atom)
        if len(frags) == 0:
            res = Chem.RWMol(Chem.molzip(mol))
        else:
            frags = [(frag, set([res for at_idx in frag_atom[i] if (res := at_map[at_idx]) != 0])) for i, frag in enumerate(frags)]
            first = frags.pop(0)
            res = first[0]
            running_set = first[1]

            params = Chem.MolzipParams()
            params.enforceValenceRules = False
            escape = False
            while len(frags):
                for i in range(len(frags)):
                    if len(running_set & frags[i][1]) != 0:
                        frag, maps = frags.pop(i)
                        running_set |= maps
                        try:
                            res = Chem.molzip(res, frag, params)
                        except:
                            escape = True
                        break
                else:
                    break
                if escape:
                    break
            try:
                res = Chem.RWMol(Chem.molzip(res, params))
            except:
                res = Chem.RWMol(res)
        # use batch edit to avoid stereochemistry loss
        idxs = []
        res.BeginBatchEdit()
        parents = {}
        for idx, atom in enumerate(res.GetAtoms()):
            if atom.GetSymbol() == '*':
                if (neigh := atom.GetNeighbors()[0]).GetIdx() not in parents:
                    parents[neigh.GetIdx()] = [neigh, 0]
                parents[neigh.GetIdx()][1] += 1
                res.RemoveAtom(idx)
        res.CommitBatchEdit()  

        # set bond dirs correctly
        bonds = defaultdict(list)
        for atom in res.GetAtoms():
            if atom.HasProp('id'):
                for idx in atom.GetProp('id').split(','):
                    bonds[idx].append(atom)

        for b_idx, ats in bonds.items():
            if len(ats) < 2:
                continue
            a1, a2 = ats
            b = res.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
            if b is None:
                continue
            if b.GetBeginAtomIdx() == a1.GetIdx():
                b.SetBondDir(BOND_DIR[a1.GetProp(f'bond_dir{b_idx}')])
            else:
                b.SetBondDir(BOND_DIR[a2.GetProp(f'bond_dir{b_idx}')])

        for parent, count in parents.values():
            parent.SetNumExplicitHs(parent.GetNumExplicitHs() + count)

        res.UpdatePropertyCache()
        Chem.AssignStereochemistry(res, force=True)
        Chem.SanitizeMol(res)
        # res = res.GetMol()
        return res
    except Exception as e:
        print(e)
        raise ValueError(f'CAN NOT DECODE {selfies}')