from tkinter import E
import networkx as nx
def to_nx(mol_graph):
    G = nx.Graph()

    for atom in mol_graph._atoms:
        G.add_node(atom.index, name=atom.element, group_obj=None)
        # G.add_node(atom.index, name=atom.element, **atom.__dict__)

    for u, v in mol_graph._bond_dict.keys():
        G.add_edge(u, v, order=mol_graph._bond_dict[(u, v)].order)
    return G

import logging
def loginfo(*args):
    logging.info('%s '*len(args), *args)

def loginfo_mol(mol):
    for atom in mol._atoms:
        loginfo(atom.index, atom.element)
        for bond in mol.get_out_dirbonds(atom.index):
            loginfo('  ', bond.src, bond.dst, bond.order, bond.ring_bond)

from rdkit import Chem
from rdkit.Chem import AllChem
import re
from collections import defaultdict
from ..bond_constraints import get_bonding_capacity

def verify_group(pat, group=False):
    potential_val = defaultdict(int)
    for atom in pat.GetAtoms(): # essentially the same checks found in group_parser
        if atom.GetAtomMapNum():
            parent = atom.GetNeighbors()
            # print(parent)
            assert len(parent) == 1, 'More than one parent for attachment point.'
            valency = atom.GetAtomMapNum()
            parent = parent[0]
            assert valency > 0, 'Can not have non positive valency.'
            if group:
                potential_val[parent.GetIdx()] += 0 if valency == 1 else valency #"unsafe" temporary patch; only works because of RDKit behaviour; 
                #single bonds contribute one valency already due to their bonds, but RDkit treats query bonds with multiple possible bond orders as contributing 0 valency
            else:
                potential_val[parent.GetIdx()] += valency - 1 #attachment points always single bonded

            atom.SetIntProp('valAvailable', valency)
            atom.SetProp('_displayLabel', f'*{valency}')
            atom.SetAtomMapNum(0)
    for k, v in potential_val.items():
        a = pat.GetAtomWithIdx(k)
        assert get_bonding_capacity(a.GetSymbol(), a.GetFormalCharge()) >= sum([bond.GetValenceContrib(a) for bond in a.GetBonds()  if (not group) or (bond.GetBeginAtom().GetAtomicNum() and bond.GetEndAtom().GetAtomicNum())]) + v
    
pattern_parts = [r'x?', r'b?', r'(?:s\[.+\])?', r'\d']

exclusive = re.compile(r'\*' + ''.join(pattern_parts[:-1]) + f'({pattern_parts[-1]})')
inclusive = re.compile(r'\*' + f'({"".join(pattern_parts)})')
within_bracket = re.compile(r'\[.+\]')

def group_parser(s, sanitize=False):
    attachment_idx = 1
    bond_symbs = ['', '=', '#']
    try:
        # mapped = re.split(r'\*x?(\d)', s)     
        mapped = exclusive.split(s)   
        for i in range(1, len(mapped), 2):
            bond_symbol = bond_symbs[int(mapped[i]) - 1] if mapped[i-1][-1:] not in bond_symbs[1:] else ""
            if i != 1 or mapped[0]:
                mapped[i] = f'{bond_symbol}[{(attachment_idx)}*:{mapped[i]}]' #wildcard atoms can have custom isotopes; give each attachment point a custom isotope so that they are unique and chirality is maintained
            else:
                mapped[i] = f'[{(attachment_idx)}*:{mapped[i]}]{bond_symbol}' #wildcard atoms can have custom isotopes; give each attachment point a custom isotope so that they are unique and chirality is maintained
            attachment_idx += 1
        mapped = Chem.MolFromSmiles(''.join(mapped), sanitize=False)
        Chem.SanitizeMol(mapped, sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY)# ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
        verify_group(mapped, group=True)
        return mapped
    except:
        raise ValueError(f'Issue in parsing group {s}')

def map_to_smiles(s, clear_valence=False):
    # convert smiles using atom mapping notation to smiles using custom notation (*N)
    return re.sub(r'[\-\=\#\@]?\[\d*\*:(\d)\]', r'*\1' if not clear_valence else r'*1', s)

def mol_to_group_s(m, smiles_override=None):
    # convert group molecule to custom notation
    if smiles_override is None:
        m_clone = Chem.Mol(m)
        Chem.KekulizeIfPossible(m_clone)
        for atom in m_clone.GetAtoms():
            if atom.HasProp('valAvailable'):
                atom.SetAtomMapNum(atom.GetIntProp('valAvailable'))
        smi = Chem.MolToSmiles(m_clone, kekuleSmiles=True, canonical=True).replace('~', '')
    else:
        smi = smiles_override
    res = map_to_smiles(smi)
    return res


def to_smarts(s): #assume s is verified already
    #therefore only possible connections are 
    #1. wildcard at front (e.g. *1C) <- NEED TO FLIP BOND IN THIS CASE
    #2. wildcard in middle (e.g. C(*1)C)
    #3. wildcard at end (e.g. C*1)
    L = inclusive.split(s)
    symbs = ['-', '=', '#']

    def process_delim(sep, reverse=False):
        matcher = '*,'#,#1'
        if 'x' in sep:
            sep = sep.replace('x', '')
            matcher += '!1#1'
        else:
            matcher += '#1'

        if 's' in sep:
            sep = sep.replace('s', '')
            matcher = within_bracket.match(sep).group(0)
            sep = sep.replace(matcher, '')
            
        if not reverse:
            return (','.join(symbs[:int(sep)]) + f'[{matcher}:{int(sep)}]')
        return f'[{matcher}:{int(sep)}]' + ','.join(symbs[:int(sep)]) 

    L[3::2] = map(process_delim, L[3::2])
    L[1] = process_delim(L[1], reverse=(L[0]==''))
    new_s = ''.join(L)

    ptable = Chem.GetPeriodicTable()
    elem_symbs = [ptable.GetElementSymbol(i) for i in range(118)]
    
    # this is a very very bad workaround for converting smiles to smarts, will eventually fix?
    # idea of workaround is to replace all instances of atoms that could be in aromatic rings with [#atomic num]
    for aromatic_atom in ['C', 'N', 'O', 'S', 'P']: 
        new_s = re.sub(f'[{aromatic_atom.upper()}{aromatic_atom.lower()}](?=[^{"".join([x[1] for x in elem_symbs if len(x) >= 2 and x[0] == aromatic_atom])}]|$)', f'[#{ptable.GetAtomicNumber(aromatic_atom)}]', new_s)
    
    # removes all nested brackets (i.e. [[A]] -> [A]), needed because SMARTS will not accept [[#6]@] for instance.
    # potentially problematic if recursive SMARTS are used
    new_s = bracket_remover(new_s)
    new_s = re.sub(r"-?([\\\/])-?", r"\1", new_s)
    new_s = new_s.replace('@H', '@')
    return new_s

def pattern_parser(s, raw_smarts=None):
    try:
        if not raw_smarts: 
            smarts = to_smarts(s) # converts custom notation into smarts 
        # can provide raw smarts string
        else:
            smarts = raw_smarts
        pat = Chem.MolFromSmarts(smarts)
        verify_group(pat)
        return pat
    except Exception as e:
        raise e
        # raise ValueError('Issue in parsing group.')

def extract_group_bonds(group, mol, matched): #extracts atoms and bonds for all instances of groups
    # scuffed workaround to ensure that groups that are not fully functionalized at all attachment points are matched
    # naive approach fails because SMARTS doesn't recognize "empty" atoms for wildcards; e.g. C* would not match C
    # to get around this, add hydrogens and then replace "*" with "[*,#1]" so that hydrogens are also recognized e.g. C[*,#1] would match CH
    
    mol_h = Chem.AddHs(mol) 
    Chem.Kekulize(mol_h)
    net_atoms = []
    net_bonds = []

    exact_match = set()

    for match in mol_h.GetSubstructMatches(group.pattern, uniquify=False, useChirality=True):
        if tuple(sorted(match)) in exact_match:
            continue
        parents_in_match = set([match[x] for x in group.parent_map.values()])
        match_set = set(match)
        net_atoms.append([])
        net_bonds.append([])
        for pattern_idx, idx in enumerate(match):
            a = mol_h.GetAtomWithIdx(idx)
            if a.GetAtomicNum() == 1 and not a.GetIsotope(): # ignore hydrogens
                continue   

            if pattern_idx in group.parent_map: # current atom is attachment point
                valid = False
                expected_parent = match[group.parent_map[pattern_idx]]
                for neigh in a.GetNeighbors():
                    n_idx = neigh.GetIdx()
                    if expected_parent == n_idx:
                        valid = True
                    elif n_idx in parents_in_match:
                        break
                else:
                    if valid:
                        net_bonds[-1].append(((idx, expected_parent), pattern_idx)) # bond mapping info: ((bond_idx1 <- outer, bond_idx2 <- outer), group_pattern_idx <- inner)
                        continue
                del net_atoms[-1]
                del net_bonds[-1]
                break
            else:
                if idx in matched:
                    del net_atoms[-1]
                    del net_bonds[-1]
                    break
                pat_at = group.mol.GetAtomWithIdx(pattern_idx)
                mol_at = mol.GetAtomWithIdx(idx)
                if (pat_at.GetFormalCharge() != mol_at.GetFormalCharge()) or \
                    (pat_at.GetNumExplicitHs() != mol_at.GetNumExplicitHs()) or \
                    (pat_at.GetIsotope() != mol_at.GetIsotope()) or \
                    (pat_at.GetNumRadicalElectrons() != mol_at.GetNumRadicalElectrons()):
                    del net_atoms[-1]
                    del net_bonds[-1]
                    break

                valid_neighbours = match_set # if atom is an attachment point parent, then any atom inside the match is valid
                if pattern_idx not in group.parent_map.values(): # if atom is not an attachment point parent, then only non attachment points are valid
                    valid_neighbours = valid_neighbours - set([match[x] for x in group.parent_map.keys()])
                for neigh in a.GetNeighbors():
                    if neigh.GetAtomicNum() == 1 and not neigh.GetIsotope():
                        continue
                    if neigh.GetIdx() not in valid_neighbours: # check if atom has invalid neighbour 
                        del net_atoms[-1]
                        del net_bonds[-1]
                        break
                else:
                    net_atoms[-1].append((idx, pattern_idx)) # add mapping if no issues
                    continue
                break
        else:
            exact_match.add(tuple(sorted(match)))

    return list(zip(net_atoms, net_bonds))


def bracket_remover(s): #removes nested brackets
    brackets = '[]'
    weights = defaultdict(int, {'[':1, ']':-1})
    new_s = ''
    val = 0
    for c in s:
        val += weights[c]
        # adds character if 1. its not a bracket 2. if current counter is <= 1 and if added, it maintains the current counter at <= 1 (i.e. last "]" or first "[")
        # assumes that brackets are balanced 
        new_s += c if (c not in brackets) or (max(val, val-weights[c]) <= 1) else '' 
    return new_s

from rdkit.Chem.rdchem import BondType as BT
bond_to_order = {BT.ZERO: 0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3}
order_to_bond = [BT.ZERO, BT.SINGLE, BT.DOUBLE, BT.TRIPLE]

def get_core(m, keep_dict=False): # remove all attachment points to get "core" of group
    m_edit = Chem.RWMol(m)
    Chem.Kekulize(m_edit, clearAromaticFlags=True)
    m_edit.BeginBatchEdit()
    if keep_dict:
        neighbor_dict = dict()
    for idx, atom in enumerate(m_edit.GetAtoms()):
        if not atom.GetAtomicNum():
            if keep_dict:
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetIdx() not in neighbor_dict:
                    neighbor_dict[neighbor.GetIdx()] = [neighbor]
                neighbor_dict[neighbor.GetIdx()].append(idx)
            m_edit.RemoveAtom(idx)
    m_edit.CommitBatchEdit()
    m_edit.UpdatePropertyCache()
    if keep_dict:
        neighbor_dict = defaultdict(list, [(v[0].GetIdx(), v[1:]) for v in neighbor_dict.values()])
        return m_edit.GetMol(), neighbor_dict
    return m_edit.GetMol()

from rdkit.Chem import rdMMPA
from rdkit.Chem.Fraggle import FraggleSim

class HashableMolecule:
    def __init__(self, mol):
        self.mol = mol
        self.hash = hash(hash_molecule(mol))

    def __hash__(self):
        return self.hash
    
    def __eq__(self, o):
        return self.hash == o.hash
    
def topology_from_rdkit(mol, only_atomic_num = False): #converts rdkit mol to nx graph for use in hashing. includes valAvailable prop in the atom properties.
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        anstr = f"{atom.GetAtomicNum()}"
        if not only_atomic_num:
            anstr += f"{atom.GetFormalCharge()}{atom.GetNumExplicitHs()}" + (f"{atom.GetProp('valAvailable')}" if atom.HasProp('valAvailable') else "")
        topology.add_node(atom.GetIdx(), an=anstr)
        for bond in atom.GetBonds():
            topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=str(bond.GetBondType()))
    return topology

def hash_molecule(mol1):
    return nx.weisfeiler_lehman_graph_hash(topology_from_rdkit(mol1, only_atomic_num=True), edge_attr='order', node_attr='an')

def raw_smarts_to_pattern(s):
    return re.sub(r'[\-\=\#\@]?\[(\d)#0\][\-\=\#\@]?', r'[*,H:1]', s)

def renumber_mol(mol, grammar, root_group : str): #group name
    extracted = grammar.extract_groups(mol)
    for group, atoms, _ in extracted:
        if group.name == root_group:
            m = Chem.MolFromSmiles(Chem.MolToSmiles(mol, rootedAtAtom=atoms[0][0]))
            Chem.Kekulize(m, clearAromaticFlags=True)
            return m
    return None #group not in mol


import copy
import itertools
from networkx.algorithms import isomorphism

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

def mol_with_atom_index(mol):
    mol_c = Chem.Mol(mol)
    for atom in mol_c.GetAtoms():
        # print(atom.GetIdx())
        atom.SetProp('_displayLabel', f'{atom.GetSymbol()}:{atom.GetIdx()}')
    return mol_c

def draw_atom_idx(Sm, size=(200, 200)):
    Sm = mol_with_atom_index(Sm)
    AllChem.Compute2DCoords(Sm)
    X = rdMolDraw2D.MolDraw2DCairo(*size)
    X.DrawMolecule(Sm)
    X.FinishDrawing()
    return Image.open(BytesIO(X.GetDrawingText()))