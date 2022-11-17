from group_selfies.bond_constraints import get_bonding_capacity
from typing import List, Optional, Union
import networkx as nx
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from group_selfies.utils.group_utils import group_parser, mol_to_group_s, pattern_parser, to_smarts, raw_smarts_to_pattern
from collections import defaultdict
from group_selfies.constants import BOND_DIR_REV
import re

class Atom:
    """An atom with associated specifications (e.g. element, isotope, h_count, charge). No chirality for now...
    """

    def __init__(
            self,
            element: str,
            isotope: Optional[int] = None,
            chirality: Optional[str] = None,
            h_count: Optional[int] = None,
            charge: int = 0,
            radical: int = 0,
    ):
        self.index = None
        self.element = element
        self.isotope = isotope
        self.chirality = chirality
        self.h_count = h_count
        self.charge = charge
        self.group_tag = None
        self.bond_count = 0
        self.pos = None
        self.radical = None
        if not self.charge:
            self.radical = radical

    def max_bonds(self):
        bond_cap = get_bonding_capacity(self.element, self.charge)
        bond_cap -= 0 if (self.h_count is None) else self.h_count
        # bond_cap -= self.radical
        return bond_cap
    
    def available_bonds(self):
        avail = self.max_bonds() - self.bond_count
        assert avail >= 0, "available_bonds is negative"
        return avail

    def __repr__(self):
        s = f"{self.index} {self.element}"
        if self.charge != 0:
            s += "{:+}".format(self.charge)
        s += f" {self.bond_count}/{self.max_bonds()} bonds filled"
        if self.group_tag is not None:
            s += f' group_tag={self.group_tag}'
        return s

class DirectedBond:

    def __init__(
            self,
            src: int,
            dst: int,
            order: Union[int, float],
            direc: str
    ):
        self.src = src
        self.dst = dst
        self.order = order
        self.direc = direc
        self.group_tag = None #not used, just temporary compat
        self.group_idx_dict = dict()

    def __repr__(self):
        s = f"{self.src} -> {self.dst} order={self.order}"
        if self.group_tag is not None:
            s += f' group_tag {self.group_tag}'
        s += f' group_idxs {list(self.group_idx_dict.keys())}'
        return s

class MolecularGraph:

    def __init__(self, mol=None, groups=[], smarts=False):
        """Precondition: groups are disjoint in mol."""
        self.atoms = list()  # stores atoms in this graph
        self.bond_dict = dict()  # stores all bonds in this graph
        self.diG = nx.DiGraph()  # directed graph
        self.bond_counts = list()  # stores number of bonds an atom has made
        self.groups = list()
        self.mol = mol
        self.visited = set()
        
        if mol is not None:
            if not smarts:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                mol.UpdatePropertyCache()
            for atom in mol.GetAtoms():
                if atom.GetNumExplicitHs() == 0:
                    if atom.GetImplicitValence() == 0:
                        h_count = 0
                    else:
                        h_count = None
                else:
                    h_count = atom.GetNumExplicitHs()

                isotope = None if not (res := atom.GetIsotope()) else res
                atom_obj = Atom(
                    element=atom.GetSymbol(),
                    h_count=h_count,
                    charge=atom.GetFormalCharge(),
                    isotope=isotope,
                    radical=atom.GetNumRadicalElectrons(),
                    )
                self.add_atom(atom_obj)
            
            bond_to_order = {BT.ZERO: 0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.QUADRUPLE: 4} # add zero order bond so attachment doesnt count to valency
            for bond in mol.GetBonds():
                src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                self.add_bond(src, dst, bond_to_order[bond.GetBondType()], BOND_DIR_REV[bond.GetBondDir()], update_bond_counts=(bond.GetBeginAtom().GetAtomicNum() and bond.GetEndAtom().GetAtomicNum()))
                
            if groups is not None:
                for group, member_idxs, bond_idxs in groups:
                    self.add_group(group, member_idxs, bond_idxs)

    def __len__(self):
        return len(self.diG)

    def __getitem__(self, key):
        return self.diG.nodes[key]['atom']
    
    def __repr__(self):
        s = 'ATOMS\n'
        for atom in self.get_atoms():
            s += repr(atom) + '\n'
        s += '\nBONDS\n'
        for bond in self.get_bonds():
            s += repr(bond) + '\n'
        
        s += '\nGROUPS\n'
        for group in self.groups:
            s += repr(group) + '\n'
        return s

    def get_atoms(self):
        return [self[idx] for idx in self.diG]
    
    def get_bond(self, src, dst) -> DirectedBond:
        return self.diG.edges[src, dst]['bond']

    def get_undirected_bond(self, u, v):
        if (u, v) in self.diG.edges:
            return self.diG.edges[(u, v)]['bond']
        elif (v, u) in self.diG.edges:
            return self.diG.edges[(v, u)]['bond']
        
    def get_bonds(self):
        return [self.get_bond(u, v) for u, v in self.diG.edges]
        # might be possible that get_bonds changes order depending on the group node labeling?

    def get_outbonds(self, src: int) -> List[DirectedBond]:
        return [self.get_bond(u, v) for u, v in self.diG.out_edges(src)]
        # might be possible that get_bonds changes order depending on node labeling?
    
    def get_inbonds(self, dst):
        return [self.get_bond(u, v) for u, v in self.diG.in_edges(dst)]

    def add_atom(self, atom: Atom) -> None:
        atom.index = len(self)
        self.diG.add_node(atom.index, atom=atom)

    def add_bond(
            self, src: int, dst: int,
            order: Union[int, float], direc : str, update_bond_counts=True
    ) -> None:

        bond = DirectedBond(src, dst, order, direc)
        self.diG.add_edge(src, dst, bond=bond)
        if update_bond_counts:
            self[src].bond_count += order
            self[dst].bond_count += order
            
    def update_bond_order(
            self, bond,
            new_order: Union[int, float]
    ) -> None:
        assert 1 <= new_order <= 3

        old_order = bond.order
        bond.order = new_order
        self[bond.src].bond_count += (new_order - old_order)
        self[bond.dst].bond_count += (new_order - old_order)
    
    #EDITED TO WORK WITH POINTED GROUPS

    def add_group(self, group, member_idxs, bond_idxs):
        """Precondition: none of the atoms in member_idxs already belong to a group."""
        group_idx = len(self.groups)
        # for inner, outer in enumerate(member_idxs):
        for outer, inner in member_idxs:
            assert self[outer].group_tag is None, f'overlapping groups: atom {outer} is already in a group'
            self[outer].group_tag = (group_idx, inner)
                
        for (outer_dst, outer_src), attach_idx in bond_idxs:
            for i in [1, -1]:
                if tuple([outer_src, outer_dst][::i]) in self.diG.edges:
                    bond_obj = self.get_bond(*[outer_src, outer_dst][::i])
                    bond_obj.group_idx_dict[group_idx] = group.attachment_points.index(attach_idx)
        group.matched_to(member_idxs)
        group.index = group_idx
        self.groups.append(group)

    def get_group(self, atom_idx):
        """Gets the group that atom_idx belongs to. Returns None if atom_idx does not belong to a group."""
        for group in self.groups:
            if atom_idx in group.member_idxs:
                return group

    def get_inner_idx(self, atom_idx):
        """Gets the atom's inner idx if it belongs to a group, returns None otherwise"""
        group_obj = self.get_group(atom_idx)
        if group_obj is not None:
            return group_obj.outer_to_inner[atom_idx]
        
    def get_group_atoms(self, group_idx):
        atoms = []
        for outer in self.groups[group_idx].member_idxs:
            atoms.append(self[outer])
        return atoms
    
    def get_group_outbonds(self, group_idx):
        # """Return all the bonds of this group's atoms that are not group bonds."""
        """Return all the used attachment bonds for this group."""
        group_atoms = self.get_group_atoms(group_idx)
        outbonds = []
        for atom in group_atoms:
            for bond in self.get_outbonds(atom.index):
                if group_idx in bond.group_idx_dict:
                    outbonds.append(bond)
        return outbonds
    
    #possibly unnecessary?
    def get_group_inbonds(self, group_idx):
        """Return all the bonds into this group's atoms."""
        group_atoms = self.get_group_atoms(group_idx)
        inbonds = []
        for atom in group_atoms:
            for bond in self.get_inbonds(atom.index):
                if group_idx in bond.group_idx_dict:
                    inbonds.append(bond)
        return inbonds
    
    def find_next_available(self, group, current_idx, rel_idx):

        possible_idx = (current_idx + rel_idx) % len(group)

        circular_range = list(range(possible_idx, len(group))) + list(range(possible_idx))
        for inner in circular_range:
            # get the group atom
            # if available bonds, return
            if self[group.member_idxs[inner]].available_bonds() > 0:
                return inner
        
        assert False, "No available bonds left"
    
    def num_available_atoms(self, group):
        num = 0
        for outer in group.member_idxs:
            if self[outer].available_bonds() > 0:
                num += 1
        return num
    
    def total_available_bonds_left(self, group_idx):
        group = self.groups[group_idx]
        tot = 0
        for outer in group.member_idxs:
            tot += self[outer].available_bonds()
        return tot

    def visit(self, atom_idx):
        self[atom_idx].pos = len(self.visited)
        self.visited.add(atom_idx)

    def get_atom_pos(self, atom_idx):
        return self[atom_idx].pos
        
order_to_bond = [BT.ZERO, BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.QUADRUPLE]
class Group:
    def __init__(self, name, canonsmiles, overload_index=None, all_attachment=False, smarts_override=None, priority=0, sanitize=False): 
        assert '-' not in name, "Group names cannot contain dashes"
        assert re.match(r'[0-9]+', name) is None, "Group names cannot start with numbers"

        self.matched = False
        self.index = None
        self.name = name
        self.canonsmiles = canonsmiles 
        self.priority = priority
        if smarts_override:
            self.pattern = pattern_parser('', raw_smarts=smarts_override)
        else:
            if all_attachment:
                mol = group_parser(canonsmiles)
                Chem.Kekulize(mol)
                temp_mol = AllChem.EditableMol(mol)
                self.attachment_points = []
                # if in all attachment mode, treat (*N) as removing N from the attached atoms valence 
                non_attachment = {}
                parent_map = defaultdict(int)
                counter = 0
                to_delete = []
                for idx, atom in enumerate(mol.GetAtoms()):
                    if atom.GetAtomicNum():
                        non_attachment[idx] = counter
                        counter += 1
                    else:
                        parent = atom.GetNeighbors()[0]
                        parent_map[parent.GetIdx()] += atom.GetIntProp('valAvailable')
                        to_delete.append(idx)
                parent_map = defaultdict(int, [(non_attachment[k], v) for k, v in parent_map.items()])
                for idx in sorted(to_delete, reverse=True):
                    temp_mol.RemoveAtom(idx)
                
                temp_graph = MolecularGraph(temp_mol.GetMol(), None)
                attachment_idx = 1
                for idx in range(counter):
                    # print('CUR CHECK IDX', idx)
                    if temp_graph[idx].available_bonds():
                        z = min((temp_graph[idx].available_bonds() -  parent_map[idx]), len(order_to_bond)-1)
                        if z > 0:
                            new_at = AllChem.Atom('*')
                            new_at.SetAtomMapNum(z)
                            new_at.SetIsotope(attachment_idx)
                            attachment_idx += 1
                            new_idx = temp_mol.AddAtom(new_at)
                            # temp_mol.AddBond(idx, new_idx, AllChem.BondType.SINGLE)
                            temp_mol.AddBond(idx, new_idx, order_to_bond[z]) 
                            self.attachment_points.append(new_idx)

                self.mol = temp_mol.GetMol()
                Chem.SetAromaticity(self.mol)
                canon_smi = Chem.MolToSmiles(self.mol, canonical=True, kekuleSmiles=True, allBondsExplicit=True)
                try:
                    self.mol = Chem.MolFromSmiles(canon_smi)
                    Chem.Kekulize(self.mol)
                except:
                    raise ValueError(f'ISSUE WITH {canon_smi}')
                for atom in self.mol.GetAtoms():
                    if not atom.GetAtomicNum():
                        z = atom.GetAtomMapNum()
                        atom.SetIntProp('valAvailable', z)
                        atom.SetProp('_displayLabel', f'*{z}')
                        atom.SetAtomMapNum(0)
                self.canonsmiles = mol_to_group_s(None, smiles_override=canon_smi) 
            else:
                self.mol = group_parser(self.canonsmiles, sanitize=sanitize)
                if overload_index is None:
                    self.overload_index = 0
                else:
                    self.overload_index = overload_index
            self.pattern = pattern_parser(self.canonsmiles)
        
        Chem.Kekulize(self.mol, clearAromaticFlags=True)
        self.mol.UpdatePropertyCache()
        # attachment points relative to SMARTS pattern rather than mol
        self.attachment_points = [i for i, a in enumerate(self.pattern.GetAtoms()) if a.HasProp('valAvailable')]
        self.parent_map = {}
        for idx in self.attachment_points:
            atom = self.pattern.GetAtomWithIdx(idx)
            self.parent_map[idx] = atom.GetNeighbors()[0].GetIdx()
        self.group_size = self.pattern.GetNumAtoms() - len(self.attachment_points)
        
        # force pattern and group indices to be in the same order
        mol_clone = Chem.Mol(self.mol)
        Chem.Kekulize(mol_clone)
        mapping1 = mol_clone.GetSubstructMatch(self.pattern)
        if mapping1:
            self.pattern = Chem.RenumberAtoms(self.pattern, mapping1)
        self.graph = MolecularGraph(self.mol, None)

    def matched_to(self, member_idxs):
        self.member_idxs = list(zip(*member_idxs))[0]
        self.outer_to_inner = dict(member_idxs)
        self.matched = True

    def __repr__(self):
        s = f'{self.name} {self.canonsmiles}'
        if self.matched:
            s += f' {self.index} {self.member_idxs}'
        return f'<Group {s}>'
    
    def __len__(self):
        return len(self.attachment_points)
    
    def find_next_available(self, current_idx, rel_idx):
        current = (current_idx + rel_idx) % len(self.attachment_points)
        for shift in range(len(self.attachment_points)):
            inner = (current + shift) % len(self.attachment_points)
            if self.mol.GetAtomWithIdx(self.attachment_points[inner]).HasProp('valAvailable'):
                return inner
        return None

    def attachment_valency(self, global_idx):
        attach = self.mol.GetAtomWithIdx(self.attachment_points[global_idx])
        assert attach.HasProp('valAvailable'), 'Not an attachment point'
        return attach.GetIntProp('valAvailable')
    
    def mol_without_attachment_points(self):
        to_remove = []
        new_mol = Chem.RWMol(self.mol)
        for idx, atom in enumerate(new_mol.GetAtoms()):
            if atom.GetSymbol() == '*':
                to_remove.append(idx)
        for idx in reversed(to_remove):
            new_mol.RemoveAtom(idx)
        return new_mol

class GroupWrapper:
    def __init__(self, group):
        # self.group = group
        for k, v in group.__dict__.items():
            setattr(self, k, v)
        self.member_idxs = []
        self.outer_to_inner = None
        self.matched = False
        self.group = group

    def matched_to(self, member_idxs):
        self.member_idxs = list(zip(*member_idxs))[0]
        self.outer_to_inner = dict(member_idxs)
        self.matched = True

    def __repr__(self):
        return self.group.__repr__()
    
    def __len__(self):
        return len(self.group)
    
    def find_next_available(self, current_idx, rel_idx):
        return self.group.find_next_available(current_idx, rel_idx)

    def attachment_valency(self, global_idx):
        return self.group.attachment_valency(global_idx)