from rdkit import Chem
from group_selfies.group_mol_graph import MolecularGraph, Group, GroupWrapper
from group_selfies.constants import INDEX_ALPHABET, POP_SYMBOL, INDEX_CODE
import numpy as np
from itertools import chain
import itertools

from group_selfies.utils.group_utils import extract_group_bonds

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import group_selfies
essential = pkg_resources.open_text(group_selfies, 'essential_set.txt')
cached_essential = None
rng = np.random.default_rng()

def name_to_token(name, n_points):
    return [f'[:{"".join(map(str, bond_type))}{name}]' for bond_type in itertools.product(['', '=', '#'], range(n_points))]

def token_to_name(token):
    return token[2:-1]

def _compatible(match, mapped_atoms, group_mol, mol):
    """Helper method for extract_groups()"""

    match_dict = dict(match)

    for outer, inner in match:
        if outer in mapped_atoms:
            return False

    for outer in match_dict.keys():
        for x in mol.GetAtomWithIdx(outer).GetNeighbors():
            x = x.GetIdx()
            if x not in match_dict:
                continue
            bond1 = mol.GetBondBetweenAtoms(outer, x)
            bond2 = group_mol.GetBondBetweenAtoms(match_dict[outer], match_dict[x])
            if bond2 is None:
                return False
            if bond1.GetBondDir() != bond2.GetBondDir():
                return False
    return True

class GroupGrammar:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = dict()
        elif type(vocab) == list:
            self.vocab = dict()
            for group in vocab:
                self.vocab[group.name] = group
        else:
            self.vocab = vocab

    @classmethod
    def from_list(cls, l):
        vocab_dict = dict([(n, Group(n, s, all_attachment=all_at, priority=priority[0] if len(priority) else 0)) for n, s, all_at, *priority in l])
        return cls(vocab=vocab_dict)

    @classmethod
    def from_file(cls, filename):
        test_gs = [l.strip().split(' ') for l in open(filename)]
        vocab_dict = dict([(n, Group(n, s, priority=int(priority[0]) if len(priority) else 0)) for n, s, *priority in test_gs])
        return cls(vocab=vocab_dict)

    @classmethod
    def essential_set(cls):
        global cached_essential
        if cached_essential:
            return cached_essential
        
        test_gs = [l.strip().split(' ') for l in essential]
        vocab_dict = dict([(n, Group(n, s, priority=int(priority[0]) if len(priority) else 0)) for n, s, *priority in test_gs])
        cached_essential = cls(vocab=vocab_dict)
        return cached_essential

    def to_file(self, filename):
        open(filename, 'w').write('\n'.join([' '.join([n, group.canonsmiles, str(group.priority)]) for n, group in self.vocab.items()]))

    def names_sorted_by_size(self):
        """Returns the names of all groups, sorted by num_atoms, from highest to lowest"""
        names = list(self.vocab.keys())
        group_sizes = []
        for name in names:
            group_sizes.append(self.vocab[name].group_size)

        return sorted(names, reverse=True, key=lambda x: (self.vocab[x].priority, self.vocab[x].group_size))
    
    def extract_groups(self, mol):
        """Finds groups in <self.group_vocabulary> that are in <mol> in a greedy way. It tries to 
        replace the largest groups first before moving onto smaller groups.

        Returns a list of (group_object, atom_idxs, bond idx tuples)"""
        mapped_atoms = set()
        groups = []
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol.UpdatePropertyCache()
        for name in self.names_sorted_by_size():
            group = self.get_group(name)
            for matched_atoms, matched_bonds in extract_group_bonds(group, mol, mapped_atoms):
                match, _ = zip(*matched_atoms)
                if _compatible(matched_atoms, mapped_atoms, group.mol, mol):
                    groups.append((self.get_group(name), matched_atoms, matched_bonds))
                    mapped_atoms.update(match)
        return groups

    def all_tokens(self):
        return list(chain.from_iterable([name_to_token(name, len(group.attachment_points)) for name, group in self.vocab.items()]))

    def add_group(self, name, canonsmiles, overload_index=None, all_attachment=False, smarts_override=None, priority=0, sanitize=False):

        if name in self.vocab:
            print(f'{name} is already in the group dictionary, skipping')
            return False

        group = Group(name, canonsmiles, overload_index, all_attachment, smarts_override, priority, sanitize)
        self.vocab[name] = group
        return True

    def delete_group(self, name):
        """Deletes the group with name `name`
        
        Precondition: `name` is in this grammar.
        """
        self.vocab.pop(name, None)


    def get_group(self, name):
        """Obtains a fresh copy of a Group object given its name."""
        return GroupWrapper(self.vocab[name])

    def encoder(self, mol, groups, **args):
        from group_selfies.group_encoder import group_encoder
        return group_encoder(self, mol, groups, **args)

    def decoder(self, selfies, **args):
        from group_selfies.group_decoder import group_decoder
        return group_decoder(self, selfies, **args)

    def full_encoder(self, mol, **args):
        from group_selfies.group_encoder import group_encoder
        groups = self.extract_groups(mol)
        return group_encoder(self, mol, groups, **args)
    
    def pretty_encode(self, gselfi, **args):
        return self.full_encoder(Chem.MolFromSmiles(Chem.MolToSmiles(self.decoder(gselfi))), **args)

    def read_index_from_group_selfies(self, symbol_iter):
        try:
            symbol = next(symbol_iter)
            if symbol == POP_SYMBOL:
                return None
            if symbol in INDEX_CODE:
                return INDEX_CODE[symbol]
            else:
                name = token_to_name(symbol)
                if name in self.vocab:
                    return self.vocab[name].overload_index
                else:
                    return 0
        except StopIteration:
            return None
    
    def get_index_from_selfies(self, symbol):
        if symbol in INDEX_CODE:
            return INDEX_CODE[symbol]
        else:
            name = token_to_name(symbol)
            if name in self.vocab:
                return self.vocab[name].overload_index
            else:
                return 0

    def merge(self, other):
        return GroupGrammar(vocab={**self.vocab, **other.vocab})

    def __or__(self, other):
        return self.merge(other)


def common_r_group_replacements_grammar(num_groups=None, rng=rng):
    """Creates a grammar with ~377 common R groups from GlobalChem.
    
    If num_groups is not None, then selects a random subset of that size.
    You can pass your own numpy rng to set the random seed.
    
    """
    from global_chem import GlobalChem
    gc = GlobalChem()
    gc.build_global_chem_network(print_output=False, debugger=False)
    # node = gc.get_all_nodes()[10] #get_node('common_r_group_replacements')['node_value']
    smiles_dict = gc.get_all_nodes()[10].get_smiles()
    named_groups = []

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if num_groups is None:
        for name, smi in smiles_dict.items():
            mol = Chem.MolFromSmiles(smi)
            name = name.replace('[', '(').replace(']', ')').replace('-','')
            if name[0].isdigit():
                name = ';' + name

            if mol is not None and mol.GetNumAtoms() > 1:
                named_groups.append([name, smi])
    else:
        count = 0
        items = list(smiles_dict.items())
        rng.shuffle(items)
        for name, smi in items:
            mol = Chem.MolFromSmiles(smi)
            name = name.replace('[', '(').replace(']', ')').replace('-','')
            if name[0].isdigit():
                name = ';' + name
            if mol is not None and mol.GetNumAtoms() > 1:
                named_groups.append([name, smi])
                count += 1
                if count == num_groups:
                    break
    RDLogger.EnableLog('rdApp.*')

    grammar = GroupGrammar()
    for name, smi in named_groups:
        try:
            grammar.add_group(name, smi, all_attachment=True)
        except:
            print("couldn't add group, skipping", name, smi)

    return grammar
