from rdkit import Chem
from group_selfies.utils.group_utils import extract_group_bonds
from group_selfies.group_mol_graph import Group, MolecularGraph
from group_selfies.group_grammar import GroupGrammar

grammar = GroupGrammar.essential_set() | GroupGrammar.from_file('../useful30.txt')
