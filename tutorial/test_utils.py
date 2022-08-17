
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

def mol_with_atom_index(mol):
    mol_c = Chem.Mol(mol)
    for atom in mol_c.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_c

# S = 'CCC1CCCCC1'
# S = 'C1CC2CCCC3CCCC(C1)C23'
def draw_atom_idx(Sm, size=(200, 200)):
    Sm = mol_with_atom_index(Sm)
    AllChem.Compute2DCoords(Sm)
    X = rdMolDraw2D.MolDraw2DCairo(*size)
    X.DrawMolecule(Sm)
    X.FinishDrawing()
    return Image.open(BytesIO(X.GetDrawingText()))

def DrawMolsZoomed(mols, molsPerRow=10, subImgSize=(200, 200)):
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    for ii, mol in enumerate(mols):
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column*subImgSize[0], row * subImgSize[1] )
        sub = draw_atom_idx(mol, size=subImgSize)
        full_image.paste(sub, box=offset)
    return full_image
    
from group_selfies.utils.group_utils import map_to_smiles
def parse_amino_acids(a1):
    carboxylic = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H]')
    amino = Chem.MolFromSmarts('[Nh2,Nh3+]')
    new_a = Chem.ReplaceSubstructs(a1, amino, Chem.MolFromSmiles('N[*:1]'), useChirality=True)[0]
    new_a = Chem.ReplaceSubstructs(new_a, carboxylic, Chem.MolFromSmiles('C(=O)[*:1]'), replacementConnectionPoint=0, useChirality=True)[0]
    return map_to_smiles(Chem.MolToSmiles(new_a))


import matplotlib.pyplot as plt
import itertools
def plot_length_histogram(string_sets, alpha=0.5, boxes=10, names=None): #can be list of tokens, not necessarily strings
    if not names:
        names = list(map(str, range(len(string_sets))))
    assert len(names) == len(string_sets), 'not all sets are labelled'
    total_lens = list(itertools.chain.from_iterable([[(x if type(x) == int else len(x)) for x in string_set] for string_set in string_sets]))
    # print(total_lens)
    plt.figure(dpi=500, figsize=(10, 6))
    M_lens, m_lens = max(total_lens), min(total_lens)
    for name, string_set in zip(names, string_sets):
        lens = [(x if type(x) == int else len(x)) for x in string_set]

        box_size = max(1, (M_lens - m_lens + 1)//boxes)

        counts = [0 for _ in range(m_lens, M_lens+1, box_size)]
        for l in lens:
            counts[(l - m_lens)//box_size] += 1
        
        counts = [x*100/len(lens) for x in counts]

        plt.bar(list(range(m_lens, M_lens+1, box_size)), counts, width=box_size, alpha=alpha, label=name)
    plt.xlabel('Length of encoding (# of tokens)')
    plt.ylabel('Percentage of molecules')
    plt.legend()
    # plt.show()
    plt.savefig('res.png')

import re
def tokenize_selfies(s):
    return re.findall(r'\[.+?\]', s)    
import os, sys
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

from group_selfies.utils.group_utils import extract_group_bonds, bracket_remover, group_parser, pattern_parser, map_to_smiles
from group_selfies.utils.fragment_utils import fragment_mols
from group_selfies.group_mol_graph import Group, MolecularGraph
from group_selfies.group_grammar import GroupGrammar
from group_selfies.group_encoder import group_encoder
def test_on_dataset(smis, save_lens=False, fragments=None):
    res = []
    blockPrint()
    fragments = fragment_mols(smis, convert=True) if fragments is None else fragments
    enablePrint()
    group_dict = {}
    for idx, frag in enumerate(fragments):
        group_dict[f'frag{idx}'] = Group(f'frag{idx}', frag, all_attachment=False)
    grammar = GroupGrammar(vocab=group_dict)
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        extracted_g = grammar.extract_groups(m)
        s = group_encoder(grammar, m, extracted_g)
        if save_lens:
            s = len(tokenize_selfies(s))
        res.append(s)
    return res
