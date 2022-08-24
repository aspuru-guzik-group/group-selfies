from group_selfies.utils.selfies_utils import split_selfies
from group_selfies.constants import INDEX_ALPHABET
 

def replace_indices(selfi):
    tokens = split_selfies(selfi)
    new_tokens = []
    for token in tokens:
        new_token = token
        if token[1] in ['+', '-']:
            try:
                index = int(token[2:-1])
                # print('INTENDED INDEX', index)
                new_token = INDEX_ALPHABET[index]
            except:
                pass
        new_tokens.append(new_token)
    return "".join(new_tokens)


test_gs = [
    ('benzene', 'C1=CC=CC=C1', True),
    ('styrene', 'C1(*1)=C(*1)C(*1)=C(*1)C(*1)=C1C=C', False),
    ('alcohol', '*1C(*1)(*1)-O', False),
    ('cyclohexane', 'C1CCCC(*2)C1', False),
    ('benzosulfanamide', 'NS(=O)(=O)C1=CC=CC=C1', True),
    ('trifluoromethane', '*1C(F)(F)F', False),
    ('toluene', 'C(*3)C1=CC=CC=C1', True),
    ('pyrazole', 'N1C=CC=N1', True),
    ('chiral', '*x1[C@H](*x1)(*x1)', False, 10),
    ('chiral2', '*x1[C@](*x1)(*x1)(*x1)', False, 10),
    ('sulfan', '*x1[S@@](=O)(=N)(*x1)', False, 10),
    ('sulfox', '*x1[S@@](=O)(*x1)', False, 10),
    ('sulfonium', '*x1[S@@+](*x1)(*x1)', False, 10),
    ('phosphine', '*x1[P@@H](=O)(*x1)', False, 10),
    # ('phosphine2', '*x1[P@@](=O)(*x1)(*x1)', False, 10),
    # ('phosphine3', '*x1[P@H](=O)(*x1)', False, 10),
    ('phosphine4', '*x1[P@H]*x1', False, 10),
    # ('phosphine5', '*x1[P@](*x1)(*x1)', False, 10),
    # ('phosphine6', '*x1[P@H](*x2)(*x1)', False, 10),
    # ('phosphine7', '*x1[P@](*x2)(*x1)*x1', False, 10),
    ('trans', '*x1\C(*1)=C(*1)\*x1', False, 5),
    ('cis', '*x1/C(*1)=C(*1)\*x1', False, 5),
    # ('imine', '*1/C(*1)=N\*1', False),
]
from group_selfies.group_mol_graph import Group, MolecularGraph
from group_selfies.group_grammar import GroupGrammar
vocab_dict = dict([(n, Group(n, s, all_attachment=all_at, priority=priority[0] if len(priority) > 0 else 0)) for n, s, all_at, *priority in test_gs])
grammar = GroupGrammar(vocab = vocab_dict)

from rdkit import Chem

def encoder(m, convert=False):
    global grammar
    try:
        if convert:
            m = Chem.MolFromSmiles(m)
        eg = grammar1.extract_groups(m)
        res = grammar1.encoder(m, eg)
        return res
    except:
        raise ValueError(f'ISSUE WITH MOL {Chem.MolToSmiles(m)}')

def decoder(m, convert=True):
    global grammar
    try:
        res = grammar.decoder(m)
        if convert:
            return Chem.MolToSmiles(res)
        return res
    except:
        raise ValueError(f'ISSUE DECOING {m}')