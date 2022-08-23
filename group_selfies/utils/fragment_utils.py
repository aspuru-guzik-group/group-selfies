from group_selfies.utils.group_utils import hash_molecule, HashableMolecule, mol_to_group_s, bond_to_order, get_core
from group_selfies.bond_constraints import get_bonding_capacity
from collections import defaultdict
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT

from rdkit.Chem import rdMMPA
from rdkit.Chem.Fraggle import FraggleSim
import math

aliases = {'default': ['default'],
            'mmpa': ['mmpa'],
            'fraggle': ['fraggle'],}
alias_translation = dict([(alias, k) for k, v in aliases.items() for alias in v])

def fragment_mol(m, MIN_SIZE=4, method='default'): # fragment a molecule based on various strategies
    if alias_translation[method.lower()] == 'default':
        m_clone = Chem.Mol(m)
        Chem.Kekulize(m_clone, clearAromaticFlags=True)
        # fragment at outgoing ring bonds
        fragment_smarts = Chem.MolFromSmarts('[Rx2;D3,D4][$([Rx2;D3,D4]),$([!R])]') 
        bs = [m.GetBondBetweenAtoms(a1, a2).GetIdx() for a1, a2 in m_clone.GetSubstructMatches(fragment_smarts)]
        if not bs:
            return []
        fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(m_clone, bs), asMols=True, sanitizeFrags=True)
    elif alias_translation[method.lower()] == 'mmpa': # use MMPA fragmentation
        fragments = [tup[0] for tup in rdMMPA.FragmentMol(m) if tup[0] is not None]
    elif alias_translation[method.lower()] == 'fraggle':  
        fragments = [res for mult in FraggleSim.generate_fraggle_fragmentation(m) for x in mult.split('.') if (res := Chem.MolFromSmiles(x)) is not None] 
    
    new_fragments = []
    unique_frag_hashes = set()
    for fragment in fragments:
        temp_fragment = Chem.EditableMol(fragment)
        atom_count = 0
        for idx, atom in enumerate(fragment.GetAtoms()):
            if atom.GetAtomicNum():
                atom_count += 1
        if atom_count < MIN_SIZE:
            continue
        new_f = fragment
        for idx, atom in enumerate(new_f.GetAtoms()):
            if not atom.GetAtomicNum():
                parent = atom.GetNeighbors()[0].GetIdx()
                valence = bond_to_order[fragment.GetBondBetweenAtoms(parent, idx).GetBondType()]
                atom.SetIntProp('valAvailable', valence)
                atom.SetIsotope(0)

        hm = hash_molecule(new_f) # hash fragment to avoid repeats
        if hm not in unique_frag_hashes: 
            try:
                Chem.Kekulize(new_f)
                new_f_core = get_core(new_f) # extract core from fragment
                new_fragments.append((new_f, new_f_core))
                unique_frag_hashes.add(hm)
            except Exception as e:
                print('ISSUE', e)
                pass
    return new_fragments

def get_attachment_points(atom):
    return [neigh.GetIntProp('valAvailable') for neigh in atom.GetNeighbors() if neigh.HasProp('valAvailable')]

def merge_list(L1, L2, limit=math.inf): 
    # merge two lists optimally? based on the following rules:
    # sum of values in merged list <= maximum sum of the initial lists
    # can merge v1 in L1 and v2 in L2 as max(v1, v2)
    L1, L2 = sorted([L1, L2], key=lambda x: len(x))
    L1.sort(reverse=True) # sort lists descending, assume that optimal merging is "greedy"
    L2.sort(reverse=True)
    if len(L1) != len(L2):
        merged = L2[len(L1):]
        if (valid := merge_list(L1[::], L2[:len(L1)])) is not None:
            merged = valid + merged
        else:
            return None        
    else:
        merged = [max(l1, l2) for l1, l2 in zip(L1, L2)]
    
    if sum(merged) <= min(max(sum(L1), sum(L2)), limit):
        return merged
    return None


def capacity(atom):
    if atom.GetSymbol() == '*':
        return 0
    explicit_valence = atom.GetExplicitValence()
    for neigh in atom.GetNeighbors():
        if neigh.GetSymbol() == '*':
            explicit_valence -= atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neigh.GetIdx()).GetValenceContrib(atom)
    return get_bonding_capacity(atom.GetSymbol(), atom.GetFormalCharge()) - explicit_valence
       

### NOTE: merging destroys ordering and might not give expected chirality!
def merge_patterns(p1, p2, core): #return None if not mergeable, else return merged
    canon_mapping = p1.GetSubstructMatch(core) # assign an arbitrary mapping for pattern 1
    best_mappings = []
    for match_mapping in p2.GetSubstructMatches(core, uniquify=False): 
        # loop through all mappings for pattern 2
        # will get all possible mappings between patterns
        intersections = 0
        modified_pattern = Chem.EditableMol(core)
        modified_pattern.BeginBatchEdit()
        for core_idx, (c_idx, m_idx) in enumerate(zip(canon_mapping, match_mapping)):
            # get attachment points at these indices
            c_attach = get_attachment_points(a1 := p1.GetAtomWithIdx(c_idx)) 
            m_attach = get_attachment_points(a2 := p2.GetAtomWithIdx(m_idx))
            
            # merge (if possible)
            if (res := merge_list(c_attach, m_attach, limit=min(capacity(a1), capacity(a2)))) is not None:
                intersections += min(len(c_attach), len(m_attach))
                
                # make edits to core to create merged pattern
                for attachment in res:
                    new_at = AllChem.Atom('*')
                    new_at.SetIntProp('valAvailable', attachment)
                    new_idx = modified_pattern.AddAtom(new_at)
                    modified_pattern.AddBond(core_idx, new_idx, AllChem.BondType.SINGLE)
            else:
                # print('bad mapping')
                break
        else:
            modified_pattern.CommitBatchEdit()
            best_mappings.append((intersections, modified_pattern.GetMol()))
    
    if not best_mappings: #if no valid mapping, then not mergeable
        return None
    return sorted(best_mappings, reverse=True, key=lambda x: x[0])[0][1] #use mapping with most intersections

def closely_contained(mol, substruct, k=2):
    match = set(mol.GetSubstructMatch(substruct))
    if not match or not (0 < mol.GetNumAtoms() - len(match) <= k):
        return False
    for x in range(mol.GetNumAtoms()):
        if x in match:
            continue
        if len(mol.GetAtomWithIdx(x).GetNeighbors()) != 1:
            return False
        elif mol.GetAtomWithIdx(x).GetBonds()[0].GetBondType() != AllChem.BondType.SINGLE:
            return False
    return True

def force_core(mol, core):
    mapping = mol.GetSubstructMatch(core)
    new_m = Chem.RWMol(mol)
    to_remove = []
    for idx, atom in enumerate(new_m.GetAtoms()):
        if idx not in mapping:
            if any(in_mapping := [neigh.GetIdx() in mapping for neigh in atom.GetNeighbors()]):
                neigh = list(atom.GetNeighbors())[in_mapping.index(True)]
                new_at = Chem.Atom('*')
                bond = new_m.GetBondBetweenAtoms(idx, neigh.GetIdx())
                if not neigh.GetAtomicNum():
                    new_at.SetIntProp('valAvailable', neigh.GetIntProp('valAvailable'))
                else:
                    new_at.SetIntProp('valAvailable', bond_to_order[bond.GetBondType()])
                bond.SetBondType(BT.SINGLE)
                new_m.ReplaceAtom(idx, new_at)
            else:
                to_remove.append(idx)
    for idx in sorted(to_remove, reverse=True):
        new_m.RemoveAtom(idx)
    return new_m.GetMol()

# fragment set of molecules into groups, and try and merge when possible
# basic algorithm:
# store a dictionary keyed using the cores of the groups
# each value is a list which contains the current groups for the core
# when fragment is added:
# 1. check if core has a similar submolecule in dict. if not, add to dict
# 2. otherwise, force group to use smaller core, then try and merge with current groups in the core
# 3. if not mergeable, append to list

def fragment_mols(m_set, convert=False, n_limit=10, target=100, fragmented=False, method='default'):
    
    core_dict = defaultdict(list)
    core_counter = defaultdict(int)
    net_fragments = []

    def merge_into_core(new_h, new_f, new_f_core):
        for idx, existing_pattern in enumerate(core_dict[new_h]):
            if (res := merge_patterns(existing_pattern, new_f, new_f_core)) is not None:
                if '*' in Chem.MolToSmiles(res):
                    core_dict[new_h][idx] = res
                    core_counter[new_h] += 1
                break
        else:
            if '*' in Chem.MolToSmiles(new_f):
                core_dict[new_h].append(new_f)
                core_counter[new_h] += 1

    for m in tqdm(m_set):
        if convert:
            m = Chem.MolFromSmiles(m)
        for new_f, new_f_core in (fragment_mol(m, method=method) if not fragmented else [(m, get_core(m))]):
            new_h = HashableMolecule(new_f_core)
            merge_into_core(new_h, new_f, new_f_core)

    fragments = []
    
    filtered_cores = [k for k in core_dict.keys() if core_counter[k] >= n_limit]
    current_cores = []

    for core in sorted(filtered_cores, key=lambda x: x.mol.GetNumAtoms()):
        for existing_core in current_cores:
            if closely_contained(core.mol, existing_core.mol):
                for group in core_dict[core]:
                    merge_into_core(existing_core, force_core(group, existing_core.mol), existing_core.mol)
                break
        else:
            current_cores.append(core)
    
    cores, weights = zip(*[(k, len(core_dict[k])) for k in current_cores])
    diverse_cores = select_diverse_set(cores, target, weights=weights)
    for k in diverse_cores:
        for x in core_dict[k]:
            try:
                res = mol_to_group_s(x)
                if '*' in res:
                    fragments.append(res)
            except Exception as e:
                pass
    return fragments


def pair_to_triangular_idx(i, j, l_set):
    i, j = sorted([i, j])
    if i == j:
        return -1
    return (l_set)*(l_set-1)//2 - (l_set-i)*(l_set - i - 1)//2 + j - i - 1
    
from rdkit.DataManip.Metric import rdMetricMatrixCalc

def select_diverse_set(l, k, weights=None):
    if weights is None:
        weights = [1]*len(l)
    temp_set = [(mol, i) for i, mol in enumerate(l)]
    mf = [Chem.RDKFingerprint(m.mol) for m in l]
    dist_matrix = rdMetricMatrixCalc.GetTanimotoDistMat(mf) + [0]
    result = [temp_set.pop(0)]
    cur_weight = 0
    while cur_weight < k and len(temp_set) > 0:
        result.append(chosen := temp_set.pop(max(range(len(temp_set)), key=lambda x: sum([dist_matrix[pair_to_triangular_idx(idx, temp_set[x][1], len(mf))]*weights[idx]*weights[temp_set[x][1]] for _, idx in result]))))
        cur_weight += weights[chosen[1]]
    return list(zip(*result))[0]