from group_selfies.group_mol_graph import MolecularGraph
from group_selfies.grammar_rules import get_selfies_from_index
from rdkit import Chem
from group_selfies.constants import ORGANIC_SUBSET, INDEX_ALPHABET
order2char = {1: '', 2: '=', 3: '#'}

def derive_ring_bond(G, bond, map_idx):
    tokens = []

    # get indices relative to map_idx
    # if atom is in a group, use map_idx of the group + index of the attachment point
    src_idx = map_idx[bond.src] if bond.src in map_idx else (map_idx[f'group{(g_idx := G[bond.src].group_tag[0])}'] + bond.group_idx_dict[g_idx])
    dst_idx = map_idx[bond.dst] if bond.dst in map_idx else (map_idx[f'group{(g_idx := G[bond.dst].group_tag[0])}'] + bond.group_idx_dict[g_idx])

    ring_length = src_idx - dst_idx
    negative_ring = False
    if ring_length < 0:
        # this can occur if making a ring bond out of a group to a branch off the same group
        ring_length = -ring_length
        negative_ring = True
    length_tokens = get_selfies_from_index(ring_length-1)
    ring_token = f"[{bond.direc}{order2char[bond.order]}Ring{len(length_tokens)}]"
    tokens.append(ring_token)
    if negative_ring:
        tokens.append('[->]')
    tokens.extend(length_tokens)
    return tokens

class MutableCounter:
    def __init__(self, i):
        self.i = i

def graph_to_selfies(grammar, G, previous_bond, root, counter=None, map_idx=None): #assume extracted groups are already pointed
    derived = []
    curr = root
    # index map; assigns an index to non-group atoms or to groups.
    # either atom_idx -> idx or "group{group_idx}" (in string to avoid overlap with atom_idx) -> idx
    map_idx = map_idx or {} 
    counter = counter or MutableCounter(0) # use mutable class instead of int so that counter increments in recursion without additional returns
    while True:
        if G[curr].group_tag is None:
            # === not a group ===
            derived.append(atom_to_selfies(previous_bond, G[curr]))
            G.visit(curr)
            
            map_idx[curr] = counter.i
            counter.i += 1

            out_bonds = G.get_outbonds(curr)

            if len(out_bonds) == 0:
                return derived

            for i, bond in enumerate(out_bonds):
                if bond.dst in G.visited:
                    derived.extend(derive_ring_bond(G, bond, map_idx))
                    # if this is the last bond and a ring bond
                    if i == len(out_bonds) - 1:
                        return derived

                elif i == len(out_bonds) - 1:
                    # last outbond and not a ring bond, chain continues
                    previous_bond = bond
                    curr = bond.dst
                else:  # branch
                    branch = graph_to_selfies(grammar, G, bond, bond.dst, counter=counter, map_idx=map_idx)
                    branch_token = f"[{order2char[bond.order]}Branch]"
                    derived.append(branch_token)
                    derived.extend(branch)
                    derived.append('[pop]')

        else:
            # === is a group ===
            # find previous atom and bond
            # group_obj = G.get_group(curr)
            group_obj = G.get_group(curr)
            name = group_obj.name 
            for outer_idx in group_obj.member_idxs:
                G.visit(outer_idx)
            
            map_idx[f'group{group_obj.index}'] = counter.i
            # after adding group, increment by number of attachment points
            # equivalent to mapping each attachment point to an index
            counter.i += len(group_obj.attachment_points)

            # take care of the previous bond into this group node
            inner_idx = group_obj.find_next_available(previous_bond.group_idx_dict[group_obj.index], 0) if previous_bond is not None else 0

            if inner_idx is None:
                inner_idx = 0

            if previous_bond is None:
                order = 1
                direc = ''
            else:
                order = previous_bond.order
                direc = previous_bond.direc

            ### starter tokens for specifying group (e.g. starting atom idx, starting bond order, # of branches)
            ### specify group, idx, bond as single "pointed group" token
            derived.append(f"[:{direc}{order2char[order]}{inner_idx}{name}]") 
            # take care of branches
            branches = G.get_group_outbonds(group_obj.index)

            if len(branches) == 0:
                # no branches, this is the end
                return derived
            #############
            prev_inner_idx = inner_idx
            for bond in branches:
                this_inner_idx = bond.group_idx_dict[group_obj.index] #index relative to attachment points array
                rel_idx = (this_inner_idx - prev_inner_idx) % len(group_obj)
                prev_inner_idx = this_inner_idx

                derived.extend([get_selfies_from_index(len(INDEX_ALPHABET) - 1)[0], '[pop]']*(rel_idx//(len(INDEX_ALPHABET) - 1)))
                derived.append(get_selfies_from_index(rel_idx % (len(INDEX_ALPHABET) - 1))[0]) #parse into chunks
                
                if bond.dst in G.visited:
                    # making ring bond branch
                    ring_bond_branch = derive_ring_bond(G, bond, map_idx)
                    derived.extend(ring_bond_branch)
                else:
                    # making regular branch
                    branch = graph_to_selfies(grammar, G, bond, bond.dst, counter=counter, map_idx=map_idx)
                    derived.extend(branch)
                    
                derived.append('[pop]')
            return derived

def atom_to_selfies(bond, atom):
    bond_char = "" if (bond is None) else (bond.direc + order2char[bond.order])
    # specs = (atom.isotope, atom.chirality, atom.h_count, atom.charge)
    builder = []
    builder.append("[")
    builder.append(bond_char)
    if atom.isotope:
        builder.append(str(atom.isotope))
    builder.append(atom.element)

    assert atom.chirality is None

    if atom.h_count != 0 and atom.h_count is not None:
        builder.append("H")
        builder.append(str(atom.h_count))
    elif atom.h_count == 0 and not (atom.element in ORGANIC_SUBSET and atom.charge == 0 and atom.radical == 0):
        builder.append("H0")

    if atom.radical:
        builder.append(f'R{atom.radical}')

    if atom.charge is not None and atom.charge != 0:
        builder.append("{:+}".format(atom.charge))
    builder.append("]")

    return "".join(builder)

def group_encoder(grammar, mol, groups, join=True):
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        graph = MolecularGraph(mol, groups)
        group_selfies = graph_to_selfies(grammar, graph, None, 0)
        if join:
            group_selfies = ''.join(group_selfies)
        return group_selfies
    except Exception as e:
        print(e)
        raise ValueError('ISSUE WITH MOL', Chem.MolToSmiles(mol))
