import os 
import pickle 
import pytest
import pandas as pd 
import rdkit.Chem
from group_selfies import PROJECT_ROOT
from group_selfies.group_grammar import GroupGrammar

def roundtrip_checker(group_grammar:GroupGrammar,df,**kwargs):
    mismatch_smiles_input = []
    mismatch_smiles_output = []
    for i, smiles in enumerate(df['smiles'].to_list()):
        canon_smiles = rdkit.Chem.CanonSmiles(smiles)
        mol = rdkit.Chem.MolFromSmiles(canon_smiles)
        group_selfies = group_grammar.full_encoder(mol,**kwargs)
        reconstructed_mol = group_grammar.decoder(group_selfies,**kwargs)
        reconstructed_smiles = rdkit.Chem.CanonSmiles(rdkit.Chem.MolToSmiles(reconstructed_mol))
        if canon_smiles != reconstructed_smiles:
            mismatch_smiles_input.append(canon_smiles)
            mismatch_smiles_output.append(reconstructed_smiles)
    return mismatch_smiles_input,mismatch_smiles_output

def test_nonfullerene_subset_roundtrip(nonfullerene_subset,group_grammar_a):
    mismatch_smiles_input,mismatch_smiles_output = roundtrip_checker(group_grammar_a,nonfullerene_subset)
    for i, target_smiles in enumerate(mismatch_smiles_input):
        print(f"Expected {target_smiles}, got {mismatch_smiles_output[i]}")
    assert len(mismatch_smiles_input) == 0

@pytest.fixture
def numeric_index_alphabet():
    return tuple(["[0]","[1]","[2]","[3]","[4]","[5]","[6]","[7]","[8]","[9]","[a]","[b]","[c]","[d]","[e]","[f]"])

@pytest.fixture
def numeric_index_code(numeric_index_alphabet):
    return {code:index for index, code in enumerate(numeric_index_alphabet)}

def test_nonfullerene_subset_roundtrip_numeric_indices(nonfullerene_subset,group_grammar_a,numeric_index_alphabet):
    mismatch_smiles_input,mismatch_smiles_output = roundtrip_checker(group_grammar_a,nonfullerene_subset,index_alphabet = numeric_index_alphabet)
    for i, target_smiles in enumerate(mismatch_smiles_input):
        print(f"Expected {target_smiles}, got {mismatch_smiles_output[i]}")
    assert len(mismatch_smiles_input) == 0

def test_advanced_cases_roundtrip(advanced_set,group_grammar_a):
    mismatch_smiles_input,mismatch_smiles_output = roundtrip_checker(group_grammar_a,advanced_set)
    for i, target_smiles in enumerate(mismatch_smiles_input):
        print(f"Expected {target_smiles}, got {mismatch_smiles_output[i]}")
    assert len(mismatch_smiles_input) == 0


def test_advanced_cases_chiral_grammar_roundtrip(advanced_set,chiral_grammar):
    mismatch_smiles_input,mismatch_smiles_output = roundtrip_checker(chiral_grammar,advanced_set)
    for i, target_smiles in enumerate(mismatch_smiles_input):
        print(f"Expected {target_smiles}, got {mismatch_smiles_output[i]}")
    assert len(mismatch_smiles_input) == 0