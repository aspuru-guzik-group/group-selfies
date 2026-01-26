import os
import pathlib
import pytest 
import pickle
import pandas as pd 
import rdkit.Chem 
from group_selfies import Group 
from group_selfies.group_grammar import GroupGrammar

PROJECT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent 

@pytest.fixture
def project_root():
    return PROJECT_ROOT

@pytest.fixture
def nonfullerene_path():
    return os.path.join(PROJECT_ROOT,"tests","test_sets","nonfullerene.csv") 
@pytest.fixture
def nonfullerene_subset(nonfullerene_path):
    subset_df = pd.read_csv(nonfullerene_path,nrows = 1000)
    return subset_df 

@pytest.fixture
def advanced_path():
    return os.path.join(PROJECT_ROOT,"tests","test_sets","advanced.csv") 
@pytest.fixture
def advanced_set(advanced_path):
    return pd.read_csv(advanced_path)

@pytest.fixture
def group_grammar_a_path():
    return os.path.join(PROJECT_ROOT,"tests","test_grammars","grammar_a")

@pytest.fixture
def group_grammar_a(group_grammar_a_path):
    with open(group_grammar_a_path,"rb") as grammar_file:
        group_grammar = pickle.load(grammar_file)
    return GroupGrammar(vocab=group_grammar.vocab)


@pytest.fixture
def chiral_grammar_path():
    return os.path.join(PROJECT_ROOT,"tests","test_grammars","chiral_grammar")

@pytest.fixture
def chiral_grammar(chiral_grammar_path):
    with open(chiral_grammar_path,"rb") as grammar_file:
        group_grammar = pickle.load(grammar_file)
    return GroupGrammar(vocab=group_grammar.vocab)


@pytest.fixture
def basic_gramamr():
    group = Group("bromine","Br",all_attachment=True)
    basic_grammar = GroupGrammar([group])
    return basic_grammar

@pytest.fixture
def aptazapine_smiles():
    return "CN1CCN2c3ccccc3Cn3cccc3[C@@H]2C1"

@pytest.fixture
def aptazapine_mol(aptazapine_smiles):
    return rdkit.Chem.MolFromSmiles(aptazapine_smiles)
