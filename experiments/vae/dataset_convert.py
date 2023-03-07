from rdkit import Chem
from group_selfies.utils.group_utils import extract_group_bonds
from group_selfies.group_mol_graph import Group, MolecularGraph
from group_selfies.group_grammar import GroupGrammar

grammar = GroupGrammar.essential_set() | GroupGrammar.from_file('../useful30.txt')
    
import moses
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
new_train = []

train = moses.get_dataset('train')

train_dataloader = DataLoader(
    train, batch_size=512,
    shuffle=False, num_workers=8
)



"""Script for testing selfies against large datasets.
"""

import argparse
import pathlib

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import os

import group_selfies as sf

import multiprocessing as mp

def worker(smis, l):
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        res = grammar.encoder(m, grammar.extract_groups(m), join=False)
        l.append(tuple(res))

def roundtrip_translation():
    constraints = sf.get_preset_constraints("hypervalent")
    constraints.update({"P": 7, "P-1": 8, "P+1": 6, "?": 12, 'C+1': 5})
    sf.set_semantic_constraints(constraints)
    
    pbar = tqdm(total=len(train))
    def update(arg):
        pbar.update(512)
    manager = mp.Manager()
    l = manager.list()
    pool = mp.Pool(processes=None)
    #put listener to work first
    # watcher = pool.apply_async(listener, (q,))
    #fire off workers
    jobs = []
    for batch in train_dataloader:
        job = pool.apply_async(worker, (batch, l), callback=update)
        if len(pool._cache) > 1e6:
            job.wait()
        jobs.append(job)
            
    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    pool.close()
    pool.join()
    pickle.dump(list(l), open('train_new.pkl', 'wb'))


if __name__ == "__main__":
    roundtrip_translation()
    
