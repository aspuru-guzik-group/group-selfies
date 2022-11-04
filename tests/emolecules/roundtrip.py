"""Script for testing group selfies against large datasets.

download emolecules dataset from https://downloads.emolecules.com/free/2022-11-01/version.smi.gz
"""

import pathlib

from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import group_selfies as gs

import multiprocessing as mp

TEST_DIR = pathlib.Path(__file__).parent
TEST_SET_PATH = TEST_DIR / "version.smi"
ERROR_PATH = TEST_DIR / "errors.txt"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def make_reader():
    return pd.read_csv(TEST_SET_PATH, sep=' ', chunksize=10000)


def worker(in_smiles, q):
    in_smiles = in_smiles.strip()

    mol = Chem.MolFromSmiles(in_smiles, sanitize=True)
    if not ((mol is None) or ("*" in in_smiles)):
        try:
            for frag_smiles in in_smiles.split('.'):
                check(frag_smiles)
        except:
            q.put(in_smiles)
            return in_smiles

def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open(ERROR_PATH, "w") as f:
        while True:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()

grammar = gs.GroupGrammar.essential_set()
def check(smi):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    selfi = grammar.full_encoder(mol)
    new_mol = grammar.decoder(selfi)
    new_smi = Chem.MolToSmiles(new_mol)
    assert Chem.CanonSmiles(smi) == Chem.CanonSmiles(new_smi)

def roundtrip_translation():
    constraints = gs.get_preset_constraints("hypervalent")
    constraints.update({"P": 7, "P-1": 8, "P+1": 6, "?": 12})
    gs.set_semantic_constraints(constraints)
    
    n_entries = 0
    for chunk in make_reader():
        n_entries += len(chunk)
    pbar = tqdm(total=n_entries)

    reader = make_reader()

    def update(arg):
        if arg is not None:
            tqdm.write(str(arg))
        pbar.update(1)
    
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(processes=None)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for _, chunk in enumerate(reader):
        for in_smiles in chunk['isosmiles']:
            job = pool.apply_async(worker, (in_smiles, q), callback=update)
            jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == "__main__":
    roundtrip_translation()
