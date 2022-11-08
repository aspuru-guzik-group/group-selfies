"""Script for testing group selfies against large datasets.

download emolecules dataset from https://downloads.emolecules.com/free/2022-11-01/version.smi.gz
"""

import pathlib

from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import group_selfies as gs
m = Chem.MolFromSmiles
c = Chem.CanonSmiles
s = Chem.MolToSmiles
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
            check(in_smiles)
        except Exception as e:
            q.put(f'{in_smiles},{e}')
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
grammar.add_group('phosphine8', '*x1[P@+](*x1)(*x1)*x1', 10)
grammar.add_group('silicon', '*x1[Si@H](*x1)(*x1)', 10)
grammar.add_group('silicon2', '*x1[Si@](*x1)(*x1)*x1', 15)
grammar.add_group('phosphine9', '*x1[P@-](*x1)*x1', 10)

def check(smi):
    # prepare smiles:
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    retry = []
    for frag_smi in smi.split('.'):
        mol = Chem.MolFromSmiles(frag_smi)
        gselfi = grammar.full_encoder(mol)
        new_mol = grammar.decoder(gselfi)
        new_smi = Chem.MolToSmiles(new_mol)
        old_smi = Chem.CanonSmiles(frag_smi)
        new_smi = Chem.CanonSmiles(new_smi)
        if old_smi != new_smi:
            new_new_smi = Chem.CanonSmiles(new_smi)
            if old_smi != new_new_smi:
                retry.append((mol, gselfi))
    
    for mol, gselfi in retry:
        for _ in range(50):
            if c(s(grammar.decoder(gselfi), doRandom=True)) == c(s(mol, doRandom=True)):
                return
        raise ValueError(f"Failed to roundtrip {smi}")

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
