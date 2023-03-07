from torch.utils.data import DataLoader
from moses import CharVocab, StringDataset
from moses.utils import OneHotVocab
from moses.vae import VAE, VAETrainer, vae_parser
from moses.models_storage import ModelsStorage
from moses.script_utils import add_train_args, read_smiles_csv, set_seed
import moses
import argparse
import group_selfies as gsf

MODELS = ModelsStorage()
def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser

parser = get_parser()
config = parser.parse_args()

import pickle
import global_grammar

train = pickle.load(open('train_new.pkl', 'rb'))
vocab = OneHotVocab(list(global_grammar.grammar.all_tokens()) + list(gsf.get_semantic_robust_alphabet()))

config.freeze_embeddings = False

train_dataset = StringDataset(vocab, train)
model = VAE(vocab, config)
trainer = VAETrainer(config)

trainer.fit(model, train_dataset)
