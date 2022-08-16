#!/usr/bin/env python

"""
SELFIES: a robust representation of semantically constrained graphs with an
         example application in chemistry.

SELFIES (SELF-referencIng Embedded Strings) is a general-purpose,
sequence-based, robust representation of semantically constrained graphs.
It is based on a Chomsky type-2 grammar, augmented with two self-referencing
functions. A main objective is to use SELFIES as direct input into machine
learning models, in particular in generative models, for the generation of
outputs with high validity.

The code presented here is a concrete application of SELFIES in chemistry, for
the robust representation of molecules.

    Typical usage example:
        import selfies as sf

        benzene = "C1=CC=CC=C1"
        benzene_selfies = sf.encoder(benzene)
        benzene_smiles = sf.decoder(benzene_selfies)

For comments, bug reports or feature ideas, please send an email to
mario.krenn@utoronto.ca and alan@aspuru.com.
"""

__version__ = "2.0.0"

__all__ = [
    "get_preset_constraints",
    "get_semantic_robust_alphabet",
    "get_semantic_constraints",
    "set_semantic_constraints",
    "len_selfies",
    "split_selfies",
    "get_alphabet_from_selfies",
    "EncoderError",
    "DecoderError",
    "group_encoder",
    "group_decoder",
    "fragment_mols",
    "GroupGrammar",
    "Group"
]

from .bond_constraints import (
    get_preset_constraints,
    get_semantic_constraints,
    get_semantic_robust_alphabet,
    set_semantic_constraints
)
from .group_decoder import group_decoder
from .group_encoder import group_encoder
from .exceptions import DecoderError, EncoderError

from .utils.selfies_utils import (
    get_alphabet_from_selfies,
    len_selfies,
    split_selfies
)
import group_selfies.group_grammar
import group_selfies.group_mol_graph
from .group_mol_graph import (
    MolecularGraph,
    Group,
)
from .group_grammar import (
    GroupGrammar
)
from .utils.fragment_utils import (
    fragment_mols,   
)