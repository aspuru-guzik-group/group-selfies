# Group SELFIES
https://arxiv.org/abs/2211.13322

## Installation
Python >= 3.8 is required.

RDKit is required to use this package. Once it is installed, clone this repository using
```bash
git clone https://github.com/aspuru-guzik-group/group-selfies
```
and run 
```bash
pip install .
``` 
in the cloned folder.

## Introduction
Group SELFIES extends SELFIES with the ability to represent groups with single tokens. This improves **interpretability**, **compactness**, and **performance in generative models**.

| Encoding |
|:--:|
| ![Encoding process](encoding.gif) |

| Decoding |
|:--:|
| ![Decoding process](decoding.gif) |



### Usage
See [tutorial.ipynb](tutorial/tutorial.ipynb) for details on usage. For key classes/functions, see below:

| Class/Function                              | Description                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| ``group_selfies.Group``                   | Class that represents groups. |
| ``group_selfies.GroupGrammar``                   | Class that represents a grammar, which is a set of groups used for encoding and decoding. |
| ``grammar.extract_groups``             | Finds occurences of the grammar's defined set of groups in a molecule.       |
| ``grammar.encoder``             | Encodes a molecule to its corresponding Group SELFIES representation. Requires extracted group occurences returned by ``grammar.extract_groups``          |
| ``grammar.decoder`` | Decodes a Group SELFIES string to its corresponding molecule.        |
| ``grammar.full_encoder``       | Extracts groups from a molecule and encodes it, essentially a combination of ``grammar.extract_groups`` and ``grammar.encoder``. Mainly for convenience. |
| ``group_selfies.fragment_mols``       | Fragments a set of molecules into a set of reasonable groups.       |

