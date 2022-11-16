#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="group_selfies",
    version="1.0.0",
    author="Austin Cheng, Andy Cai, Mario Krenn, Alston Lo, and many other contributors",
    author_email="austin.cheng@mail.utoronto.ca, alan@aspuru.com",
    description="Group SELFIES incorporates group tokens which represent functional groups or entire substructures into SELFIES while maintaining robustness.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aspuru-guzik-group/group-selfies",
    packages=setuptools.find_packages(),
    package_data = {
        '': ['*'],
        'test': ['*.txt']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'networkx',
        'tqdm',
        'rdkit',
        'global_chem',
    ]
)