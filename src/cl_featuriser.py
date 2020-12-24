# Author: Pyeong Eun Kim
# Date: Dec.07.2020
# Title: Graph-based Deep Generative Model version 3 (single batch only)
# Affiliation: JLK Genome Research Centre

from dataclasses import dataclass, field
from typing import List
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import itertools
import numpy as np
import dgl.backend as F
from enum import Enum, unique
from rdkit.Chem import BondType
import dgl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class AtomFeaturiser(object):

    def __init__(self, atom_types=None, chiral_types=None,
                 charge_types=None, aromatic_types=None,
                 implicit_hydrogen=None, hybridisation = None):
        if atom_types is None:
            atom_types = ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
        self._atomic_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                            Chem.rdchem.ChiralType.CHI_OTHER]
        self._chiral_types = chiral_types

        if charge_types is None:
            charge_types = [0, 1, -1, 2, -2, 3,-3, 4, -4]
        self._charge_types = charge_types

        if aromatic_types is None:
            aromatic_types = [False, True]
        self._aromatic_types = aromatic_types

        if implicit_hydrogen is None:
            implicit_hydrogen = [0, 1, 2, 3]
        self._explicit_hydrogen = implicit_hydrogen
        """
        if hybridisation is None:
            hybridisation = [Chem.rdchem.HybridizationType.OTHER,
                             Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2,
                             Chem.rdchem.HybridizationType.SP3,
                             Chem.rdchem.HybridizationType.SP3D,
                             Chem.rdchem.HybridizationType.SP3D2,
                             Chem.rdchem.HybridizationType.UNSPECIFIED
                             ]
        self._hybridisation = hybridisation"""
    def max_atom_type(self):
      return len(self._atomic_types)

    def __call__(self, mol):
        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append([
                self._atomic_types.index(atom.GetSymbol()),
                self._chiral_types.index(atom.GetChiralTag()),
                self._charge_types.index(atom.GetFormalCharge()),
                self._aromatic_types.index(atom.GetIsAromatic()),
                self._explicit_hydrogen.index(atom.GetNumExplicitHs())  #self._hybridisation.index(atom.GetHybridization())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.uint8))

        return {
            'atom_type': atom_features[:, 0].long(),
            'chirality_type': atom_features[:, 1].long(),
            'charge_type': atom_features[:, 2].long(),
            'aromatic_type': atom_features[:, 3].long(),
            'explicit_hydrogen': atom_features[:, 4].long()
        }


class BondFeaturiser(object):
    def __init__(self, bond_types=None, stereo_types=None, self_loop=False):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE
            ]
            """bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]"""
        self._bond_types = bond_types

        if stereo_types is None:
            stereo_types = [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
                Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE
            ]
        self._stereo_types = stereo_types
        self._self_loop = self_loop
    def max_bond_type(self):
      return len(self._bond_types)

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'bond_type' and 'bond_direction_type' separately to an int64
            tensor of shape (N, 1), where N is the number of edges.
        """
        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feats = [
                self._bond_types.index(bond.GetBondType()),
                self._stereo_types.index(bond.GetStereo())
            ]
            edge_features.extend([bond_feats, bond_feats.copy()])

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = F.zerocopy_from_numpy(edge_features.astype(np.uint8))

        return {
            'bond_type': edge_features[:, 0].long(),
            'stereo_type': edge_features[:, 1].long()
        }
