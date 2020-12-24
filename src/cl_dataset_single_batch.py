# Author: Pyeong Eun Kim
# Date: Nov.30.2020
# Title: Mol to Graph for Graph-based Deep Generative Model version 3
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
                 charge_types=None, aromatic_types=None, implicit_hydrogen=None):
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
            charge_types = [0, 1, -1]
        self._charge_types = charge_types

        if aromatic_types is None:
            aromatic_types = [False, True]
        self._aromatic_types = aromatic_types

        if implicit_hydrogen is None:
            implicit_hydrogen = [0, 1, 2, 3]
        self._implicit_hydrogen = implicit_hydrogen

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
                self._implicit_hydrogen.index(atom.GetTotalNumHs())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.uint8))

        return {
            'atom_type': atom_features[:, 0].long(),
            'chirality_type': atom_features[:, 1].long(),
            'charge_type': atom_features[:, 2].long(),
            'aromatic_type': atom_features[:, 3].long(),
            'implicit_hydrogen': atom_features[:, 4].long()
        }


class BondFeaturiser(object):
    def __init__(self, bond_types=None, stereo_types=None, self_loop=False):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]
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



class MolGraphData(Dataset):
    def __init__(self, path_to_df, node_featuriser, edge_featuriser, conditions, normalised=True):
        super(MolGraphData, self).__init__()
        self.whole_df = pd.read_csv(path_to_df)
        self.atom_featuriser = node_featuriser
        self.bond_featuriser = edge_featuriser
        conditions.extend(["scaffold_"+c for c in conditions])
        self.conditions = conditions
        print("condition",conditions, self.conditions)
        self.data_len = self.whole_df.shape[0]
        self.normalised = normalised
        if not normalised:
          self.mean = {"clogp":3.589836, "tpsa":77.059375, "mw":389.977723}
          self.std = {"clogp":1.961023, "tpsa":25.281207, "mw":80.655741}

    def __getitem__(self, index):
        single_smi = self.whole_df.loc[index].smiles
        #print(single_smi)
        mol = Chem.MolFromSmiles(single_smi)
        g, g_scaffold, action = self.mol_to_graph(mol)
        conditions = torch.tensor(self.whole_df.loc[index][self.conditions])
        if not self.normalised:
          """TO DO: if not normalised standardise following inputs"""
          pass
        return g, g_scaffold, action, conditions

    def construct_bigraph_from_mol(self, mol):

        g = dgl.graph(([], []), idtype=torch.int32)

        # Add nodes
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        # Add edges
        src_list = []
        dst_list = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])

        g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

        return g

    def mol_to_graph(self, mol, canonical_atom_order=True):
        if mol is None:
            print('Invalid mol found')
            return None

        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        # get core from whole mol
        core = MurckoScaffold.GetScaffoldForMol(mol)
        new_order_core = rdmolfiles.CanonicalRankAtoms(core)
        core = rdmolops.RenumberAtoms(core, new_order_core)
        try:
            sub_order = list(mol.GetSubstructMatches(core)[0])
            scaffold_list = sub_order
        except:
            print(Chem.MolToSmiles(mol), "\n",Chem.MolToSmiles(core))
            print(mol.GetSubstructMatches(core))

        for i in range(mol.GetNumAtoms()):
            if i in sub_order:
                continue
            else:
                scaffold_list.append(i)
        mol = Chem.RenumberAtoms(mol, tuple(scaffold_list))
        g = self.construct_bigraph_from_mol(mol)
        g_scaffold = self.construct_bigraph_from_mol(core)


        g.ndata.update(self.atom_featuriser(mol))
        g_scaffold.ndata.update(self.atom_featuriser(core))

        g.edata.update(self.bond_featuriser(mol))
        g_scaffold.edata.update(self.bond_featuriser(core))

        actions = []
        src, dest = g.edges()
        for i in range(core.GetNumAtoms(), mol.GetNumAtoms()):
            node_feat = g.ndata["atom_type"][i]
            edge_dests = torch.nonzero(src==i, as_tuple=False).flatten()
            edge_srcs = dest[edge_dests]
            edge_srcs_index = torch.nonzero(edge_srcs<i, as_tuple=False).flatten()
            edge_srcs = edge_srcs[edge_srcs_index]
            edge_feat = g.edata["bond_type"][edge_dests]
            edge_feat = edge_feat[edge_srcs_index]
            #[atom type, edge type, destination]
            if len(edge_srcs) > 0:
                actions.append([node_feat, edge_feat, edge_srcs])
            elif len(edge_srcs) == 0:
                actions.append([node_feat, torch.tensor([self.bond_featuriser.max_bond_type() -1], dtype=torch.uint8), False])
        actions.append([torch.tensor([self.atom_featuriser.max_atom_type()], dtype=torch.uint8), False, False])

        return g, g_scaffold, actions

    def __len__(self):
        return self.data_len

    def collate_fn(self, batch):
        for i, (g, g_scaffold, action, condition) in enumerate(batch):
            pass
        #nodes = [action[i][0] for i in range(len(action))]
        #edges = [actions[i][1] for i in range(len(action))]
        #dests = [actions[i][2] for i in range(len(action))]
        return g, g_scaffold, action, condition




if __name__=="__main__":
    import time
    from cl_graph_v3 import ScaffoldGNN
    #IPythonConsole.ipython_useSVG=True
    start = time.process_time()
    conditions = ["clogp","mw","tpsa"]
    path_to_df = "/BiO/pekim/GRAPHNET/data/small_data.csv"
    dataset = MolGraphData(path_to_df, AtomFeaturiser(), BondFeaturiser(), conditions)
    end = time.process_time()
    print("time spent to initialise dataset {}".format(end-start))
    leng = dataset.__len__()
    model = ScaffoldGNN(6)
    start = time.process_time()
    ds_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=dataset.collate_fn,batch_size=1, shuffle=False)
    for i, (g, g_scaffold, action, condition) in enumerate(ds_loader):
        print(i)
        model(g, g_scaffold, condition, action)
        if i == 39:
            break
    end = time.process_time()
    print("time spent to read dataset {}".format(end-start))
    """start = time.process_time()
    for i in range(leng):
        ms = time.process_time()
        g, g_scaffold, actions, conditions = dataset.__getitem__(i)
        z, logvar, mu, n, e, d = model(g, g_scaffold, conditions, actions)
        me = time.process_time()
        #print(z.size())
        print(i, me-ms)
    #z, logvar, mu, n, e, d = model(g, g_scaffold, conditions, actions)
    end = time.process_time()"""
    #print(n)
    #print(e)
    #print(d)
    print("time spent to load {} data {}".format(leng, end-start))
    """
    start = time.process_time()
    s = 35
    e = 40
    for i in range(s,e):
      dataset.__getitem__(i)
    end = time.process_time()
    print("time spent to load 1000 data {}".format(end-start))
    """
    """smile = 'CC(=O)Oc1ccc(cc1)C(=O)[C@@H]1N2C=Cc3c([C@@H]2[C@@H]2[C@H]1C(=O)N(C2=O)c1ccc(cc1Cl)Cl)cccc3' # formic
    mol = Chem.MolFromSmiles(smile)
    af = AtomFeaturiser()
    bf = BondFeaturiser()
    print(af(mol))
    print(bf(mol))
    g, s, p = mol_to_graph(mol, AtomFeaturiser(), BondFeaturiser())
    print(g.nodes(), len(g.nodes()))
    print(s.nodes(), len(s.nodes()))
    for i in p:
      print(i)"""
