# Author: Pyeong Eun Kim
# Date: Dec.8.2020
# Title: Helper functions for processing data or output of model
# Affiliation: JLK Genome Research Centre

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem import BondType
import torch
from cl_featuriser import AtomFeaturiser, BondFeaturiser
import dgl
from molvs import charge

def dgl_to_mol(G):

    # create mol object
    mol = Chem.RWMol()

    # {atomic_type: atomic number} for nodes in graph G
    atomic_num = {0:6, 1:7, 2:8, 3:9, 4:15, 5:16, 6:17, 7:35}
    #explicit_type = {0:0, 1:1, 2:2, 3:3}
    charges = {0:0, 1:1, 2:-1, 3:2, 4:-2, 5:3, 6:-3, 7:4, 8:-4}
    aromaticity = {0:False, 1:True}
    # add atoms to mol object
    #node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_num[G.ndata["atom_type"][node].item()])
        #a.SetIsAromatic(aromaticity[G.ndata["aromatic_type"][node].item()])
        #a.SetNumExplicitHs(G.ndata["explicit_hydrogen"][node].item())
        #a.SetFormalCharge(charges[G.ndata["charge_type"][node].item()])
        idx = mol.AddAtom(a)
        #node_to_idx[node.item()] = idx

    # {bond_type: rdkit.BondType} for edges in graph G
    bond_types = {
        0:Chem.rdchem.BondType.SINGLE,
        1:Chem.rdchem.BondType.DOUBLE,
        2:Chem.rdchem.BondType.TRIPLE
        }
    # add bonds to mol object
    edges = zip(G.edges()[0].tolist()[::2],G.edges()[1].tolist()[::2])
    #assert "incorrect bond", G.edata["bond_type"].tolist()[::2] == G.edata["bond_type"].tolist()[1::2]
    for j,edge in enumerate(edges):
        first, second = edge
        bond_type = bond_types[G.edata["bond_type"][j*2].item()]
        mol.AddBond(first, second, bond_type)
    # add charged to N where it has unnecessary unpaired electrons
    mol.UpdatePropertyCache(strict=False)
    pt = Chem.GetPeriodicTable()
    neutralise = charge.Uncharger()
    for j,at in enumerate(mol.GetAtoms()):
        atomic_num = at.GetAtomicNum()
        #print(j,at.GetSymbol(), at.GetFormalCharge(), at.GetNumExplicitHs(),at.GetExplicitValence(), at.GetTotalValence(), pt.GetDefaultValence(atomic_num))
        tv, dv = at.GetTotalValence(), pt.GetDefaultValence(atomic_num)
        if tv > dv:
            if at.GetNumExplicitHs() > 0:
                at.SetNumExplicitHs(at.GetNumExplicitHs() - (tv-dv))
                at.SetFormalCharge(0)
                print("N")
            elif at.GetNumExplicitHs() == 0 and at.GetFormalCharge() == 0:
                if at.GetSymbol() not in ["P", "S"]:
                    at.SetFormalCharge(tv-dv)
            elif at.GetNumExplicitHs() == 0 and at.GetExplicitValence() == dv and at.GetFormalCharge() != 0:
                if at.GetSymbol() not in ["P", "S"]:
                    at.SetFormalCharge(0)
            #elif at.GetNumExplicitHs() == 0 and at.GetExplicitValence() > 0
            if at.GetSymbol() in ["P", "S"]:
                if at.GetExplicitValence() != at.GetTotalValence():
                    at.SetFormalCharge(at.GetTotalValence() - at.GetExplicitValence())

        elif tv < dv and at.GetFormalCharge() != 0:
            at.SetFormalCharge(0)
            at.SetNumExplicitHs(at.GetNumExplicitHs() + dv-tv)
        at.SetNumRadicalElectrons(0)
        #print(j,at.GetSymbol(), at.GetFormalCharge(), at.GetNumExplicitHs(),at.GetExplicitValence(),at.GetTotalValence(), pt.GetDefaultValence(atomic_num))


    Chem.rdmolops.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    #print(Chem.MolToSmiles(mol, canonical=True))
    return mol, Chem.MolToSmiles(mol, canonical=True)


def mol_to_graph(mol, canonical_atom_order=True):
    atom_featuriser = AtomFeaturiser()
    bond_featuriser = BondFeaturiser()
    if mol is None:
        print('Invalid mol found')
        return None

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    g = construct_bigraph_from_mol(mol)

    g.ndata.update(atom_featuriser(mol))

    g.edata.update(bond_featuriser(mol))

    return g

def enum_stereoisomer(molecule, training=True):
    smi = Chem.MolToSmiles(molecule, isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)
    isomers_mol = tuple(EnumerateStereoisomers(mol))
    graph_list = []
    smi_list = []
    for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers_mol):
        smi_list.append(smi)
        mol = Chem.MolFromSmiles(smi)
        Chem.rdmolops.Kekulize(mol)
        g = mol_to_graph(mol)
        graph_list.append(g)
    if not training:
        return dgl.batch(graph_list), smi_list
    else:
        return dgl.batch(graph_list)

def construct_bigraph_from_mol(mol):

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
