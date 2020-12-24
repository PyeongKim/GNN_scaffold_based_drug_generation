import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm

def protein_seq_parser(fasta):
    reader = open(fasta, "r")
    line = reader.readline()
    aa_list = []
    while line:
        aa_frag = line.rstrip()
        aa_frag = list(aa_frag)
        aa_frag = [re.sub(r"[UZOB]", "X", sequence) for sequence in aa_frag]
        aa_list.extend(aa_frag)
        line = reader.readline()
    return aa_list

def id_generation(aa_list, device=0):
    ids = tokenizer.batch_encode_plus(aa_list, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    return input_ids, attention_mask

def protein_embedding(model, input_ids, attention_mask, device=0):
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    embedding = embedding.cpu().numpy()
    features = []
    # Remove padding ([PAD]) and special tokens ([CLS],[SEP]) that is added by ProtBert-BFD model
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)

    return features


if __name__=="__main__":

    device = 0
    # load pre-trained model
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
    model = model.eval()

    prot_fasta = "/BiO/pekim/example.fasta"
    sequence = protein_seq_parser(prot_fasta)
    input_ids, attention_mask = id_generation(sequence, device)
    features = protein_embedding(model, input_ids, attention_mask, device)
     """
     # TO DO: Use features as input of MLP.
     """
