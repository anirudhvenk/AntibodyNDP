import pandas as pd
import numpy as np
import re

raw_mut_df = pd.read_csv("./data/urn_mavedb_00000001-a-1_scores.csv")
alphabet = ["A", "V", "L", "I", "M", "F", "Y", "W", "R", "H", "K", "D", "E", "S", "T", "N", "Q", "G", "C", "P"]
aa_codes = {"Ala": "A", 
            "Val": "V", 
            "Leu": "L",
            "Ile": "I",
            "Met": "M",
            "Phe": "F",
            "Tyr": "Y",
            "Trp": "W",
            "Arg": "R",
            "His": "H",
            "Lys": "K",
            "Asp": "D",
            "Glu": "E",
            "Ser": "S",
            "Thr": "T",
            "Asn": "N",
            "Gln": "Q",
            "Gly": "G",
            "Cys": "C",
            "Pro": "P"}

orig_seq = "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS"
muts = raw_mut_df["hgvs_pro"][:-1]
scores = raw_mut_df["score"][:-1]
mut_matrix = np.zeros((len(orig_seq), len(alphabet)))
mut_pattern = r"(\D+)(\d+)(.*)"


for mut, score in zip(muts, scores):
    orig_aa = re.search(mut_pattern, mut).group(1)[2:]
    mut_idx = int(re.search(mut_pattern, mut).group(2)) - 1
    mut_aa = re.search(mut_pattern, mut).group(3)
    
    if mut_idx == len(orig_seq):
        continue
    
    if mut_aa == "=":
        mut_aa = orig_aa
    
    mut_matrix[mut_idx][alphabet.index(aa_codes[mut_aa])] = score
    
print(mut_matrix.shape)
