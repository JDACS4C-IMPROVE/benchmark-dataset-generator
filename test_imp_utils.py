import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional


fdir = Path(__file__).resolve().parent

import improve_utils
from improve_utils import improve_globals  # imp_globals, load_rsp_data, read_df, get_common_samples, load_ge_data
from improve_utils import load_single_drug_response_data

# import pdb; pdb.set_trace()
# Load drug response data
# rs = improve_utils.load_single_drug_response_data(source="CCLE")  # load all samples
# rs = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type=["train", "val"])
rs_tr = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="train")  # load train
rs_vl = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="val")    # load val
rs_te = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="test")   # load test

# Load omic feature data
# cv = improve_utils.load_copy_number_data(gene_system_identifier="Gene_Symbol")
# ge = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
# mt = improve_utils.load_dna_methylation_data(gene_system_identifier="TSS")

# Load drug feature data
sm = improve_utils.load_smiles_data()
dd = improve_utils.load_mordred_descriptor_data()
fp = improve_utils.load_morgan_fingerprint_data()

print(df.shape)
