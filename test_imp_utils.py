import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional


fdir = Path(__file__).resolve().parent

import improve_utils
from improve_utils import improve_globals # imp_globals, load_rsp_data, read_df, get_common_samples, load_ge_data
from improve_utils import load_single_drug_response_data

import pdb; pdb.set_trace()
# rs = improve_utils.load_single_drug_response_data(source="CCLE")
# cv = improve_utils.load_copy_number_data(gene_system_identifier="Gene_Symbol")
ge = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")

# cv = improve_utils.load_copy_number_data(gene_system_identifier="Entrez")
# cv = improve_utils.load_copy_number_data(gene_system_identifier=["Entrez"])
# cv = improve_utils.load_copy_number_data(gene_system_identifier=["Entrez", "Gene_Symbol", "Ensembl"])
# cv = improve_utils.load_copy_number_data(gene_system_identifier=["Entrez", "Ensembl"])
# cv = improve_utils.load_copy_number_data(gene_system_identifier=["Blah", "Ensembl"])

print(df.shape)
