import pandas as pd
import numpy as np
import os
from collections import Counter
from pathlib import Path

from Functions import generate_cross_validation_partition


fdir = Path(__file__).resolve().parent

pd.set_option('display.max_columns', None)

omics_data_dir = fdir/'../../Data_Curation_final/Curated_CCLE_Multiomics_files/'

response_data_dir = fdir/'../../Response_Data'

drug_data_dir = fdir/'../../Drug_Data'

auxiliary_data_dir = fdir/'../auxiliary_data'

benchmark_data_dir = fdir/'../csa_data'
y_data_dir = benchmark_data_dir/'y_data'
x_data_dir = benchmark_data_dir/'x_data'
splits_dir = benchmark_data_dir/'splits'
os.makedirs(benchmark_data_dir, exist_ok=True)
os.makedirs(y_data_dir, exist_ok=True)
os.makedirs(x_data_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)



# Load response data
res = pd.read_csv(response_data_dir/'experiments.tsv', sep='\t', engine='c', na_values=['na', '-', ''], header=0,
                  index_col=None)
print("Experiments (response):", res.shape)
res.columns = [str(i).lower() for i in res.columns[:2]] + ['improve_chem_id'] + [str(i).lower() for i in res.columns[3:]]
res.source = res.study

res.improve_sample_id = ['CC_' + str(int(x)) for x in res.improve_sample_id]
id = np.where(np.isin(res.source, ['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI']))[0]
res = res.iloc[id, :]
id = np.where(np.invert(pd.isna(res.auc)))[0]
res = res.iloc[id, :]
id = np.intersect1d(np.where(res.auc >= 0)[0], np.where(res.auc <= 1)[0])
res = res.iloc[id, :]

# Load cell line information and generate a mapping between cell line IMPROVE IDs and Cellosaurus IDs.
ccl_info = pd.read_csv(response_data_dir/'samples.csv', sep=',', engine='c', na_values=['na', '-', ''], header=0,
                  index_col=None, low_memory=False)
imp_id = np.unique(ccl_info.improve_sample_id.map(str))
imp_id_mapping = pd.DataFrame({'improve_sample_id': ['CC_' + str(i) for i in imp_id], 'other_id': imp_id})
ccl_info.improve_sample_id = ['CC_' + str(x) for x in ccl_info.improve_sample_id]
id = np.where(ccl_info.id_source == 'Cellosaurus')[0]
ccl_mapping = ccl_info.iloc[id, :]
ccl_mapping = ccl_mapping.loc[:, ['improve_sample_id', 'other_id']].drop_duplicates()
id = np.where(np.invert(np.isin(imp_id_mapping.improve_sample_id, ccl_mapping.improve_sample_id)))[0]
ccl_mapping = pd.concat((ccl_mapping, imp_id_mapping.iloc[id, ]), axis=0)

# Load the list of cell lines with gene expressions, mutations, copy numbers, and DNA methylations
ccl_list = pd.read_csv(auxiliary_data_dir/'ccl_ge_mu_cnv_me.txt', sep='\t', engine='c',
                               na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
id = np.where(np.isin(ccl_mapping.other_id, ccl_list))[0]
ccl_mapping = ccl_mapping.iloc[id, :]

# Load drug fingerprint data
df = pd.read_parquet(drug_data_dir/'ecfp4_nbits512.parquet', engine='pyarrow')
df = df.drop_duplicates()
df.index = df.improve_chem_id
df = df.iloc[:, 1:]
print(df.drop_duplicates().shape[0]) # 41366

# Load drug descriptor data
dd = pd.read_parquet(drug_data_dir/'mordred.parquet', engine='pyarrow')
dd = dd.drop_duplicates()
dd.index = dd.improve_chem_id
dd = dd.iloc[:, 1:]
print(dd.drop_duplicates().shape[0]) # 41785

# Load drug information
drug_info = pd.read_csv(drug_data_dir/'drug_meta.tsv', sep='\t', engine='c', na_values=['na', '-', ''],
                        header=0, index_col=None, low_memory=False)
drug_info.canSMILES = drug_info.canSMILES_new
drug_info = drug_info.loc[:, ['formula', 'weight', 'canSMILES', 'isoSMILES', 'InChIKey', 'chem_name',
       'improve_chem_id']]
print(len(np.unique(drug_info.canSMILES))) # 41785
print(len(np.unique(drug_info.improve_chem_id))) # 47409

# # This code shows that many canSMILES associate with more than one improve_chem_id/pubchem_id
# # So improve_chem_id/pubchem_id is not good as drug identifier. We shall use either canSMILES or descriptor profiles
# # to define unique drugs.
# uni_s = np.unique(drug_info.canSMILES)
# for i in uni_s:
#     idi = np.where(drug_info.canSMILES == i)[0]
#     if len(np.unique(drug_info.iloc[idi, :].improve_chem_id)) > 1:
#         print(drug_info.iloc[idi, :])

drug_list = np.unique(np.intersect1d(np.intersect1d(df.index, dd.index),
                                     np.intersect1d(drug_info.improve_chem_id, res.improve_chem_id)))

# Keep only (1) cell lines have both omics data and response data and (2) drugs with both feature data and response data
id = np.intersect1d(np.where(np.isin(res.improve_sample_id, ccl_mapping.improve_sample_id))[0],
                    np.where(np.isin(res.improve_chem_id, drug_list))[0])
res = res.iloc[id, :]
res.to_csv(y_data_dir/'response.txt', header=True, index=False, sep='\t', line_terminator='\r\n')

id = np.where(np.isin(ccl_mapping.improve_sample_id, res.improve_sample_id))[0]
ccl_mapping = ccl_mapping.iloc[id, :]
ccl_mapping = ccl_mapping.drop_duplicates()
ccl_mapping.to_csv(x_data_dir/'ccl_info.txt', header=True, index=False, sep='\t', line_terminator='\r\n')


id = np.where(np.isin(drug_list, res.improve_chem_id))[0]
drug_list = drug_list[id]
id = np.where(np.isin(drug_info.improve_chem_id, drug_list))[0]
drug_info = drug_info.iloc[id, :].drop_duplicates()
drug_info.to_csv(x_data_dir/'drug_info.txt', header=True, index=False, sep='\t', line_terminator='\r\n')

study = np.unique(res.source)
for s in study:
    ids = np.where(res.source == s)[0]
    pd.DataFrame(ids).to_csv(splits_dir/(s + '_all.txt'), header=False, index=False, sep='\t',
                             line_terminator='\r\n')
    p = generate_cross_validation_partition(list(range(len(ids))), n_folds=10, n_repeats=1, portions=[8, 1, 1],
                                            random_seed=1)
    for i in range(len(p)):
        pd.DataFrame(ids[p[i][0]]).to_csv(splits_dir/(s + '_split_' + str(i) + '_train.txt'), header=False,
                                          index=False, sep='\t', line_terminator='\r\n')
        pd.DataFrame(ids[p[i][1]]).to_csv(splits_dir/(s + '_split_' + str(i) + '_val.txt'), header=False,
                                          index=False, sep='\t', line_terminator='\r\n')
        pd.DataFrame(ids[p[i][2]]).to_csv(splits_dir/(s + '_split_' + str(i) + '_test.txt'), header=False,
                                          index=False, sep='\t', line_terminator='\r\n')


print(Counter(res.source))
print('Number of cell lines is ' + str(len(np.unique(res.improve_sample_id))))
print('Number of drugs is ' + str(len(np.unique(res.improve_chem_id))))
# Counter({'CTRPv2': 153826, 'GDSCv1': 134773, 'GDSCv2': 66608, 'gCSI': 10541, 'CCLE': 9418})
# Number of cell lines is 788
# Number of drugs is 493