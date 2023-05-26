import pandas as pd
import numpy as np
from collections import Counter
from Functions import replace_ccl_name
from pathlib import Path


fdir = Path(__file__).resolve().parent

pd.set_option('display.max_columns', None)

omics_data_dir = fdir/'../../Data_Curation_final/Curated_CCLE_Multiomics_files'

benchmark_data_dir = fdir/'../csa_data'
x_data_dir = benchmark_data_dir/'x_data'



# Load cell line information
ccl_info = pd.read_csv(x_data_dir/'ccl_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
                 header=0, index_col=None, low_memory=False)



# Load gene expression data
ge = pd.read_csv(omics_data_dir/'CCLE_AID_expression_full.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
ge = ge.drop_duplicates()
ge = replace_ccl_name(data=ge, keep_id=[0, 1, 2], ccl_info=ccl_info)
ge.iloc[0, 0] = ''
ge.to_csv(x_data_dir/'cancer_gene_expression.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
ge = None



# Load copy number data
cn = pd.read_csv(omics_data_dir/'CCLE_AID_gene_cn.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
cn = cn.drop_duplicates()
cn = replace_ccl_name(data=cn, keep_id=[0, 1, 2], ccl_info=ccl_info)
cn.iloc[0, 0] = ''
cn.to_csv(x_data_dir/'cancer_copy_number.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
cn = None



# Load discretized copy number data
discretized_cn = pd.read_csv(omics_data_dir/'CCLE_AID_gene_cn_discretized.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
discretized_cn = discretized_cn.drop_duplicates()
discretized_cn = replace_ccl_name(data=discretized_cn, keep_id=[0, 1, 2], ccl_info=ccl_info)
discretized_cn.iloc[0, 0] = ''
discretized_cn.to_csv(x_data_dir/'cancer_discretized_copy_number.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
discretized_cn = None



# Load miRNA expression data
miRNA = pd.read_csv(omics_data_dir/'CCLE_AID_miRNA_20180525.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
miRNA = miRNA.drop_duplicates()
miRNA = replace_ccl_name(data=miRNA, keep_id=[0], ccl_info=ccl_info)
miRNA.iloc[0, 0] = ''
miRNA.to_csv(x_data_dir/'cancer_miRNA_expression.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
miRNA = None



# Load protein expression data
rppa = pd.read_csv(omics_data_dir/'CCLE_AID_RPPA_20180123.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
rppa = rppa.drop_duplicates()
rppa = replace_ccl_name(data=rppa, keep_id=[0], ccl_info=ccl_info)
rppa.iloc[0, 0] = ''
rppa.to_csv(x_data_dir/'cancer_RPPA.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
rppa = None



# Load DNA methylation data
me = pd.read_csv(omics_data_dir/'CCLE_AID_RRBS_TSS_1kb_20180614.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
me = me.drop_duplicates()
me = replace_ccl_name(data=me, keep_id=[0, 1, 2, 3], ccl_info=ccl_info)
me.iloc[0, 0] = ''
me.to_csv(x_data_dir/'cancer_DNA_methylation.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
me = None



# Load mutation count data
mu_count = pd.read_csv(omics_data_dir/'Mutation_AID_count.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
mu_count = mu_count.drop_duplicates()
mu_count = replace_ccl_name(data=mu_count, keep_id=[0, 1, 2], ccl_info=ccl_info)
mu_count.iloc[0, 0] = ''
mu_count.to_csv(x_data_dir/'cancer_mutation_count.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
mu_count = None



# Load mutation data
mu = pd.read_parquet(omics_data_dir/'Mutation_AID_binary.parquet', engine='pyarrow')
id_data = np.array(range(14, mu.shape[1]))
id_data = id_data[np.where(np.isin(mu.columns[id_data], ccl_info.other_id))[0]]
mu = mu.iloc[:, np.concatenate((list(range(14)), np.sort(id_data)))]
id_data = np.array(range(14, mu.shape[1]))
ccl_name = [str(i) for i in mu.columns[id_data]]
ccl_info.index = ccl_info.other_id
new_ccl_name = ccl_info.loc[ccl_name, 'improve_sample_id'].values
mu.columns = [str(i) for i in mu.columns[:14]] + [str(i) for i in new_ccl_name]
mu.to_parquet(x_data_dir/'cancer_mutation.parquet', engine='pyarrow')
mu = None



# Load_long_format mutation data
mu_long = pd.read_csv(omics_data_dir/'Mutation_AID_long_format.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=0, index_col=None, low_memory=False)
id = np.sort(np.where(np.isin(mu_long.RRID, ccl_info.other_id))[0])
mu_long = mu_long.iloc[id, ]
ccl_info.index = ccl_info.other_id
mu_long.loc[:, 'RRID'] = ccl_info.loc[mu_long.RRID, 'improve_sample_id'].values
mu_long.to_csv(x_data_dir/'cancer_mutation_long_format.txt', header=True, index=False, sep='\t',
               line_terminator='\r\n')