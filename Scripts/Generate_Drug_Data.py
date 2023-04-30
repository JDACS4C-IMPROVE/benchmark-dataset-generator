import pandas as pd
import numpy as np



pd.set_option('display.max_columns', None)

drug_data_dir = '../../Drug_Data/'

benchmark_data_dir = '../CSA_Data/'



# Load drug fingerprint data
df = pd.read_parquet(drug_data_dir + 'ecfp4_nbits512.parquet', engine='pyarrow')
df = df.drop_duplicates()
df.index = df.improve_chem_id
df = df.iloc[:, 1:]

# Load drug descriptor data
dd = pd.read_parquet(drug_data_dir + 'mordred.parquet', engine='pyarrow')
dd = dd.drop_duplicates()
dd.index = dd.improve_chem_id
dd = dd.iloc[:, 1:]

# Load dug info
drug_info = pd.read_csv(benchmark_data_dir + 'drug_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
                        header=0, index_col=None, low_memory=False)

ds = drug_info.loc[:, ['canSMILES', 'improve_chem_id']].drop_duplicates()
ds.index = ds.improve_chem_id
ds = ds.iloc[:, [0]]
id = np.argsort(ds.iloc[:, 0])
ds = ds.iloc[id, :]


id = np.where(np.isin(df.index, ds.index))[0]
df = df.iloc[id, :]

id = np.where(np.isin(dd.index, ds.index))[0]
dd = dd.iloc[id, :]

ds.to_csv(benchmark_data_dir + 'drug_SMILES.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
df.to_csv(benchmark_data_dir + 'drug_fingerprint.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
dd.to_csv(benchmark_data_dir + 'drug_descriptor.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
