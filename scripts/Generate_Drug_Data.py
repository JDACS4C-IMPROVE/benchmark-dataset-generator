import pandas as pd
import numpy as np
from pathlib import Path  # pathlib introduced in Python 3.4 (standard, intuitive, powerful as compared to os.path)


fdir = Path(__file__).resolve().parent

pd.set_option('display.max_columns', None)

drug_data_dir = fdir/'../../Drug_Data'

benchmark_data_dir = fdir/'../csa_data'
x_data_dir = benchmark_data_dir/'x_data'



# Load drug fingerprint data
df = pd.read_parquet(drug_data_dir/'ecfp4_nbits512.parquet', engine='pyarrow')
df = df.drop_duplicates()
df.index = df.improve_chem_id
df = df.iloc[:, 1:]
print("ECFP4", df.shape)
print('There are ' + str(df.drop_duplicates().shape[0]) + ' unique fingerprint profiles')

# Load drug descriptor data
dd = pd.read_parquet(drug_data_dir/'mordred.parquet', engine='pyarrow')
dd = dd.drop_duplicates()
dd.index = dd.improve_chem_id
dd = dd.iloc[:, 1:]
print("Mordred", dd.shape)
print('There are ' + str(dd.drop_duplicates().shape[0]) + ' unique descriptor profiles')

# Load drug info (metadata)
drug_info = pd.read_csv(x_data_dir/'drug_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
                        header=0, index_col=None, low_memory=False)
# Note that not the original canSMILES were used to calculate ecfp and mordred!
ds = drug_info.loc[:, ['canSMILES', 'improve_chem_id']].drop_duplicates()
ds.index = ds.improve_chem_id  # TODO: consider renaming 'improve_chem_id' to 'imp_drug_id'
ds = ds.iloc[:, [0]]
id = np.argsort(ds.iloc[:, 0])  # Note! Same as ds.sort_values('canSMILES', ascending=True)
ds = ds.iloc[id, :]
print("canSMILES", ds.shape)
print('There are ' + str(len(np.unique(ds))) + ' unique canSMILES')

id = np.where(np.isin(df.index, ds.index))[0]
df = df.iloc[id, :]

id = np.where(np.isin(dd.index, ds.index))[0]
dd = dd.iloc[id, :]

ds.to_csv(x_data_dir/'drug_SMILES.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
df.to_csv(x_data_dir/'drug_ecfp4_nbits512.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
dd.to_csv(x_data_dir/'drug_mordred_descriptor.txt', header=True, index=True, sep='\t', line_terminator='\r\n')