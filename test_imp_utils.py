import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional
from sklearn.preprocessing import StandardScaler

import improve_utils
from improve_utils import improve_globals as ig


fdir = Path(__file__).resolve().parent


# Settings
# y_col_name = "auc"
y_col_name = "auc1"
split = 0
source_data_name = "CCLE"


# ------------------------
# Load train response data
# ------------------------
# import pdb; pdb.set_trace()
print("\nLoad response train data ...")
rs_tr = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_train.txt",
    y_col_name=y_col_name, verbose=True)

# Load val response data
print("\nLoad response val data ...")
rs_vl = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_val.txt",
    y_col_name=y_col_name, verbose=True)

# Load test response data
print("\nLoad response test data ...")
rs_te = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_test.txt",
    y_col_name=y_col_name, verbose=True)


# ----------------------
# Load omic feature data
# ----------------------
# cv = improve_utils.load_copy_number_data(gene_system_identifier="Gene_Symbol")
print("\nLoad gene expression ...")
ge = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
assert len(set(rs_tr[ig.canc_col_name]).intersection(set(ge.index))) == rs_tr[ig.canc_col_name].nunique(), "Something is missing..."
# mt = improve_utils.load_dna_methylation_data(gene_system_identifier="TSS")
# mc = improve_utils.load_mutation_count_data(gene_system_identifier="Gene_Symbol")
# cn_d = improve_utils.load_discretized_copy_number_data(gene_system_identifier="Gene_Symbol")
# rppa = improve_utils.load_rppa_data(gene_system_identifier="Gene_Symbol")

# Gene selection (LINCS landmark genes)
# TODO: we'll need to figure out how lincs genes will be provided for models!
use_lincs = True
if use_lincs:
    with open(fdir/"landmark_genes") as f:
        genes = [str(line.rstrip()) for line in f]
    genes = list(set(genes).intersection(set(ge.columns)))
    cols = genes
    ge = ge[cols]


# ----------------------
# Load drug feature data
# ----------------------
# sm = improve_utils.load_smiles_data()  # SMILES
# assert len(set(rs_tr[ig.drug_col_name]).intersection(set(sm[ig.drug_col_name]))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."
print("\nLoad Mordred descriptors ...")
dd = improve_utils.load_mordred_descriptor_data()  # Mordred descriptors
assert len(set(rs_tr[ig.drug_col_name]).intersection(set(dd.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."
# fp = improve_utils.load_morgan_fingerprint_data()  # Morgan fingerprints
# assert len(set(rs_tr[ig.drug_col_name]).intersection(set(fp.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."


# ----------------------
# Scale features
# ----------------------
# TODO: we might need to save the scaler object (to scale test/infer data)
# Scale omic features
ge_scaler = StandardScaler()
ge_scaled = pd.DataFrame(ge_scaler.fit_transform(ge),
                         index=ge.index, columns=ge.columns, dtype=np.float32)
# Scale mordred features
dd_scaler = StandardScaler()
dd_scaled = pd.DataFrame(dd_scaler.fit_transform(dd),
                         index=dd.index, columns=dd.columns, dtype=np.float32)


# ------------------------------------
# Mege data and extract y, x, and meta
# ------------------------------------

def get_xym(rs, canc_fea, drug_fea):
    """
    Returns:
        pd.DataFrame: y response 
        pd.DataFrame: x combined omic and drug features
        pd.DataFrame: m meta data
    """
    canc_cols = canc_fea.columns.tolist()
    drug_cols = drug_fea.columns.tolist()
    fea_cols = canc_cols + drug_cols
    canc_fea = canc_fea.reset_index()
    drug_fea = drug_fea.reset_index()

    data = pd.merge(rs, canc_fea, on=ig.canc_col_name, how='inner')
    data = pd.merge(data, drug_fea, on=ig.drug_col_name, how='inner')
    
    meta = data.drop(columns=fea_cols)
    y_data = data[[y_col_name]]
    x_data = data[fea_cols]
    return y_data, x_data, meta

print("\nMerge response and features ...")
y_train, x_train, m_train = get_xym(rs_tr, ge_scaled, dd_scaled)
y_val,   x_val,   m_val   = get_xym(rs_vl, ge_scaled, dd_scaled)
y_test,  x_test,  m_test  = get_xym(rs_te, ge_scaled, dd_scaled)
# y_train, x_train, m_train = get_xym(rs_tr, ge, dd)
# y_val,   x_val,   m_val   = get_xym(rs_vl, ge, dd)
# y_test,  x_test,  m_test  = get_xym(rs_te, ge, dd)

print(f"Train. y: {y_train.shape}, x: {x_train.shape}, meta: {m_train.shape}")
print(f"Val.   y: {y_val.shape}, x: {x_val.shape}, meta: {m_val.shape}")
print(f"Test.  y: {y_test.shape}, x: {x_test.shape}, meta: {m_test.shape}")


# ------------------------------------
# Train model
# ------------------------------------

# print("\nAdd model predictions to dataframe and save")
# preds_df = rs_te.copy()
# print(preds_df.head())
# model_preds = rs_te[y_col_name] + np.random.normal(loc=0, scale=0.1, size=rs_te.shape[0])  # DL model predictions
# preds_df[y_col_name + ig.pred_col_name_suffix] = model_preds
# print(preds_df.head())
# import pdb; pdb.set_trace()
# outdir_preds = fdir/"model_preds" # TODO: we will determine later what should be the output dir for model predictions
# os.makedirs(outdir_preds, exist_ok=True)
# outpath = outdir_preds/"test_preds.csv"
# improve_utils.save_preds(preds_df, y_col_name, outpath)

pred_df = m_test.copy()
pred_col_name = y_col_name + ig.pred_col_name_suffix

train = True
if train:
    # Train LightGBM
    # try:
    #     import lightgbm as lgb
    # except:
    #     print("lightgbm not found")
    # ml_init_args = {'n_estimators': 1000,
    #                 'max_depth': -1,
    #                 'learning_rate': 0.01,
    #                 'num_leaves': 31,
    #                 'n_jobs': 8,
    #                 'random_state': None}
    # ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    # ml_fit_args['eval_set'] = (x_val, y_val)
    # model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
    # model.fit(x_train, y_train, **ml_fit_args)

    # y_pred = model.predict(x_test)
    # pred_df[pred_col_name] = y_pred
    # r2 = improve_utils.r_square(pred_df[y_col_name], pred_df[pred_col_name])
    # print(f"R-square (LightGBM) {r2}")
    
    # Train Random Forest (RF)
    print("\nTrain Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        n_jobs=8, random_state=0)
    rf_reg.fit(x_train.values, y_train.squeeze().values)

    # Predict with RF
    y_pred = rf_reg.predict(x_test.values)
    pred_df[pred_col_name] = y_pred

    # Save predictions
    outdir_preds = fdir/"model_preds" # TODO: we will determine later what should be the output dir for model preds
    os.makedirs(outdir_preds, exist_ok=True)
    outpath = outdir_preds/"test_preds.csv"
    improve_utils.save_preds(pred_df, y_col_name, outpath, round_decimals=4)

    r2 = improve_utils.r_square(pred_df[y_col_name], y_pred)
    print(f"R-square (random forest) {np.round(r2, 5)}")


print("Finished.")
