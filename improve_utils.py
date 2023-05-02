import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional


fdir = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# TODO
# Note!
# We need to decide how this utils file will be provided for each model.
# Meanwhile, place this .py file in the level as your data preprocessing script.
# For example:
# GraphDRP/
# |_______ preprocess.py
# |_______ improve_utils.py
# |
# | 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Globals
# ---------
# These are globals for all models
import types
improve_globals = types.SimpleNamespace()

# TODO:
# This is CANDLE_DATA_DIR (or something...).
# How this is going to be passed to the code?
# imp_globals.main_data_dir = fdir/"CSA_Data"  # TODO: rename
improve_globals.main_data_dir = fdir/"improve_data_dir"
# imp_globals.main_data_dir = fdir/"candle_data_dir"

# Dir names corresponding to the primary input/output blocks in the pipeline
# {}: input/output
# []: process
# train path:      {raw_data} --> [preprocess] --> {ml_data} --> [train] --> {models}
# inference path:  {ml_data, models} --> [inference] --> {infer}
improve_globals.raw_data_dir_name = "raw_data"  # benchmark data
improve_globals.ml_data_dir_name = "ml_data"    # preprocessed data for a specific ML model
improve_globals.models_dir_name = "models"      # output from model training
improve_globals.infer_dir_name = "infer"        # output from model inference (testing)

# Secondary dirs in raw_data
improve_globals.x_data_dir_name = "x_data"      # feature data
improve_globals.y_data_dir_name = "y_data"      # target data

# Column names in the raw data files
# imp_globals.canc_col_name = "CancID"
# imp_globals.drug_col_name = "DrugID"
improve_globals.canc_col_name = "improve_sample_id"  # column name that contains the cancer sample ids TODO: rename to sample_col_name
improve_globals.drug_col_name = "improve_chem_id"    # column name that contains the drug ids
improve_globals.source_col_name = "source"           # column name that contains source/study names (CCLE, GDSCv1, etc.)

# Response data file name
improve_globals.y_file_name = "response.txt"  # response data

# Cancer sample features file names
improve_globals.copy_number_fname = "cancer_copy_number.txt"  # cancer feature
improve_globals.discretized_copy_number_fname = "cancer_discretized_copy_number.txt"  # cancer feature
improve_globals.dna_methylation_fname = "cancer_DNA_methylation.txt"  # cancer feature
improve_globals.gene_expression_fname = "cancer_gene_expression.txt"  # cancer feature
# TODO: add the other omics types
# ...
# ...
# ...

# Drug features file names
improve_globals.smiles_file_name = "smiles.csv"  # drug feature
improve_globals.mordred_file_name = "mordred.parquet"  # drug feature
improve_globals.ecfp4_512bit_file_name = "ecfp4_512bit.csv"  # drug feature

# Globals derived from the ones defined above
improve_globals.raw_data_dir = improve_globals.main_data_dir/improve_globals.raw_data_dir_name # raw_data
improve_globals.x_data_dir   = improve_globals.raw_data_dir/improve_globals.x_data_dir_name    # x_data
improve_globals.y_data_dir   = improve_globals.raw_data_dir/improve_globals.y_data_dir_name    # y_data
improve_globals.models_dir   = improve_globals.raw_data_dir/improve_globals.models_dir_name    # models
improve_globals.infer_dir    = improve_globals.raw_data_dir/improve_globals.infer_dir_name     # infer

improve_globals.y_file_path = improve_globals.y_data_dir/improve_globals.y_file_name           # response.txt
improve_globals.copy_number_file_path = improve_globals.x_data_dir/improve_globals.copy_number_fname  # cancer_copy_number.txt
improve_globals.dna_methylation_file_path = improve_globals.x_data_dir/improve_globals.dna_methylation_fname  # cancer_DNA_methylation.txt
improve_globals.gene_expression_file_path = improve_globals.x_data_dir/improve_globals.gene_expression_fname  # cancer_gene_expression.txt
# TODO: add the other omics types
# -----------------------------------------------------------------------------


# def load_rsp_data(src_raw_data_dir: str, y_col_name: str="AUC", verbose: bool=True):
# def load_single_drug_response_data(src_raw_data_dir: str, y_col_name: str="AUC", verbose: bool=True):
def load_single_drug_response_data(
    # raw_data_dir: str,
    source: Union[str, List[str]], y_col_name: str="auc", sep="\t", verbose: bool=True) -> pd.DataFrame:
    """
    Returns datarame with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    sources.

    Args:
        source (str or list of str): DRP source name (str) of multiple sources (list of strings)
        y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)

    Returns:
        pd.Dataframe: dataframe that contains drug response values
    """
    df = pd.read_csv(improve_globals.y_file_path, sep=sep)
    if isinstance(source, str):
        source = [source]
    df = df[df[improve_globals.source_col_name].isin(source)]
    cols = [improve_globals.source_col_name,
            improve_globals.drug_col_name,
            improve_globals.canc_col_name,
            y_col_name]
    df = df[cols]  # [source, drug id, cancer id, response]
    df = df.reset_index(drop=True)
    if verbose:
        print(f"Response data: {df.shape}")
        print(df[[improve_globals.canc_col_name, improve_globals.drug_col_name]].nunique())
    return df



def set_col_names_in_multilevel_dataframe(
    df: pd.DataFrame,
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol") -> pd.DataFrame:
    """ Returns the input dataframe with the multi-level column names renamed as
    specified in gene_system_identifier.

    Args:
        df (pd.DataFrame): omics dataframe
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: the input dataframe with the specified multi-level column names
    """
    df = df.copy()

    level_map = {"Entrez": 0, "Gene_Symbol": 1, "Ensembl": 2}
    level_names = list(level_map.keys())
    n_levels = len(level_map.keys())
    
    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    # print(gene_system_identifier)
    # import pdb; pdb.set_trace()
    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            df.columns = df.columns.rename(["Entrez", "Gene_Symbol", "Ensembl"], level=[0, 1, 2])  # assign multi-level col names
        else:
            df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retian specific column level
    else:
        assert len(gene_system_identifier) <= n_levels, f"'gene_system_identifier' can't contain more than {n_levels} items."
        set_diff = list(set(gene_system_identifier).difference(set(level_names)))
        assert len(set_diff) == 0, f"Passed unknown gene identifiers: {set_diff}"
        kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
        print(list(kk.keys()))
        print(list(kk.values()))
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
        drop_levels = list(set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)

    return df


def load_copy_number_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep="\t", verbose: bool=True) -> pd.DataFrame:
    """
    Returns copy number data. Omics data files are multi-level tables with 3
    column types, each specifies gene names using a different gene identifier
    system (level 0: Entrez ID, level 1: Gene Symbol, level 2: Ensembl ID).

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: multi-level dataframe that contains copy number data
    """
    # level_map describes how the multi-level columns are organized (this is
    # usually different in different omics datasets)
    level_map = {"Entrez":0, "Gene_Symbol": 1, "Ensembl": 2}

    df = pd.read_csv(improve_globals.copy_number_file_path, sep=sep, index_col=0, header=[0,1,2])
    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, gene_system_identifier)
    # Test the func
    # d0 = set_col_names_in_multilevel_dataframe(df, "all")
    # d1 = set_col_names_in_multilevel_dataframe(df, "Ensembl")
    # d2 = set_col_names_in_multilevel_dataframe(df, ["Ensembl"])
    # d3 = set_col_names_in_multilevel_dataframe(df, ["Entrez", "Gene_Symbol", "Ensembl"])
    # d4 = set_col_names_in_multilevel_dataframe(df, ["Entrez", "Ensembl"])
    # d5 = set_col_names_in_multilevel_dataframe(df, ["Blah", "Ensembl"])
    if verbose:
        print(f"Copy number data: {df.shape}")
    return df



def load_gene_expression_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep="\t", verbose: bool=True) -> pd.DataFrame:
    """
    Returns gene expression data. Omics data files are multi-level tables with 3
    column types, each specifies gene names using a different gene identifier
    system (level 0: Entrez ID, level 1: Gene Symbol, level 2: Ensembl ID).
    """
    # level_map describes how the multi-level columns are organized (this is
    # usually different in different omics datasets)
    level_map = {"Entrez":0, "Gene_Symbol": 1, "Ensembl": 2}

    cn = pd.read_csv(improve_globals.copy_number_file_path, sep=sep, index_col=0, header=[0,1,2])
    ge = pd.read_csv(improve_globals.gene_expression_file_path, sep=sep, index_col=0, header=[0,1,2])


    df = pd.read_csv(improve_globals.gene_expression_file_path, sep=sep, index_col=0, header=[0,1,2])
    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, gene_system_identifier)
    if verbose:
        print(f"Gene expression data: {df.shape}")
    return df


def load_dna_methylation(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep="\t", verbose: bool=True) -> pd.DataFrame:
    """
    """
    pass
    return None


def get_common_samples(df1: pd.DataFrame, df2: pd.DataFrame, ref_col: str):
    """
    IMPROVE-specific func.
    df1, df2 : dataframes
    ref_col : the ref column to find the common values

    Returns:
        df1, df2

    Example:
        TODO
    """
    # Retain (canc, drug) response samples for which we have omic data
    # TODO: consider making this an IMPROVE func
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    # print(df1.shape)
    df1 = df1[ df1[imp_globals.canc_col_name].isin(common_ids) ]
    # print(df1.shape)
    # print(df2.shape)
    df2 = df2[ df2[imp_globals.canc_col_name].isin(common_ids) ]
    # print(df2.shape)
    return df1, df2


def load_smiles_data(src_raw_data_dir: str):
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    smi = read_df(src_raw_data_dir/imp_globals.x_data_dir_name/imp_globals.smiles_file_name)
    return smi


def get_subset_df(df: pd.DataFrame, ids: list):
    """ Get a subset of the input dataframe based on row ids."""
    df = df.loc[ids]
    return df







def get_data_splits(src_raw_data_dir: str, splitdir_name: str,
                    split_file_name: str, rsp_df: pd.DataFrame):
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    splitdir = src_raw_data_dir/splitdir_name
    if len(split_file_name) == 1 and split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)

    """
    # Method 1
    splitdir = Path(os.path.join(src_raw_data_dir))/"splits"
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
        outdir_name = "full"
    else:
        # Check if the split file exists and load
        ids = []
        split_id_str = []    # e.g. split_5
        split_type_str = []  # e.g. tr, vl, te
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                # Get the ids
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
                # Get the name
                fname_sep = fname.split("_")
                split_id_str.append("_".join([s for s in fname_sep[:2]]))
                split_type_str.append(fname_sep[2])
        assert len(set(split_id_str)) == 1, "Data splits must be from the same dataset source."
        split_id_str = list(set(split_id_str))[0]
        split_type_str = "_".join([x for x in split_type_str])
        outdir_name = f"{split_id_str}_{split_type_str}"
    ML_DATADIR = main_data_dir/"ml_data"
    root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    os.makedirs(root, exist_ok=True)
    """

    """
    # Method 2
    splitdir = src_raw_data_dir/args.splitdir_name
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
    """
    return ids










def read_df(fpath: str, sep: str=","):
    """
    IMPROVE-specific func.
    Load a dataframe. Supports csv and parquet files.
    sep : the sepator in the csv file
    """
    # TODO: this func might be available in candle
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep)
    return df


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs
