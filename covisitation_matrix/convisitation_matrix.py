# script for calculating covisitation

from tqdm.auto import tqdm
import pandas as pd
import os
import polars as pl
import sys

version = sys.argv[1]

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"

df = pd.concat([
    pd.read_parquet(CFG.train_path), 
    pd.read_parquet(CFG.val_path),
], axis=0).reset_index(drop=True)
print(df.shape)

if version == "v12":
    th = df["ts"].max() - 60 * 60 * 24 * 1000 * 14
    df = df[df["ts"] > th].reset_index(drop=True)
    print(df.shape)

weights = {"clicks" : 1, "carts" : 3, "orders" : 5}

n_lookback = 2

topk = 100


df["weight"] = df["type"].map(weights)
df["chunk"] = df["session"]//10000


def count(df):
    
    ss = df["session"].to_list()
    aa = df["aid"].to_list()
    ww = df["weight"].to_list()
    
    s_ = ss[0]
    a_list = [aa[0]]
    recs1 = []
    for i in (range(1, len(ss))):
        s = ss[i]
        a = aa[i]
        w = ww[i]
        if s_ == s:
            recs1.append([a_list[-n_lookback:], a, w])
        else:
            a_list = []
        a_list = a_list + [a]
        s_ = s

    rec_df = pd.DataFrame(recs1)
    rec_df = pl.from_pandas(rec_df)
    agg = rec_df.explode("0").groupby(["0","1"]).sum()
    agg.columns = ["aid_key", "aid_future", "score"]
    return agg
aggs = []
for _, g in tqdm(df.groupby("chunk")):
    if len(g) < 2:
        continue
    agg = count(g)
    aggs.append(agg)
aggs = pl.concat(aggs)
count_df = aggs.groupby(["aid_key", "aid_future"]).sum()

count_info_list = [-1] * (count_df["aid_key"].max() + 1)
for aid_key, g in tqdm(count_df.groupby("aid_key"), total=len(count_df["aid_key"].unique())):
    g = g.sort(by="score", reverse=True)[:topk]
    count_info_list[aid_key] = [g["aid_future"].to_list(), g["score"].to_list()]
os.system("mkdir ../inputs/comatrix")
pd.to_pickle(count_info_list, f"../inputs/comatrix/count_dic_{version}_all.pkl")