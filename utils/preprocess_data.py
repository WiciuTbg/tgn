import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    _ = next(f)  # skip header
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])

      feat = np.array([float(x) for x in e[4:]], dtype=np.float32)

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      feat_l.append(feat)

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l, dtype=np.float32)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_df.i = df.i + upper_u

  new_df.u += 1
  new_df.i += 1
  new_df.idx += 1
  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  # prepend a zero row for padding index 0
  empty = np.zeros(feat.shape[1], dtype=feat.dtype)[np.newaxis, :]
  feat = np.vstack([empty, feat])

  new_df.to_csv(OUT_DF, index=False)
  np.save(OUT_FEAT, feat)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, default='wikipedia',
                    help='Dataset name (eg. wikipedia or reddit)')
parser.add_argument('--bipartite', action='store_true',
                    help='Whether the graph is bipartite')

args = parser.parse_args()
run(args.data, bipartite=args.bipartite)
