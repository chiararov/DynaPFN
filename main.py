"""Run DynaPFN on the chaotic benchmark.

For each equation, loads {equation}.npy (shape (n_series, n_points, D)) and
{equation}_t.npy (shape (n_series, n_points)) from --trajectories_dir, uses the
first --context_length points as context and forecasts the rest autoregressively,
block_size timesteps at a time.

Outputs, per (equation, block size), with output_path = {output_dir}/{equation}_b{b}:
    {output_path}_y_val.npy       ground truth, shape (n_series * H, D)
    {output_path}_y_val_pred.npy  forecast,     shape (n_series * H, D)
    {output_path}_pred_vs_actual_dim{d}.png
"""
import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_DIR, "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.tabicl import TabICLRegressor
from utils import extend_features, make_df_per_dimension, plot_pred_and_actual_ts, split_by_dim, transpose_tsdf

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

NAMES = ['AtmosphericRegime', 'BeerRNN', 'Bouali', 'BurkeShaw', 'CaTwoPlusQuasiperiodic',
         'ChenLee', 'Chua', 'Coullet', 'Dadras', 'DequanLi', 'Finance', 'GenesioTesi',
         'GuckenheimerHolmes', 'Hadley', 'Halvorsen', 'HindmarshRose', 'IsothermalChemical',
         'ItikBanksTumor', 'JerkCircuit', 'Laser', 'Lorenz', 'Lorenz84', 'LuChen', 'MooreSpiegel',
         'MultiChua', 'NoseHoover', 'PehlivanWei', 'Sakarya', 'SanUmSrisuchinwong',
         'SprottA', 'SprottB', 'SprottC', 'SprottD', 'SprottF', 'SprottG', 'SprottI',
         'SprottL', 'SprottM', 'SprottO', 'SprottP', 'SprottQ', 'SprottR', 'SprottS',
         'SprottTorus', 'Thomas', 'ThomasLabyrinth', "Torus", "Tsucs2", 'VallisElNino',
         'WangSun', 'WindmiReduced', 'YuWang', 'YuWang2', 'ZhouChen']

MODELS = ["TabPFN", "TabICL", "TabICL-modified", "CatBoost"]
TABPFN_CKPT = os.path.join(REPO_DIR, "src", "tabpfn", "tabpfn-v2-regressor.ckpt")
TABICL_CKPT = os.path.join(REPO_DIR, "src", "tabicl", "tabicl-regressor-v2-20260212.ckpt")


def make_regressor(model_name):
    if model_name == "TabPFN":
        from src.tabpfn import TabPFNRegressor
        reg = TabPFNRegressor(ignore_pretraining_limits=True, model_path=TABPFN_CKPT)
        reg.n_estimators = 1
    elif model_name == "TabICL":
        reg = TabICLRegressor(model_path=TABICL_CKPT)
    elif model_name == "TabICL-modified":
        reg = TabICLRegressor(n_estimators=1,
                              norm_methods=None,
                              feat_shuffle_method=None,
                              outlier_threshold=np.inf,
                              model_path=TABICL_CKPT)
    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor
        # item_id is the only non-numeric column, so declare it categorical.
        reg = CatBoostRegressor(cat_features=["item_id"], verbose=False, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return reg


def lagged_features(x, cols_to_shift, block_size, num_dim, atf, polynomial_degree):
    """Feature library G of the last block_size timesteps of x (long format),
    repeated once per dimension."""
    x = x.pivot(index=['item_id', 'timestamp'], columns='dim', values='target').reset_index().drop(columns='item_id')
    F = extend_features(x, atf, polynomial_degree)[cols_to_shift].iloc[-block_size:]
    return F.loc[F.index.repeat(num_dim)].reset_index(drop=True)


def dynapfn(data, model_name, output_path, block_size=5, true_features=False,
            atf=False, polynomial_degree=4, refit=True, use_timestamp=True):
    """Forecast and save the results as {output_path}_y_val.npy, {output_path}_y_val_pred.npy
    and {output_path}_pred_vs_actual_{dim}.png."""
    tsdfs_forecast, tsdfs_train = data
    out_prefix = output_path
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    if os.path.exists(f"{out_prefix}_y_val.npy") and os.path.exists(f"{out_prefix}_y_val_pred.npy"):
        print(f"Skipping {out_prefix}: output already exists.", flush=True)
        return

    cols_fixed = ['item_id', 'timestamp', 'dim', 'target']
    drop_cols = ["target"] if use_timestamp else ["target", "timestamp"]

    Pred = []
    item_ids = tsdfs_train.index.get_level_values('item_id').unique()
    for item_id in item_ids:
        series_ckpt_path = f"{out_prefix}_series_{item_id}.pkl"
        if os.path.exists(series_ckpt_path):
            print(f"Skipping {out_prefix} item_id {item_id}: checkpoint already exists.", flush=True)
            Pred.append(pd.read_pickle(series_ckpt_path))
            continue

        X_train = tsdfs_train.loc[item_id]
        X_forecast = tsdfs_forecast.loc[item_id]

        # Feature library G on the context
        X_flat = X_train.copy()
        X_flat.columns = [str(c0) for c0, _ in X_flat.columns]
        dim_names = X_flat.columns.tolist()
        num_dim = len(dim_names)
        X_extended = extend_features(X_flat.reset_index(), atf, polynomial_degree)

        X_train = transpose_tsdf(X_train, item_id).reset_index()
        X_train = X_train.merge(X_extended, on="timestamp", how="left")
        cols_to_shift = [c for c in X_train.columns if c not in cols_fixed]

        # Lagged pairs: features at time i, target at time i + block_size
        ld = num_dim * block_size
        F_t_minus_1 = X_train[cols_to_shift].iloc[-ld:].reset_index(drop=True)
        X_train_shifted = X_train.copy()
        X_train_shifted[cols_to_shift] = X_train_shifted[cols_to_shift].shift(ld)

        X_forecast.columns = [str(c0) for c0, _ in X_forecast.columns]
        true_y_test = transpose_tsdf(X_forecast, item_id).reset_index()

        T = -(-len(true_y_test) // ld)  # number of blocks, rounded up
        regressors = {dim_name: make_regressor(model_name) for dim_name in dim_names}

        for t in tqdm(range(T), desc=f"Forecasting for item_id {item_id}"):
            x_t = true_y_test.loc[t*ld:(t+1)*ld-1, cols_fixed].reset_index(drop=True)
            x_t["target"] = np.nan
            new_line = pd.concat([x_t, F_t_minus_1[:len(x_t)]], axis=1)

            X_train_per_dim = split_by_dim(X_train_shifted, dim_names)
            for dim_name in dim_names:
                X_tr = X_train_per_dim[dim_name]
                new_line_dim = new_line[new_line["dim"] == dim_name].drop(columns=["dim"]).reset_index(drop=True)

                reg = regressors[dim_name]
                if t == 0 or refit:
                    reg.fit(X_tr.drop(columns=drop_cols), X_tr["target"])
                X_test = new_line_dim.drop(columns=drop_cols)
                if model_name == "TabPFN":
                    pred_dim = reg.predict(X_test, output_type="main")["mean"]
                else:
                    pred_dim = reg.predict(X_test)

                new_line.loc[new_line["dim"] == dim_name, "target"] = pred_dim
                x_t.loc[x_t["dim"] == dim_name, "target"] = pred_dim

            X_train_shifted = pd.concat([X_train_shifted, new_line], ignore_index=True)

            # Features for the next block: from the true values (true_features) or from the predictions
            x = true_y_test.loc[t*ld:(t+1)*ld-1, cols_fixed].reset_index(drop=True) if true_features else x_t
            F_t_minus_1 = lagged_features(x, cols_to_shift, block_size, num_dim, atf, polynomial_degree)

        result = X_train_shifted.loc[len(X_train):][cols_fixed]
        result = result.set_index(['item_id', 'timestamp', 'dim'])
        result.to_pickle(series_ckpt_path)
        Pred.append(result)

    Pred = pd.concat(Pred)

    for dim_name in set(Pred.index.get_level_values('dim')):
        plot_pred_and_actual_ts(train=tsdfs_train[dim_name],
                                test=tsdfs_forecast[dim_name],
                                pred=Pred.xs(dim_name, level='dim'),
                                save_path=f"{out_prefix}_pred_vs_actual_{dim_name}.png",
                                show_quantiles=False,
                                block_size=block_size)

    Pred = Pred['target'].reset_index().pivot_table(index=['item_id', 'timestamp'], columns='dim', values='target')
    Pred = Pred.reindex(index=tsdfs_forecast.index, columns=dim_names)

    np.save(f"{out_prefix}_y_val.npy", tsdfs_forecast.to_numpy())
    np.save(f"{out_prefix}_y_val_pred.npy", Pred.to_numpy())

    for item_id in item_ids:
        series_ckpt_path = f"{out_prefix}_series_{item_id}.pkl"
        if os.path.exists(series_ckpt_path):
            os.remove(series_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories_dir", required=True,
                        help="Folder with the {equation}.npy and {equation}_t.npy trajectory files")
    parser.add_argument("--output_dir", required=True, help="Folder where predictions are saved")
    parser.add_argument("--model", default="TabPFN", choices=MODELS, help="Backbone (default: TabPFN)")
    parser.add_argument("--b", type=int, nargs="+", default=[5], help="One or more block sizes b (default: 5)")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length C (default: 1024)")
    parser.add_argument("--equations", nargs="+", default=NAMES, help="Systems to run (default: all 54)")
    parser.add_argument("--polynomial_degree", type=int, default=4, help="Polynomial degree of the feature library (default: 4)")
    parser.add_argument("--refit", action="store_true", help="Refit the model at each forecast step")
    parser.add_argument("--true_features", action="store_true", help="Compute the features from the true values (true-features experiment)")
    parser.add_argument("--no_timestamp", action="store_true", help="Drop the timestamp input feature (default: kept)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    C = args.context_length

    for equation_name in args.equations:
        print(f"Running {equation_name} with {args.model}", flush=True)
        data = np.load(os.path.join(args.trajectories_dir, f"{equation_name}.npy"))
        times = np.load(os.path.join(args.trajectories_dir, f"{equation_name}_t.npy"))

        item_ids = [f"series_{i}" for i in range(data.shape[0])]
        tsdfs_train = pd.concat(make_df_per_dimension(data[:, :C, :], item_ids, times[:, :C]), axis=1)
        tsdfs_forecast = pd.concat(make_df_per_dimension(data[:, C:, :], item_ids, times[:, C:]), axis=1)

        for block_size in args.b:
            print(f"  block size {block_size}", flush=True)
            dynapfn(data=(tsdfs_forecast, tsdfs_train),
                    model_name=args.model,
                    output_path=os.path.join(args.output_dir, f"{equation_name}_b{block_size}"),
                    block_size=block_size,
                    true_features=args.true_features,
                    refit=args.refit,
                    polynomial_degree=args.polynomial_degree,
                    use_timestamp=not args.no_timestamp)
