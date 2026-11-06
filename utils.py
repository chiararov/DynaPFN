import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

def is_subset(tsdf_A, tsdf_B):
    return set(tsdf_A.index).issubset(set(tsdf_B.index))


def plot_pred_and_actual_ts(pred, train, test, item_ids=None,
                             show_quantiles=True, show_points=False,
                             save_path=None, block_size=None):

    if item_ids is None:
        item_ids = train.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(train.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    if pred.shape[0] != test.shape[0]:
        if not is_subset(pred, test):
            raise ValueError(
                "Pred and Test have different number of items and "
                "Pred is not a subset of Test"
            )
        filled_pred = test.copy()
        filled_pred["target"] = np.nan
        for col in pred.columns:
            filled_pred.loc[pred.index, col] = pred[col]
        pred = filled_pred

    assert pred.shape[0] == test.shape[0]

    fig, ax = plt.subplots(len(item_ids), 1, figsize=(10, 3 * len(item_ids)))
    ax = [ax] if not isinstance(ax, np.ndarray) else ax

    def plot_single_item(ax, item_id):
        pred_item  = pred.xs(item_id,  level="item_id")
        train_item = train.xs(item_id, level="item_id")
        test_item  = test.xs(item_id,  level="item_id")

        ground_truth = (
            test_item[["target"]]
            if is_subset(train_item, test_item)
            else pd.concat([train_item[["target"]], test_item[["target"]]])
        )

        ax.plot(ground_truth.index, ground_truth["target"],
                label="Ground Truth", color="tab:blue")
        ax.plot(pred_item.index, pred_item["target"],
                label="Prediction", color="tab:orange")

        if show_points:
            ax.scatter(ground_truth.index, ground_truth,
                       color="lightblue", s=8, alpha=0.8)

        if show_quantiles:
            quantile_cols = sorted(
                pred_item.columns.drop(["target"]).tolist(),
                key=lambda x: float(x)
            )
            ax.fill_between(
                pred_item.index,
                pred_item[quantile_cols[0]],
                pred_item[quantile_cols[-1]],
                color="gray", alpha=0.2,
                label=f"{quantile_cols[0]}-{quantile_cols[-1]} Quantile Range",
            )

        split_x = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(x=split_x, color="r", linestyle="--", label="Train/Test Split")

        if block_size is not None:
            forecast_timestamps = pred_item.index.tolist()
            for k in range(block_size, len(forecast_timestamps), block_size):
                ax.axvline(
                    x=forecast_timestamps[k],
                    color="black",
                    linestyle=":",
                    alpha=0.4,
                    label="Block boundary" if k == block_size else None,
                )

        ax.set_title(f"Item ID: {item_id}")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

    for i, item_id in enumerate(item_ids):
        plot_single_item(ax[i], item_id)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def transpose_tsdf(tsdf,item_id):
        tsdf.columns = tsdf.columns.get_level_values(0)
        df_reset = tsdf.reset_index()
        if "DateTime" in tsdf.index.names:
            df_long = df_reset.melt(id_vars=['timestamp', 'DateTime'], var_name='dim', value_name='target')
            df_long['item_id'] = item_id
            df_indexed = df_long.set_index(['item_id', 'timestamp', 'DateTime', 'dim']).sort_index()
        else:
            df_long = df_reset.melt(id_vars='timestamp', var_name='dim', value_name='target')
            df_long['item_id'] = item_id
            df_indexed = df_long.set_index(['item_id', 'timestamp', 'dim']).sort_index()
        
        return df_indexed


def extend_features(df, add_time_features=False,polynomial_degree=4):
    df = df.copy()
    feature_dict = {}

    dim_cols = [c for c in df.columns if c.startswith('dim')]
    ndim = len(dim_cols)

    # -------------------------
    # Base nonlinear features
    # -------------------------
    if polynomial_degree >=1:
        for i in range(ndim):
            d = df[f'dim{i}'].values
            feature_dict[f'cos_{i}'] = np.cos(d)
            feature_dict[f'sin_{i}'] = np.sin(d)

    for degree in range(2, polynomial_degree + 1):
        for combo in combinations_with_replacement(range(ndim), degree):
            name = 'poly_' + ''.join(map(str, combo))
            result = np.ones(len(df))
            for idx in combo:
                result *= df[f'dim{idx}'].values
            feature_dict[name] = result

    # -------------------------
    # Time features
    # -------------------------
    if add_time_features:
        if 'DateTime' in df.columns:
            ts = pd.to_datetime(df['DateTime'])
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            raise ValueError("No datetime found")

        feature_dict['hour'] = ts.dt.hour
        feature_dict['dayofweek'] = ts.dt.dayofweek
        feature_dict['dayofmonth'] = ts.dt.day
        feature_dict['month'] = ts.dt.month
        feature_dict["dayofyear"] = ts.dt.dayofyear
        feature_dict['weekofyear'] = ts.dt.isocalendar().week.astype(int)
        feature_dict['year'] = ts.dt.year

        feature_dict['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
        feature_dict['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)

        feature_dict['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        feature_dict['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        feature_dict['month_sin'] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12)
        feature_dict['month_cos'] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12)

    df_ext = pd.DataFrame(feature_dict, index=df.index)

    return pd.concat([df, df_ext], axis=1)


def make_df_per_dimension(trajectories, item_ids, times):
    N, T, D = trajectories.shape
    dfs_by_dim = {}

    for d in range(D):
        dfs = []
        for i in range(N):
            df = pd.DataFrame({
                "target": trajectories[i, :, d],
                "timestamp": times[i],
                "item_id": item_ids[i],
            })
            df.set_index(["item_id", "timestamp"], inplace=True)
            dfs.append(df)

        dfs_by_dim[f"dim{d}"] = pd.concat(dfs)

    return dfs_by_dim


def split_by_dim(df, dim_names):
    return {
        dim_name: df[df["dim"] == dim_name].drop(columns=["dim"]).reset_index(drop=True)
        for dim_name in dim_names
    }
