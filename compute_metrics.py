"""Compute forecasting metrics for one predicted trajectory file.

Loads the ground-truth trajectories --trajectory ({equation}.npy, shape
(n_series, n_points, D)) and the time axes next to it ({equation}_t.npy, shape
(n_series, n_points)), drops the first --context_length points, and compares
the rest with the predictions in --predictions (e.g. a *_y_val_pred.npy file
written by main.py).

Results are saved to --output as a dict {metric_name: [one value per series]}.
"""
import argparse
import os

import numpy as np
import torch

from metrics import (
    estimate_kl_divergence,
    average_hellinger_distance,
    mae,
    mse,
    smape,
    geometrical_misalignment,
    max_lyapunov_exponent_rosenstein_multivariate,
)

KL_STAGES = [300, 1000, 4000, 7000]
LYAP_TIME = 30  # timesteps per Lyapunov time


def safe_rosenstein(traj, tau):
    """Largest Lyapunov exponent, or NaN when it cannot be estimated
    (near-periodic series or diverged predictions). tau is the physical time
    step, since each system is sampled on its own Lyapunov timescale."""
    try:
        return max_lyapunov_exponent_rosenstein_multivariate(traj, tau=tau)
    except (ValueError, OverflowError):
        return np.nan


def safe_hellinger(yt, yp):
    """Hellinger distance between power spectra, or NaN when the spectrum of a
    diverged prediction overflows (welch then returns inf/NaN and the
    normalization check in metrics.py fails)."""
    try:
        return average_hellinger_distance(yt, yp)
    except AssertionError:
        return np.nan


def compute_equation_metrics(y_true, y_pred, dt):
    """y_true, y_pred: (n_series, n_timesteps, D); dt: (n_series,) time step."""
    n_series, n_timesteps, _ = y_true.shape
    keys = ["kl", "h", "mse30", f"mse{n_timesteps}", "smape30", f"smape{n_timesteps}",
            "mae30", f"mae{n_timesteps}", "kl_unnormalized", "kl_hist",
            "lyap_rosenstein_true", "lyap_rosenstein_pred"]
    for stage in KL_STAGES:
        keys += [f"kl_{stage}", f"h_{stage}", f"kl_hist_{stage}"]
    res = {k: [] for k in keys}

    for i in range(n_series):
        yt_raw, yp_raw = y_true[i], y_pred[i]

        # Unnormalized metrics (D_stsp^hist depends on absolute magnitude)
        res["kl_hist"].append(geometrical_misalignment(torch.as_tensor(yp_raw), torch.as_tensor(yt_raw)))
        res["kl_unnormalized"].append(estimate_kl_divergence(yt_raw, yp_raw))

        # Standardize with the ground-truth statistics
        mean = yt_raw.mean(axis=0, keepdims=True)
        std = yt_raw.std(axis=0, keepdims=True) + 1e-8
        yt = (yt_raw - mean) / std
        yp = (yp_raw - mean) / std

        res["kl"].append(estimate_kl_divergence(yt, yp))
        res["h"].append(safe_hellinger(yt, yp))
        res["mse30"].append(mse(yt, yp, LYAP_TIME))
        res[f"mse{n_timesteps}"].append(mse(yt, yp, n_timesteps))
        res["smape30"].append(smape(yt, yp, lyap_time=LYAP_TIME))
        res[f"smape{n_timesteps}"].append(smape(yt, yp, lyap_time=n_timesteps))
        res["mae30"].append(mae(yt, yp, LYAP_TIME))
        res[f"mae{n_timesteps}"].append(mae(yt, yp, n_timesteps))
        res["lyap_rosenstein_true"].append(safe_rosenstein(yt, tau=dt[i]))
        res["lyap_rosenstein_pred"].append(safe_rosenstein(yp, tau=dt[i]))

        for stage in KL_STAGES:
            res[f"kl_{stage}"].append(estimate_kl_divergence(yt[:stage], yp[:stage]))
            res[f"h_{stage}"].append(safe_hellinger(yt[:stage], yp[:stage]))
            res[f"kl_hist_{stage}"].append(
                geometrical_misalignment(torch.as_tensor(yp_raw[:stage]), torch.as_tensor(yt_raw[:stage]))
            )

    res["h_stages"] = {stage: res[f"h_{stage}"] for stage in KL_STAGES}
    res["kl_hist_stages"] = {stage: res[f"kl_hist_{stage}"] for stage in KL_STAGES}
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", required=True,
                        help="Ground-truth {equation}.npy file ({equation}_t.npy must be in the same folder)")
    parser.add_argument("--predictions", required=True, help="Predicted trajectory .npy file")
    parser.add_argument("--output", required=True, help="Output .npy file for the metrics dict")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length C (default: 1024)")
    args = parser.parse_args()

    data = np.load(args.trajectory)
    times = np.load(args.trajectory[:-len(".npy")] + "_t.npy")
    y_true = data[:, args.context_length:, :]
    n_series, n_timesteps, n_dim = y_true.shape
    dt = np.diff(times[:, args.context_length:], axis=1).mean(axis=1)

    y_pred = np.load(args.predictions)
    if y_pred.size < n_series * n_timesteps * n_dim:
        raise ValueError(f"{args.predictions} has shape {y_pred.shape}, "
                         f"expected at least ({n_series * n_timesteps}, {n_dim})")
    y_pred = y_pred.reshape(n_series, -1, n_dim)[:, :n_timesteps, :]

    results = compute_equation_metrics(y_true, y_pred, dt)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, results)
    print(f"Saved metrics to {args.output}")
