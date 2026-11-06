"""Generate the chaotic benchmark trajectories with dysts.

For each system, n_series trajectories (one per initial condition) are integrated
on the Lyapunov timescale, i.e. with --pts_per_period points per Lyapunov time,
so that every system covers the same number of Lyapunov times
(n_points / pts_per_period ~= 333 with the defaults).

Output per system, in --output_dir:
    {equation}.npy    shape (n_series, n_points, D)
    {equation}_t.npy  shape (n_series, n_points), physical time
Systems already generated are skipped.
"""
import argparse
import os

import dysts.flows
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

NAMES = ['AtmosphericRegime', 'BeerRNN', 'Bouali', 'BurkeShaw', 'CaTwoPlusQuasiperiodic',
         'ChenLee', 'Chua', 'Coullet', 'Dadras', 'DequanLi', 'Finance', 'GenesioTesi',
         'GuckenheimerHolmes', 'Hadley', 'Halvorsen', 'HindmarshRose', 'IsothermalChemical',
         'ItikBanksTumor', 'JerkCircuit', 'Laser', 'Lorenz', 'Lorenz84', 'LuChen', 'MooreSpiegel',
         'MultiChua', 'NoseHoover', 'PehlivanWei', 'Sakarya', 'SanUmSrisuchinwong',
         'SprottA', 'SprottB', 'SprottC', 'SprottD', 'SprottF', 'SprottG', 'SprottI',
         'SprottL', 'SprottM', 'SprottO', 'SprottP', 'SprottQ', 'SprottR', 'SprottS',
         'SprottTorus', 'Thomas', 'ThomasLabyrinth', "Torus", "Tsucs2", 'VallisElNino',
         'WangSun', 'WindmiReduced', 'YuWang', 'YuWang2', 'ZhouChen']

N_CANDIDATE_IC = 100  # initial conditions are sampled among the points of a reference trajectory
TOL = 1e-12


def generate_trajectories(equation_name, n_series, n_points, pts_per_period, n_jobs=20, seed=0):
    """Returns trajectories (n_series, n_points, D) and times (n_series, n_points)."""
    # Candidate initial conditions: points on a reference trajectory of the attractor
    eq = getattr(dysts.flows, equation_name)()
    candidates = eq.make_trajectory(N_CANDIDATE_IC, pts_per_period=10, atol=TOL, rtol=TOL)
    rng = np.random.default_rng(seed)
    ics = candidates[rng.choice(N_CANDIDATE_IC, size=n_series, replace=False)]

    def solve_one(ic):
        eq_local = getattr(dysts.flows, equation_name)()
        eq_local.ic = ic
        t, traj = eq_local.make_trajectory(n_points, timescale="Lyapunov", method="Radau", return_times=True,
                                           pts_per_period=pts_per_period, atol=TOL, rtol=TOL)
        return traj, t

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(solve_one)(ic) for ic in ics)
    trajectories = np.stack([traj for traj, _ in results])
    times = np.stack([t for _, t in results])
    return trajectories, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="chaotic_benchmark",
                        help="Folder where trajectories are saved (default: chaotic_benchmark)")
    parser.add_argument("--n_series", type=int, default=20, help="Initial conditions per system (default: 20)")
    parser.add_argument("--n_points", type=int, default=10000, help="Points per trajectory (default: 10000)")
    parser.add_argument("--pts_per_period", type=int, default=30, help="Points per Lyapunov time (default: 30)")
    parser.add_argument("--equations", nargs="+", default=NAMES, help="Systems to generate (default: all 54)")
    parser.add_argument("--n_jobs", type=int, default=20, help="Parallel integrations (default: 20)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {len(args.equations)} systems: {args.n_series} series of {args.n_points} points, "
          f"{args.n_points / args.pts_per_period:.1f} Lyapunov times each")

    failed = []
    for name in tqdm(args.equations):
        traj_path = os.path.join(args.output_dir, f"{name}.npy")
        t_path = os.path.join(args.output_dir, f"{name}_t.npy")
        if os.path.exists(traj_path) and os.path.exists(t_path):
            print(f"[{name}] already generated, skipping")
            continue
        try:
            traj, t = generate_trajectories(name, args.n_series, args.n_points, args.pts_per_period, args.n_jobs)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            failed.append(name)
            continue
        np.save(traj_path, traj)
        np.save(t_path, t)
        print(f"[{name}] saved shape={traj.shape}")

    print(f"\nDone. {len(args.equations) - len(failed)}/{len(args.equations)} succeeded.")
    if failed:
        print("Failed:", failed)
