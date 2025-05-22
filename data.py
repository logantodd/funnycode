

Extra CLI options let you rename columns if your schema differs.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect trains whose reported GPS position is frozen for too long"
    )
    p.add_argument("--input", required=True, help="Path to the input CSV file")
    p.add_argument("--output_rows", default="stopped_rows.csv",
                   help="CSV to write detailed stopped rows")
    p.add_argument("--output_summary", default="stopped_summary.csv",
                   help="CSV to write one‑line summaries per stop location")

    p.add_argument("--window", type=float, default=60,
                   help="Window length in minutes (default: 60)")
    p.add_argument("--threshold", type=int, default=3,
                   help="Minimum identical reports within window (default: 3)")
    p.add_argument("--decimals", type=int, default=5,
                   help="Coordinate rounding decimals (default: 5)")

    # Column names are configurable so the script can adapt to any schema
    p.add_argument("--company_col", default="company")
    p.add_argument("--train_col", default="train_number")
    p.add_argument("--route_col", default="route_number",
                   help="Name of route column, blank if none")
    p.add_argument("--time_col", default="time")
    p.add_argument("--lon_col", default="longitude")
    p.add_argument("--lat_col", default="latitude")
    return p.parse_args()


def read_data(path: str | Path, cols: List[str], time_col: str) -> pd.DataFrame:
    # Memory‑friendly dtypes – categorical strings and float32 coords.
    dtype = {c: "category" for c in cols if c not in (time_col,)}
    dtype.update({cols[-2]: "float32", cols[-1]: "float32"})  # lon / lat assumed last two
    return pd.read_csv(path, usecols=cols, dtype=dtype, parse_dates=[time_col])


def find_stopped(
    df: pd.DataFrame,
    id_cols: List[str],
    time_col: str,
    window_minutes: float,
    threshold: int,
    decimals: int,
):
    # Add rounded coordinate columns once – keeps originals intact for output
    df["lat_r"] = df["latitude"].round(decimals)
    df["lon_r"] = df["longitude"].round(decimals)

    # Build final grouping list (skip None cols)
    grp_cols = [c for c in id_cols if c and c in df.columns] + ["lat_r", "lon_r"]

    # Deterministic ordering helps reproducibility
    df = df.sort_values(grp_cols + [time_col]).copy()

    # Each unique (train @ rounded‑location) group gets an int id
    df["group_id"] = df.groupby(grp_cols, sort=False).ngroup()

    sequences: List[pd.DataFrame] = []
    for _, g in df.groupby("group_id", sort=False):
        # Two‑pointer scan over NumPy datetime64 array (fast)
        t = g[time_col].to_numpy()
        start = 0
        for end in range(len(t)):
            # Move start until window size <= user limit
            while (t[end] - t[start]).astype('timedelta64[m]').astype(int) > window_minutes:
                start += 1
            if end - start + 1 >= threshold:
                sequences.append(g.iloc[start:end + 1])

    if not sequences:
        return pd.DataFrame(columns=df.columns), pd.DataFrame()

    stopped_rows = pd.concat(sequences).sort_values(time_col)

    summary = (
        stopped_rows
        .groupby(grp_cols, as_index=False)
        .agg(
            first_time=(time_col, "min"),
            last_time=(time_col, "max"),
            reports=(time_col, "size"),
        )
        .rename(columns={"lat_r": "latitude", "lon_r": "longitude"})
    )
    return stopped_rows, summary


def main():
    args = parse_args()

    # Column list order: company, train, route(optional), time, lon, lat
    cols: List[str] = [args.company_col, args.train_col]
    if args.route_col:
        cols.append(args.route_col)
    cols += [args.time_col, args.lon_col, args.lat_col]

    df = read_data(args.input, cols, args.time_col)

    id_cols = [args.company_col, args.train_col]
    if args.route_col and args.route_col in df.columns:
        id_cols.append(args.route_col)

    stopped_rows, summary = find_stopped(
        df,
        id_cols=id_cols,
        time_col=args.time_col,
        window_minutes=args.window,
        threshold=args.threshold,
        decimals=args.decimals,
    )

    stopped_rows.to_csv(args.output_rows, index=False)
    summary.to_csv(args.output_summary, index=False)

    print(
        f"Wrote {len(stopped_rows):,} stopped records → {args.output_rows}\n"
        f"Wrote {len(summary):,} location summaries → {args.output_summary}"
    )


if __name__ == "__main__":
    main()