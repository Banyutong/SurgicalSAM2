"""Generate a Markdown report with the comprehensive performance table from W&B runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

DEFAULT_CSV = Path(__file__).with_name("wandb_export_2025-09-18T12_21_56.746+08_00.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("performance_table_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown table covering every trained configuration.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the W&B CSV export (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Markdown file (default: %(default)s)",
    )
    return parser.parse_args()


def load_runs(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading runs from {}", csv_path)
    df = pd.read_csv(csv_path)
    df["trainer.max_epochs"] = pd.to_numeric(df["trainer.max_epochs"], errors="coerce")
    return df


def prepare_combined_table(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building combined performance table")
    df = df.copy()
    df["dataset"] = df["data_module.data.name"]
    df["prompt_type"] = df["module.model.prompt_type"]
    df["config"] = df["Name"].apply(lambda x: x.split("_")[2])

    df_trained = df[df["trainer.max_epochs"] > 0].copy()
    df_baseline = df[df["trainer.max_epochs"] == 0].copy()

    df_trained["perf_str_trained"] = (
        df_trained["eval/Dice"].round(3).astype(str)
        + " / "
        + df_trained["eval/mIoU"].round(3).astype(str)
        + " / "
        + df_trained["eval/MAE"].round(2).astype(str)
    )

    df_baseline["config"] = df_baseline["config"].apply(lambda name: "mem" if "mem" in name else name)
    df_baseline["perf_str_baseline"] = (
        df_baseline["eval/Dice"].round(3).astype(str)
        + " / "
        + df_baseline["eval/mIoU"].round(3).astype(str)
        + " / "
        + df_baseline["eval/MAE"].round(2).astype(str)
    )

    join_cols = ["dataset", "prompt_type", "config"]
    df_trained.set_index(join_cols, inplace=True)
    df_baseline.set_index(join_cols, inplace=True)

    combined = df_trained.join(df_baseline[["perf_str_baseline"]], how="left")

    combined.reset_index(inplace=True)

    trained_table = combined.pivot_table(
        index="config",
        columns=["dataset", "prompt_type"],
        values="perf_str_trained",
        aggfunc="first",
    )

    row_order = [
        "md",
        "md+pe",
        "md+pe+ie",
        "mem",
        "mem+md",
        "mem+md+pe",
        "mem+md+pe+ie",
    ]

    existing_rows = [row for row in row_order if row in trained_table.index]
    trained_table = trained_table.reindex(existing_rows)

    if not trained_table.empty:

        def mark_best_dice(column: pd.Series) -> pd.Series:
            dice_scores = pd.to_numeric(column.str.split(" / ").str[0], errors="coerce")
            if dice_scores.notna().any():
                best_idx = dice_scores.idxmax()
                column.loc[best_idx] = f"{column.loc[best_idx]}*"
            return column

        trained_table = trained_table.apply(mark_best_dice, axis=0)

    baseline_candidates = df_baseline.reset_index()[
        ["dataset", "prompt_type", "config", "perf_str_baseline"]
    ]
    baseline_table = pd.DataFrame()
    if not baseline_candidates.empty:
        baseline_candidates["priority"] = (baseline_candidates["config"] != "mem").astype(int)
        baseline_candidates.sort_values(
            ["dataset", "prompt_type", "priority", "config"],
            inplace=True,
        )
        baseline_summary = baseline_candidates.drop_duplicates(
            ["dataset", "prompt_type"],
            keep="first",
        )
        baseline_summary["config"] = "baseline"
        baseline_table = baseline_summary.pivot_table(
            index="config",
            columns=["dataset", "prompt_type"],
            values="perf_str_baseline",
            aggfunc="first",
        )

    tables_to_concat: list[pd.DataFrame] = []
    if not baseline_table.empty:
        tables_to_concat.append(baseline_table)
    if not trained_table.empty:
        tables_to_concat.append(trained_table)

    if tables_to_concat:
        table = pd.concat(tables_to_concat)
        column_order: list[tuple[str, str]] = []
        for tbl in tables_to_concat:
            for col in tbl.columns:
                if col not in column_order:
                    column_order.append(col)
        if column_order:
            table = table.reindex(columns=pd.MultiIndex.from_tuples(column_order))
    else:
        table = pd.DataFrame()

    table.fillna("-", inplace=True)

    if not table.empty:
        flattened_columns = [f"{dataset} · {prompt}" for dataset, prompt in table.columns]
        table.columns = flattened_columns

    return table


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    return df.to_markdown()


def build_report(csv_path: Path, table: pd.DataFrame) -> str:
    lines: list[str] = [
        "# Comprehensive Performance Table",
        "",
        f"Source CSV: `{csv_path.name}`",
        "",
        "Each dataset/prompt column lists Dice / mIoU / MAE.",
        "The top row reports baseline metrics; subsequent rows contain trained configurations.",
        "An asterisk marks the highest trained Dice score within each dataset/prompt column.",
        "",
        df_to_markdown(table),
        "",
    ]
    return "\n".join(lines)


@logger.catch(onerror=lambda _: sys.exit(1))
def main() -> None:
    args = parse_args()
    df = load_runs(args.csv)
    table = prepare_combined_table(df)
    report = build_report(args.csv, table)

    args.output.write_text(report, encoding="utf-8")
    logger.info("Wrote report to {}", args.output)


if __name__ == "__main__":
    main()
