import json
import os
from math import sqrt
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src.utils.constants import (
    CLARKE_WRIGHT_SAVINGS_NAME,
    GENETIC_ALGORITHM_NAME,
    GRASP_ALGORITHM_NAME,
    NN_ALGORITHM_NAME,
)
from src.utils.logger_config import logger

_BASE_PATH = os.path.join("..", "..", "results")

METHODS = [
    NN_ALGORITHM_NAME,
    GENETIC_ALGORITHM_NAME,
    CLARKE_WRIGHT_SAVINGS_NAME,
    GRASP_ALGORITHM_NAME,
]


class DataAnalyst:
    """
    Class for analyzing and visualizing results of VRP algorithms.
    """

    @staticmethod
    def _route_length(route: List[Dict[str, Any]]) -> float:
        """
        Calculate the total length of a route.

        :param route: List of dicts with 'x' and 'y' coordinates
        :return: Total route length
        """
        total = 0.0
        for a, b in zip(route, route[1:]):
            dx, dy = a["x"] - b["x"], a["y"] - b["y"]
            total += sqrt(dx * dx + dy * dy)
        return total

    @staticmethod
    def save_results(df: pd.DataFrame, file_name: str) -> None:
        """
        Save DataFrame with results to CSV file.

        :param df: DataFrame to save
        :param file_name: Output file name (without extension)
        """
        tables_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "results", "tables")
        )
        os.makedirs(tables_dir, exist_ok=True)
        output_path = os.path.join(tables_dir, f"comparison_results_{file_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to path: {output_path}")

    @staticmethod
    def create_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for each method.

        :param df: DataFrame with results
        :return: Summary DataFrame
        """
        summary = (
            df.groupby("method")
            .agg(
                avg_total_distance=("total_distance", "mean"),
                avg_cost=("cost", "mean"),
                avg_time_sec=("processing_time", "mean"),
                avg_distance_per_sec=("distance_per_second", "mean"),
                avg_cost_per_sec=("cost_per_second", "mean"),
                avg_coef_variation_length=("coef_variation_length", "mean"),
                avg_num_vehicles=("num_vehicles", "mean"),
                std_route_length_std=("std_route_length", "mean"),
            )
            .sort_values("avg_cost")
            .reset_index()
        )

        summary = summary.round(2)
        logger.info(f"Saved to path: {summary}")
        return summary

    def get_best_by_speed(self, summary: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Sort summary by average processing time.

        :param summary: Summary DataFrame
        :param name: Name for output file
        :return: Sorted DataFrame
        """
        df = summary.sort_values("avg_time_sec")
        self.save_results(df, f"{name}_avg_time_sec")
        return df

    def get_best_by_variation(self, summary: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Sort summary by coefficient of variation of route length.

        :param summary: Summary DataFrame
        :param name: Name for output file
        :return: Sorted DataFrame
        """
        df = summary.sort_values("avg_coef_variation_length")
        self.save_results(df, f"{name}_avg_coef_variation_length")
        return df

    def get_best_by_cost(self, summary: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Sort summary by average cost.

        :param summary: Summary DataFrame
        :param name: Name for output file
        :return: Sorted DataFrame
        """
        df = summary.sort_values("avg_cost")
        self.save_results(df, f"{name}_avg_cost")
        return df

    def create_result_table(self, set_name: str, result_name: str = "") -> pd.DataFrame:
        """
        Create a result table for a given dataset.

        :param set_name: Name of the dataset (e.g. 'small', 'medium')
        :param result_name: Specific result file name (optional)
        :return: DataFrame with results
        """
        records = []

        for method in METHODS:
            method_dir = os.path.join(_BASE_PATH, method, set_name)
            if not os.path.isdir(method_dir):
                continue

            files = (
                [f"{result_name}.json"]
                if result_name
                else [f for f in os.listdir(method_dir) if f.endswith(".json")]
            )

            for fname in files:
                path = os.path.join(method_dir, fname)
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    data = json.load(f)

                lengths = [
                    DataAnalyst._route_length(route)
                    for route in data["routes"].values()
                ]
                stops_counts = [
                    len(r) - 2 for r in data["routes"].values()
                ]  # -2: do not count depot at start/end

                records.append(
                    {
                        "instance": fname,
                        "method": method,
                        "total_distance": data["distance"],
                        "cost": data["cost"],
                        "processing_time": data["processing_time"],
                        "num_vehicles": len(lengths),
                        "mean_route_length": np.mean(lengths),
                        "std_route_length": np.std(lengths),
                        "min_route_length": np.min(lengths),
                        "max_route_length": np.max(lengths),
                        "mean_customers_per_route": np.mean(stops_counts),
                        "std_customers_per_route": np.std(stops_counts),
                        "min_customers_per_route": np.min(stops_counts),
                        "max_customers_per_route": np.max(stops_counts),
                        "coef_variation_length": (
                            np.std(lengths) / np.mean(lengths)
                            if np.mean(lengths)
                            else np.nan
                        ),
                        "distance_per_second": (
                            data["distance"] / data["processing_time"]
                            if data["processing_time"]
                            else np.nan
                        ),
                        "cost_per_second": (
                            data["cost"] / data["processing_time"]
                            if data["processing_time"]
                            else np.nan
                        ),
                    }
                )

        df = pd.DataFrame(records)
        df.sort_values(["instance", "method"], inplace=True)
        self.save_results(df, set_name)
        return df

    @staticmethod
    def plot_cost_vs_time_dual_axis(
        data: pd.DataFrame, name: Optional[str] = None, dpi: int = 300
    ) -> None:
        """
        Dual-axis bar/line plot for avg_cost and avg_time_sec per method.

        :param data: DataFrame with summary statistics
        :param name: Optional name for saving the plot
        :param dpi: Dots per inch for saved plot
        """
        required = {"method", "avg_cost", "avg_time_sec"}
        if not required.issubset(data.columns):
            raise ValueError(f"Data must include columns: {required}")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left Y-axis: avg_cost
        sns.barplot(
            data=data,
            x="method",
            y="avg_cost",
            ax=ax1,
            color="skyblue",
            label="Avg Cost",
        )
        ax1.set_ylabel("Average Cost", color="skyblue")
        ax1.tick_params(axis="y", labelcolor="skyblue")

        # Right Y-axis: avg_time_sec
        ax2 = ax1.twinx()
        sns.pointplot(
            data=data,
            x="method",
            y="avg_time_sec",
            ax=ax2,
            color="red",
            label="Avg Time",
            markers="o",
        )
        ax2.set_ylabel("Average Time (sec)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title("Average Cost vs Time per Method (Dual Axis)")
        fig.tight_layout()

        if name:
            path = f"plot_cost_vs_time_dual_axis{name}.png"
            tables_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "results", "charts")
            )
            os.makedirs(tables_dir, exist_ok=True)
            output_path = os.path.join(tables_dir, path)

            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Plot Cost Vs Time Dual Axis saved to: {path}")

        plt.show()

    @staticmethod
    def plot_normalized_heatmap(
        data: pd.DataFrame, name: Optional[str] = None, dpi: int = 300
    ) -> None:
        """
        Heatmap of normalized metrics for each VRP algorithm.

        :param data: DataFrame with summary statistics
        :param name: Optional name for saving the plot
        :param dpi: Dots per inch for saved plot
        """
        if "method" not in data.columns:
            raise ValueError("Data must contain 'method' column.")

        numeric_cols = data.select_dtypes(include="number").columns
        features = (
            numeric_cols.drop("method") if "method" in numeric_cols else numeric_cols
        )

        scaler = MinMaxScaler()
        norm_values = scaler.fit_transform(data[features])
        df_norm = pd.DataFrame(norm_values, columns=features, index=data["method"])

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_norm, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Normalized Metrics Heatmap")
        plt.tight_layout()

        if name:
            path = f"plot_normalized_heatmap_{name}.png"
            tables_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "results", "charts")
            )
            os.makedirs(tables_dir, exist_ok=True)
            output_path = os.path.join(tables_dir, path)

            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Heatmap saved to: {path}")
        plt.show()

    @staticmethod
    def plot_cost_vs_distance_scatter(
        data: pd.DataFrame, name: Optional[str] = None, dpi: int = 300
    ) -> None:
        """
        Scatter plot comparing total distance and cost per method.

        :param data: DataFrame with summary statistics
        :param name: Optional name for saving the plot
        :param dpi: Dots per inch for saved plot
        """
        required = {"method", "avg_cost", "avg_total_distance"}
        if not required.issubset(data.columns):
            raise ValueError(f"Data must include columns: {required}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=data, x="avg_total_distance", y="avg_cost", hue="method", s=100
        )

        for _, row in data.iterrows():
            plt.text(
                row["avg_total_distance"],
                row["avg_cost"],
                row["method"],
                fontsize=8,
                ha="right",
            )

        plt.title("Cost vs Total Distance")
        plt.xlabel("Avg Total Distance")
        plt.ylabel("Avg Cost")
        plt.tight_layout()

        if name:
            path = f"plot_cost_vs_distance_scatter_{name}.png"
            tables_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "results", "charts")
            )
            os.makedirs(tables_dir, exist_ok=True)
            output_path = os.path.join(tables_dir, path)

            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Scatter plot saved to: {path}")
        plt.show()