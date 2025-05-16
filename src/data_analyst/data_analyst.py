import json
import os
import pandas as pd
import numpy as np
from math import sqrt
from src.utils.constants import NN_ALGORITHM_NAME, GENETIC_ALGORITHM_NAME, CLARKE_WRIGHT_SAVINGS_NAME, GRASP_ALGORITHM_NAME

_BASE_PATH = os.path.join("..", "..", "results")

METHODS = [
            NN_ALGORITHM_NAME,
            GENETIC_ALGORITHM_NAME,
            CLARKE_WRIGHT_SAVINGS_NAME,
            GRASP_ALGORITHM_NAME,
        ]


class DataAnalyst:
    @staticmethod
    def _route_length(route):
        total = 0.0
        for a, b in zip(route, route[1:]):
            dx, dy = a["x"] - b["x"], a["y"] - b["y"]
            total += sqrt(dx * dx + dy * dy)
        return total
    @staticmethod
    def compare_results(set_name, result_name=""):
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

                lengths = [DataAnalyst._route_length(route) for route in data["routes"].values()]
                stops_counts = [len(r) - 2 for r in
                                data["routes"].values()]  # -2 bo nie liczymy depozyciku na początku i końcu

                records.append({
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
                    "coef_variation_length": np.std(lengths) / np.mean(lengths) if np.mean(lengths) else np.nan,
                    "distance_per_second": data["distance"] / data["processing_time"] if data[
                        "processing_time"] else np.nan,
                    "cost_per_second": data["cost"] / data["processing_time"] if data["processing_time"] else np.nan,
                })

        df = pd.DataFrame(records)
        df.sort_values(["instance", "method"], inplace=True)

        output_path = os.path.join(os.path.dirname(__file__), "comparison_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Zapisano do pliku: {output_path}")
DataAnalyst.compare_results("train")