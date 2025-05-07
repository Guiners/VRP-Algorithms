import csv
import json
import logging
import random
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()



class FileGenerator():
    def __init__(self):
        self.output_json_train_path = "../../datasets/train/{path}"
        self.output_json_validate_path = "../../datasets/validate/{path}"
        self.output_json_test_path = "../../datasets/test/{path}"
        self.input_csv_path = "../../datasets/raw/cities_in_csv.csv"
        self.datasets_sizes = range(50, 1000, 50)

    @staticmethod
    def generate_file_name(cities: int, vehicles: int, instance_number: int) -> str:
        """Generates a file name based on the number of cities, vehicles, and instance number.

        Args:
            cities (int): Number of cities.
            vehicles (int): Number of vehicles.
            instance_number (int): Instance number.

        Returns:
            str: Generated file name.
        """
        return f"vpr{cities}_{vehicles}_{instance_number}"

    @staticmethod
    def get_vehicles_number(cities: int) -> int:
        """Calculates the number of vehicles based on the number of cities.

        Args:
            cities (int): Number of cities.

        Returns:
            int: Number of vehicles.
        """
        return max(1, cities // 20)


    def generate_parameters_json(self, output_path: str, vehicles: int, cities: int, depot_id: int, instance_number: int) -> None:
        """Generates a JSON file with parameters for the problem instance.

        Args:
            output_path (str): Path to save the JSON file.
            vehicles (int): Number of vehicles.
            cities (int): Number of cities.
            depot_id (int): Depot ID.
            instance_number (int): Instance number.
        """
        data = {
            "cities": cities,
            "vehicles": vehicles,
            "depot_id": depot_id
        }
        final_output_path = output_path.format(path=self.generate_file_name(cities, vehicles, instance_number)) + ".json"
        with open(final_output_path, mode="w") as json_file:
            json.dump(data, json_file, indent=4)

        logger.info(f"Saved JSON file in: {final_output_path}")


    def generate_csv_data(self, header: List[str], selected_rows: List[List[str]], output_path: str, vehicles: int, cities: int,
                          instance_number: int) -> None:
        """Generates a CSV file with selected rows.

        Args:
            header (List[str]): Header of the CSV file.
            selected_rows (List[List[str]]): Selected rows to write to the CSV file.
            output_path (str): Path to save the CSV file.
            vehicles (int): Number of vehicles.
            cities (int): Number of cities.
            instance_number (int): Instance number.
        """
        final_output_path = output_path.format(path=self.generate_file_name(cities, vehicles, instance_number)) + ".csv"
        with open(final_output_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(selected_rows)

        logger.info(f"Saved {len(selected_rows)} random rows to file: {final_output_path}")


    def get_info_from_csv(self) -> Tuple[List[List[str]], List[str]]:
        """Reads a CSV file and extracts its header and rows.

        Returns:
            Tuple[List[List[str]], List[str]]: A tuple containing rows and the header.
        """
        with open(self.input_csv_file_path, mode="r") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            rows = list(reader)

        return rows, header


    def generate_sigle_file_pair(self, cities: int, vehicles: int, instance_number: int, rows: List[List[str]], header: List[str],
                                 output_path: str, depot_id: int = 0) -> None:
        """Generates a pair of CSV and JSON files for a single problem instance.

        Args:
            cities (int): Number of cities.
            vehicles (int): Number of vehicles.
            instance_number (int): Instance number.
            rows (List[List[str]]): Rows from the input CSV file.
            header (List[str]): Header of the input CSV file.
            output_path (str): Path to save the output files.
            depot_id (int, optional): Depot ID. Defaults to 0.
        """
        selected_rows = random.sample(rows, min(cities, len(rows)))
        self.generate_csv_data(header, selected_rows, output_path, vehicles, cities, instance_number)
        self.generate_parameters_json(output_path, vehicles, cities, depot_id, instance_number)


    def generate_data(self, input_csv_file_path: str, output_folder_path: str) -> None:
        """Generates multiple problem instances as CSV and JSON files.

        Args:
            input_csv_file_path (str): Path to the input CSV file.
            output_folder_path (str): Path to the output folder.
        """
        rows, header = self.get_info_from_csv()
        for cities in self.datasets_sizes:
            for instance_number in range(1, 4):
                vehicles = self.get_vehicles_number(cities)
                self.generate_sigle_file_pair(cities, vehicles, instance_number, rows, header, output_folder_path)
        logger.info("Finished generating files")

print('abc')