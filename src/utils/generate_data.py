import csv
import json
import logging
import random
from typing import List, Tuple

from src.utils.constants import (CSV, JSON, RELATIVE_INPUT_CSV_PATH,
                                 RELATIVE_OUTPUT_LARGE_PATH,
                                 RELATIVE_OUTPUT_MEDIUM_PATH,
                                 RELATIVE_OUTPUT_SMALL_PATH,
                                 RELATIVE_OUTPUT_XLARGE_PATH)
from src.utils.logger_config import logger


class FileGenerator:
    """
    Class for generating VRP problem instances as CSV and JSON files.
    """

    def __init__(self) -> None:
        self.output_small_path = RELATIVE_OUTPUT_SMALL_PATH
        self.output_medium_path = RELATIVE_OUTPUT_MEDIUM_PATH
        self.output_large_path = RELATIVE_OUTPUT_LARGE_PATH
        self.output_xlarge_path = RELATIVE_OUTPUT_XLARGE_PATH

        self.input_csv_path = RELATIVE_INPUT_CSV_PATH
        self.rows, self.header = self.get_info_from_csv()

    @staticmethod
    def generate_file_name(cities: int, vehicles: int, instance_number: int) -> str:
        """
        Generates a file name based on the number of cities, vehicles, and instance number.

        :param cities: Number of cities.
        :param vehicles: Number of vehicles.
        :param instance_number: Instance number.
        :return: Generated file name.
        """
        return f"vpr{cities}_{vehicles}_{instance_number}"

    @staticmethod
    def get_vehicles_number(
        num_cities: int,
        min_clients_per_vehicle: int = 20,
        max_clients_per_vehicle: int = 30,
    ) -> int:
        """
        Calculates the number of vehicles required based on the number of cities and clients per vehicle.

        :param num_cities: Total number of cities.
        :param min_clients_per_vehicle: Minimum number of clients per vehicle.
        :param max_clients_per_vehicle: Maximum number of clients per vehicle.
        :return: Calculated number of vehicles.
        """
        clients = num_cities - 1
        clients_per_vehicle = random.randint(
            min_clients_per_vehicle, max_clients_per_vehicle
        )
        return max(2, round(clients / clients_per_vehicle))

    @staticmethod
    def get_deport_id(selected_rows: List[List[str]]) -> int:
        """
        Selects a random depot ID from the selected rows.

        :param selected_rows: List of selected rows, where each row is a list of strings.
        :return: Depot ID selected randomly from the first column of the rows.
        """
        return int(random.choice(selected_rows)[0])

    def generate_parameters_json(
        self,
        output_path: str,
        vehicles: int,
        cities: int,
        depot_id: int,
        instance_number: int,
    ) -> None:
        """
        Generates a JSON file with parameters for the problem instance.

        :param output_path: Path to save the JSON file.
        :param vehicles: Number of vehicles.
        :param cities: Number of cities.
        :param depot_id: Depot ID.
        :param instance_number: Instance number.
        """
        data = {"cities": cities, "vehicles": vehicles, "depot_id": depot_id}
        final_output_path = (
            output_path.format(
                path=self.generate_file_name(cities, vehicles, instance_number)
            )
            + JSON
        )
        with open(final_output_path, mode="w") as json_file:
            json.dump(data, json_file, indent=4)

        logger.debug(f"Saved JSON file in: {final_output_path}")

    def generate_csv_data(
        self,
        header: List[str],
        selected_rows: List[List[str]],
        output_path: str,
        vehicles: int,
        cities: int,
        instance_number: int,
    ) -> None:
        """
        Generates a CSV file with selected rows.

        :param header: Header of the CSV file.
        :param selected_rows: Selected rows to write to the CSV file.
        :param output_path: Path to save the CSV file.
        :param vehicles: Number of vehicles.
        :param cities: Number of cities.
        :param instance_number: Instance number.
        """
        final_output_path = (
            output_path.format(
                path=self.generate_file_name(cities, vehicles, instance_number)
            )
            + CSV
        )
        with open(final_output_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(selected_rows)

        logger.debug(
            f"Saved {len(selected_rows)} random rows to file: {final_output_path}"
        )

    def get_info_from_csv(self) -> Tuple[List[List[str]], List[str]]:
        """
        Reads a CSV file and extracts its header and rows.

        :return: A tuple containing rows and the header.
        """
        with open(self.input_csv_path, mode="r") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            rows = list(reader)

        return rows, header

    def generate_single_file_pair(
        self,
        cities: int,
        vehicles: int,
        instance_number: int,
        output_path: str,
    ) -> None:
        """
        Generates a pair of CSV and JSON files for a single problem instance.

        :param cities: Number of cities.
        :param vehicles: Number of vehicles.
        :param instance_number: Instance number.
        :param output_path: Path to save the output files.
        """
        selected_rows = random.sample(self.rows, min(cities, len(self.rows)))
        self.generate_csv_data(
            self.header, selected_rows, output_path, vehicles, cities, instance_number
        )
        self.generate_parameters_json(
            output_path,
            vehicles,
            cities,
            self.get_deport_id(selected_rows),
            instance_number,
        )

    def generate_data(
        self,
        output_folder_path: str,
        datasets_sizes: List[int],
        instances_per_size: int,
    ) -> None:
        """
        Generates multiple problem instances as CSV and JSON files.

        :param output_folder_path: Path to the output folder where files will be saved.
        :param datasets_sizes: List of dataset sizes, representing the number of cities.
        :param instances_per_size: Number of instances to generate for each dataset size.
        """
        for cities in datasets_sizes:
            for instance_number in range(1, instances_per_size):
                vehicles = self.get_vehicles_number(cities)
                self.generate_single_file_pair(
                    cities, vehicles, instance_number, output_folder_path
                )
        logger.info("Finished generating files")

    def generate_small_data(
        self,
        datasets_sizes: List[int] = range(51, 251, 50),
        instances_per_size: int = 3,
    ) -> None:
        """
        Generates training data files.

        :param datasets_sizes: List of dataset sizes for training data.
        :param instances_per_size: Number of instances to generate for each dataset size.
        """
        self.generate_data(self.output_small_path, datasets_sizes, instances_per_size)

    def generate_medium_data(
        self,
        datasets_sizes: List[int] = range(301, 601, 100),
        instances_per_size: int = 3,
    ) -> None:
        """
        Generates validation data files.

        :param datasets_sizes: List of dataset sizes for validation data.
        :param instances_per_size: Number of instances to generate for each dataset size.
        """
        self.generate_data(self.output_medium_path, datasets_sizes, instances_per_size)

    def generate_large_data(
        self,
        datasets_sizes: List[int] = range(701, 1001, 100),
        instances_per_size: int = 3,
    ) -> None:
        """
        Generates test data files.

        :param datasets_sizes: List of dataset sizes for test data.
        :param instances_per_size: Number of instances to generate for each dataset size.
        """
        self.generate_data(self.output_large_path, datasets_sizes, instances_per_size)

    def generate_xlarge_data(
        self,
        datasets_sizes: List[int] = range(1101, 1501, 200),
        instances_per_size: int = 3,
    ) -> None:
        """
        Generates test data files.

        :param datasets_sizes: List of dataset sizes for test data.
        :param instances_per_size: Number of instances to generate for each dataset size.
        """
        self.generate_data(self.output_xlarge_path, datasets_sizes, instances_per_size)


File_generator = FileGenerator()
File_generator.generate_small_data()
File_generator.generate_medium_data()
File_generator.generate_large_data()
File_generator.generate_xlarge_data()
