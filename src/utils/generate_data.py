import csv
import json
import logging
import random
from typing import List, Tuple

from src.utils.constants import (
    CSV,
    JSON,
    RELATIVE_INPUT_CSV_PATH,
    RELATIVE_OUTPUT_TEST_PATH,
    RELATIVE_OUTPUT_TRAIN_PATH,
    RELATIVE_OUTPUT_VALIDATE_PATH,
)
from src.utils.logger_config import logger


class FileGenerator:
    def __init__(self):
        self.output_train_path = RELATIVE_OUTPUT_TRAIN_PATH
        self.output_validate_path = RELATIVE_OUTPUT_VALIDATE_PATH
        self.output_test_path = RELATIVE_OUTPUT_TEST_PATH
        self.input_csv_path = RELATIVE_INPUT_CSV_PATH
        self.rows, self.header = self.get_info_from_csv()

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
    def get_vehicles_number(
        num_cities: int,
        min_clients_per_vehicle: int = 20,
        max_clients_per_vehicle: int = 30,
    ) -> int:
        """Calculates the number of vehicles required based on the number of cities and clients per vehicle.

        Args:
            num_cities (int): Total number of cities.
            min_clients_per_vehicle (int, optional): Minimum number of clients per vehicle. Defaults to 20.
            max_clients_per_vehicle (int, optional): Maximum number of clients per vehicle. Defaults to 30.

        Returns:
            int: Calculated number of vehicles.
        """
        clients = num_cities - 1
        clients_per_vehicle = random.randint(
            min_clients_per_vehicle, max_clients_per_vehicle
        )
        return max(2, round(clients / clients_per_vehicle))

    @staticmethod
    def get_deport_id(selected_rows: List[List[str]]) -> int:
        """Selects a random depot ID from the selected rows.

        Args:
            selected_rows (List[List[str]]): List of selected rows, where each row is a list of strings.

        Returns:
            int: Depot ID selected randomly from the first column of the rows.
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
        """Generates a JSON file with parameters for the problem instance.

        Args:
            output_path (str): Path to save the JSON file.
            vehicles (int): Number of vehicles.
            cities (int): Number of cities.
            depot_id (int): Depot ID.
            instance_number (int): Instance number.
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
        """Generates a CSV file with selected rows.

        Args:
            header (List[str]): Header of the CSV file.
            selected_rows (List[List[str]]): Selected rows to write to the CSV file.
            output_path (str): Path to save the CSV file.
            vehicles (int): Number of vehicles.
            cities (int): Number of cities.
            instance_number (int): Instance number.
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
        """Reads a CSV file and extracts its header and rows.

        Returns:
            Tuple[List[List[str]], List[str]]: A tuple containing rows and the header.
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
        """Generates a pair of CSV and JSON files for a single problem instance.

        Args:
            cities (int): Number of cities.
            vehicles (int): Number of vehicles.
            instance_number (int): Instance number.
            output_path (str): Path to save the output files.
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
        """Generates multiple problem instances as CSV and JSON files.

        Args:
            output_folder_path (str): Path to the output folder where files will be saved.
            datasets_sizes (List[int]): List of dataset sizes, representing the number of cities.
            instances_per_size (int): Number of instances to generate for each dataset size.

        Returns:
            None
        """
        for cities in datasets_sizes:
            for instance_number in range(1, instances_per_size):
                vehicles = self.get_vehicles_number(cities)
                self.generate_single_file_pair(
                    cities, vehicles, instance_number, output_folder_path
                )
        logger.info("Finished generating files")

    def generate_train_data(
        self,
        datasets_sizes: List[int] = range(101, 401, 50),
        instances_per_size: int = 10,
    ) -> None:
        """Generates training data files.

        Args:
            datasets_sizes (List[int], optional): List of dataset sizes for training data. Defaults to range(101, 401, 50).
            instances_per_size (int, optional): Number of instances to generate for each dataset size. Defaults to 10.

        Returns:
            None
        """
        self.generate_data(self.output_train_path, datasets_sizes, instances_per_size)

    def generate_validate_data(
        self,
        datasets_sizes: List[int] = range(151, 401, 50),
        instances_per_size: int = 4,
    ) -> None:
        """Generates validation data files.

        Args:
            datasets_sizes (List[int], optional): List of dataset sizes for validation data. Defaults to range(151, 401, 50).
            instances_per_size (int, optional): Number of instances to generate for each dataset size. Defaults to 4.

        Returns:
            None
        """
        self.generate_data(
            self.output_validate_path, datasets_sizes, instances_per_size
        )

    def generate_test_data(
        self,
        datasets_sizes: List[int] = range(201, 501, 100),
        instances_per_size: int = 4,
    ) -> None:
        """Generates test data files.

        Args:
            datasets_sizes (List[int], optional): List of dataset sizes for test data. Defaults to range(201, 501, 100).
            instances_per_size (int, optional): Number of instances to generate for each dataset size. Defaults to 4.

        Returns:
            None
        """
        self.generate_data(self.output_test_path, datasets_sizes, instances_per_size)


File_generator = FileGenerator()
File_generator.generate_train_data()
File_generator.generate_test_data()
File_generator.generate_validate_data()
