import glob
import os
import re
from src.algorithms.nearest_neighbor_algorithm.main import NearestNeighborVRP
from src.algorithms.utils.vrp_tools import VehicleInfo
from src.utils.constants import (
    NN_ALGORITHM_NAME,
ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
ABSOLUTE_INPUT_TEST_FOLDER_PATH,
ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH,
ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH,
ABSOLUTE_OUTPUT_TEST_FOLDER_PATH,
    JSON
)
from src.utils.logger_config import logger

_PATTERN = r"^[^.]+"

vehicle_info = VehicleInfo(avg_fuel_consumption=12, fuel_price=7)


def iterate_files(input_folder_path, output_folder_path, algorithm):
    os.chdir(input_folder_path)
    csv_files = sorted(glob.glob("*.csv"))
    json_files = sorted(glob.glob("*.json"))

    for csv_file, json_file in zip(csv_files, json_files):
        file_name = f"{re.search(_PATTERN, csv_file).group()}_results"
        output_path = rf"{output_folder_path}\{file_name}{JSON}"
        algorithm.solve(csv_file, json_file, output_path)


def run_nn_for_train():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
                  ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME), nn_algorithm)

def run_nn_for_validate():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
                  ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME), nn_algorithm)

def run_nn_for_test():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(ABSOLUTE_INPUT_TEST_FOLDER_PATH,
                  ABSOLUTE_OUTPUT_TEST_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME), nn_algorithm)



run_nn_for_train()
run_nn_for_validate()
run_nn_for_test()