import glob
import os
import re


from src.algorithms.grasp_algorithm import GRASPVRP
from src.algorithms.clarke_wright_savings_algorithm import ClarkeWrightSavingsVRP
from src.algorithms.genetic_algorithm import GeneticAlgorithmVRP
from src.algorithms.nearest_neighbor_algorithm import NearestNeighborVRP


from src.algorithms.tools.vrp_tools import VehicleInfo
from src.utils.constants import (
    ABSOLUTE_INPUT_TEST_FOLDER_PATH,
    ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
    ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
    ABSOLUTE_OUTPUT_TEST_FOLDER_PATH,
    ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH,
    ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH,
    CLARKE_WRIGHT_SAVINGS_NAME,
    GENETIC_ALGORITHM_NAME,
    JSON,
    NN_ALGORITHM_NAME,
    GRASP_ALGORITHM_NAME
)
from src.utils.logger_config import logger

_PATTERN = r"^[^.]+"

vehicle_info = VehicleInfo(avg_fuel_consumption=12, fuel_price=7)


def iterate_files(input_folder_path, output_folder_path, algorithm):
    os.chdir(input_folder_path)
    csv_files = sorted(glob.glob("*.csv"))
    json_files = sorted(glob.glob("*.json"))
    file_counter = 1
    for csv_file, json_file in zip(csv_files, json_files):
        file_name = f"{re.search(_PATTERN, csv_file).group()}_results"
        output_path = rf"{output_folder_path}\{file_name}{JSON}"
        logger.info("file path %s, output path %s", file_name, output_path)
        algorithm.solve(csv_file, json_file, output_path)
        file_counter += 1

    logger.info("Created %s files", file_counter)




def run_grasp_for_train():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )


def run_grasp_for_validate():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )


def run_grasp_for_test():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TEST_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TEST_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )

def run_clarke_wright_savings_for_train():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_clarke_wright_savings_for_validate():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_clarke_wright_savings_for_test():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TEST_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TEST_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_generic_for_train():
    genetic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        genetic_algorithm,
    )


def run_generic_for_validate():
    generic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        generic_algorithm,
    )


def run_generic_for_test():
    generic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TEST_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TEST_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        generic_algorithm,
    )


# Nearest neighbor algorithm
def run_nn_for_train():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TRAIN_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )


def run_nn_for_validate():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )


def run_nn_for_test():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_TEST_FOLDER_PATH,
        ABSOLUTE_OUTPUT_TEST_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )



run_nn_for_train()
run_nn_for_validate()
run_nn_for_test()

run_generic_for_train()
run_generic_for_validate()
run_generic_for_test()

run_clarke_wright_savings_for_train()
run_clarke_wright_savings_for_validate()
run_clarke_wright_savings_for_test()

run_grasp_for_train()
run_grasp_for_validate()
run_grasp_for_test()