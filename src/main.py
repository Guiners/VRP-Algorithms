import glob
import os
import re


from src.algorithms.grasp_algorithm import GRASPVRP
from src.algorithms.clarke_wright_savings_algorithm import ClarkeWrightSavingsVRP
from src.algorithms.genetic_algorithm import GeneticAlgorithmVRP
from src.algorithms.nearest_neighbor_algorithm import NearestNeighborVRP


from src.algorithms.tools.vrp_tools import VehicleInfo
from src.utils.constants import (
    ABSOLUTE_INPUT_SMALL_FOLDER_PATH,
    ABSOLUTE_INPUT_MEDIUM_FOLDER_PATH,
    ABSOLUTE_INPUT_LARGE_FOLDER_PATH,
    ABSOLUTE_OUTPUT_SMALL_FOLDER_PATH,
    ABSOLUTE_OUTPUT_MEDIUM_FOLDER_PATH,
    ABSOLUTE_OUTPUT_LARGE_FOLDER_PATH,
    ABSOLUTE_OUTPUT_XLARGE_FOLDER_PATH,
    ABSOLUTE_INPUT_XLARGE_FOLDER_PATH,
    CLARKE_WRIGHT_SAVINGS_NAME,
    GENETIC_ALGORITHM_NAME,
    JSON,
    NN_ALGORITHM_NAME,
    GRASP_ALGORITHM_NAME
)
from src.utils.logger_config import logger

# from src.data_analyst.data_analyst import DataAnalyst
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
        logger.debug("file path %s, output path %s", file_name, output_path)
        algorithm.solve(csv_file, json_file, output_path)
        file_counter += 1
        # if file_counter == 2:
        #     break

    logger.info("Created %s files", file_counter)




def run_grasp_for_medium():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_MEDIUM_FOLDER_PATH,
        ABSOLUTE_OUTPUT_MEDIUM_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )


def run_grasp_for_large():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_LARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_LARGE_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )

def run_grasp_for_xlarge():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_XLARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_XLARGE_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )


def run_grasp_for_small():
    grasp_algorithm = GRASPVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_SMALL_FOLDER_PATH,
        ABSOLUTE_OUTPUT_SMALL_FOLDER_PATH.format(algorithm=GRASP_ALGORITHM_NAME),
        grasp_algorithm,
    )

def run_clarke_wright_savings_for_medium():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_MEDIUM_FOLDER_PATH,
        ABSOLUTE_OUTPUT_MEDIUM_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_clarke_wright_savings_for_large():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_LARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_LARGE_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )

def run_clarke_wright_savings_for_xlarge():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_XLARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_XLARGE_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_clarke_wright_savings_for_small():
    clarke_wright_savings_algorithm = ClarkeWrightSavingsVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_SMALL_FOLDER_PATH,
        ABSOLUTE_OUTPUT_SMALL_FOLDER_PATH.format(algorithm=CLARKE_WRIGHT_SAVINGS_NAME),
        clarke_wright_savings_algorithm,
    )


def run_generic_for_medium():
    genetic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_MEDIUM_FOLDER_PATH,
        ABSOLUTE_OUTPUT_MEDIUM_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        genetic_algorithm,
    )


def run_generic_for_large():
    generic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_LARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_LARGE_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        generic_algorithm,
    )

def run_generic_for_xlarge():
    generic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_XLARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_XLARGE_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        generic_algorithm,
    )

def run_generic_for_small():
    generic_algorithm = GeneticAlgorithmVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_SMALL_FOLDER_PATH,
        ABSOLUTE_OUTPUT_SMALL_FOLDER_PATH.format(algorithm=GENETIC_ALGORITHM_NAME),
        generic_algorithm,
    )


# Nearest neighbor algorithm
def run_nn_for_medium():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_MEDIUM_FOLDER_PATH,
        ABSOLUTE_OUTPUT_MEDIUM_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )


def run_nn_for_large():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_LARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_LARGE_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )

def run_nn_for_xlarge():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_XLARGE_FOLDER_PATH,
        ABSOLUTE_OUTPUT_XLARGE_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )


def run_nn_for_small():
    nn_algorithm = NearestNeighborVRP(vehicle_info)
    iterate_files(
        ABSOLUTE_INPUT_SMALL_FOLDER_PATH,
        ABSOLUTE_OUTPUT_SMALL_FOLDER_PATH.format(algorithm=NN_ALGORITHM_NAME),
        nn_algorithm,
    )

# run_nn_for_xlarge()
# run_nn_for_small()
# run_nn_for_medium()
# run_nn_for_large()
#
#
# run_generic_for_xlarge()
# run_generic_for_small()
# run_generic_for_medium()
# run_generic_for_large()
#
# run_clarke_wright_savings_for_xlarge()
# run_clarke_wright_savings_for_small()
# run_clarke_wright_savings_for_medium()
# run_clarke_wright_savings_for_large()


run_grasp_for_xlarge()
run_grasp_for_small()
run_grasp_for_medium()
run_grasp_for_large()
