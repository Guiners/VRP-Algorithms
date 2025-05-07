import math
import time
import json
from copy import deepcopy

from src.algorithms.utils.vrp_tools import VehicleInfo, VRPInstanceLoader
from src.utils.logger_config import logger


class NearestNeighborVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info):
        self.vehicle_info = vehicle_info
    @staticmethod
    def _euclidean_distance(city1, city2):
        return math.hypot(city1.x - city2.x, city1.y - city2.y)

    def solve(self, csv_path, config_path, output_file_path):
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        clients = deepcopy(data.cities)
        total_distance_in_meters = 0

        routes = [[] for _ in range(data.vehicles)]
        remaining_clients = clients

        per_vehicle = len(clients) // data.vehicles
        extras = len(clients) % data.vehicles
        split_sizes = [
            per_vehicle + (1 if i < extras else 0) for i in range(data.vehicles)
        ]

        start_index = 0
        for vehicle_index in range(data.vehicles):
            route_clients = remaining_clients[
                start_index : start_index + split_sizes[vehicle_index]
            ]
            start_index += split_sizes[vehicle_index]

            route = [data.depot]
            current_city = data.depot
            while route_clients:
                next_city = min(
                    route_clients,
                    key=lambda c: self._euclidean_distance(current_city, c),
                )
                total_distance_in_meters += current_city.distance_to(next_city)
                route.append(next_city)
                route_clients.remove(next_city)
                current_city = next_city

            route.append(data.depot)
            total_distance_in_meters += current_city.distance_to(data.depot)
            routes[vehicle_index] = route

        end_time = time.time()
        total_distance_in_km = total_distance_in_meters / 1000
        processing_time = end_time - start_time
        data_to_save = {
            "distance": round(total_distance_in_km, 2),
            "processing_time": processing_time,
            "cost": self.vehicle_info.count_cost(total_distance_in_km),
        }
        logger.info("Total distance: %s", data_to_save["distance"])
        logger.info("Total processing time: %s", data_to_save["processing_time"])
        logger.info("Cost of whole operation in $: %s", data_to_save["cost"])
        with open(output_file_path, "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)

        return routes
