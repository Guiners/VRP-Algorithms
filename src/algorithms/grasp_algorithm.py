import random
import time
from copy import deepcopy
from typing import List, Any

from src.algorithms.tools.vrp_tools import VRPInstance, VehicleInfo, VRPInstanceLoader
from src.utils.logger_config import logger


class GRASPVRP(VRPInstanceLoader):
    """
    GRASP (Greedy Randomized Adaptive Search Procedure) algorithm for the Vehicle Routing Problem (VRP).
    """

    def __init__(self, vehicle_info: VehicleInfo, max_iter: int = 100, neighborhood_size: int = 5) -> None:
        """
        Initialize the GRASP VRP solver.

        :param vehicle_info: Vehicle information (fuel consumption, price, etc.)
        :param max_iter: Number of GRASP iterations
        :param neighborhood_size: Initial size of the Restricted Candidate List (RCL)
        """
        super().__init__()
        self.vehicle_info = vehicle_info
        self.max_iter = max_iter
        self.neighborhood_size = neighborhood_size
        logger.info("Initialized GRASP VRP algorithm")

    def _distance(self, city1: VRPInstance, city2: VRPInstance) -> float:
        """
        Calculate the distance between two cities.

        :param city1: First city
        :param city2: Second city
        :return: Euclidean distance
        """
        return city1.distance_to(city2)

    def _split_cities(self, cities: List[VRPInstance], v: int) -> List[List[VRPInstance]]:
        """
        Split the list of cities into v chunks of nearly equal size.

        :param cities: List of city instances
        :param v: Number of vehicles (chunks)
        :return: List of city chunks
        """
        n = len(cities)
        base, rem = divmod(n, v)
        chunks = []
        start = 0
        for i in range(v):
            size = base + (1 if i < rem else 0)
            chunks.append(cities[start : start + size])
            start += size
        return chunks

    def _construct_route(self, chunk: List[VRPInstance], depot: VRPInstance, rcl_size: int) -> List[VRPInstance]:
        """
        Build a single route: depot -> cities in chunk -> depot,
        always choosing randomly from the RCL of nearest neighbors.

        :param chunk: List of cities for this route
        :param depot: Depot instance
        :param rcl_size: Size of the Restricted Candidate List
        :return: Constructed route
        """
        unvisited = set(chunk)
        route = [depot]
        current = depot

        while unvisited:
            neighbors = sorted(
                [(city, self._distance(current, city)) for city in unvisited],
                key=lambda x: x[1],
            )
            rcl = [city for city, _ in neighbors[:rcl_size]]
            next_city = random.choice(rcl)
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        route.append(depot)
        return route

    def _two_opt(self, route: List[VRPInstance]) -> List[VRPInstance]:
        """
        Apply 2-opt local search to a single route.

        :param route: Route to optimize
        :return: Locally optimized route
        """
        best = route
        best_dist = self._route_distance(best)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                    cand_dist = self._route_distance(candidate)
                    if cand_dist < best_dist:
                        best = candidate
                        best_dist = cand_dist
                        improved = True
                        break
                if improved:
                    break
        return best

    def _route_distance(self, route: List[VRPInstance]) -> float:
        """
        Calculate the total distance of a route.

        :param route: List of cities in the route
        :return: Total distance
        """
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += a.distance_to(b)
        return d

    def _total_distance(self, routes: List[List[VRPInstance]]) -> float:
        """
        Calculate the total distance for all routes.

        :param routes: List of routes
        :return: Total distance
        """
        return sum(self._route_distance(r) for r in routes)

    def solve(
        self, csv_path: str, config_path: str, output_file_path: str
    ) -> List[List[VRPInstance]]:
        """
        Run the GRASP algorithm to solve the VRP and save the results.

        :param csv_path: Path to the CSV file with city data
        :param config_path: Path to the JSON config file
        :param output_file_path: Path to save the results
        :return: List of best found routes (each route is a list of VRPInstance)
        """
        logger.info("Started GRASP VRP algorithm")
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        depot = data.depot
        cities = data.cities
        vehicles = data.vehicles

        chunks = self._split_cities(cities, vehicles)

        best_solution = None
        best_distance = float("inf")

        for iteration in range(self.max_iter):
            logger.debug("GRASP iteration %d/%d", iteration + 1, self.max_iter)
            rcl_size = self.neighborhood_size

            initial_routes = [
                self._construct_route(chunk, depot, rcl_size) for chunk in chunks
            ]

            improved_routes = [self._two_opt(route) for route in initial_routes]

            total_dist = self._total_distance(improved_routes)
            if total_dist < best_distance:
                best_distance = total_dist
                best_solution = deepcopy(improved_routes)
                logger.info(" New best at iter %d: %.2f", iteration + 1, best_distance)

        processing_time = time.time() - start_time
        total_km = best_distance / 1000.0

        logger.info(
            "GRASP completed in %.2f s, distance %.2f km", processing_time, total_km
        )
        self.save_results_to_file(
            total_km,
            processing_time,
            best_solution,
            self.vehicle_info,
            output_file_path,
        )
        logger.info("Results saved to %s", output_file_path)
        return best_solution