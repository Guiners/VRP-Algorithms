import time
import random
from copy import deepcopy

from src.algorithms.tools.vrp_tools import VRPInstanceLoader
from src.utils.logger_config import logger


class GRASPVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info, max_iter=100, neighborhood_size=5):
        super().__init__()
        self.vehicle_info = vehicle_info
        self.max_iter = max_iter                # liczba prób GRASP
        self.neighborhood_size = neighborhood_size  # początkowy rozmiar RCL
        logger.info("Initialized GRASP VRP algorithm")

    def _distance(self, city1, city2):
        return city1.distance_to(city2)

    def _split_cities(self, cities, v):
        """
        Podział listy cities na v fragmentów (chunks) możliwie równych rozmiarów.
        """
        n = len(cities)
        base, rem = divmod(n, v)
        chunks = []
        start = 0
        for i in range(v):
            size = base + (1 if i < rem else 0)
            chunks.append(cities[start:start+size])
            start += size
        return chunks

    def _construct_route(self, chunk, depot, rcl_size):
        """
        Buduje pojedynczą trasę: depot -> cities w chunk -> depot,
        wybierając zawsze losowo z RCL najbliższych sąsiadów.
        """
        unvisited = set(chunk)
        route = [depot]
        current = depot

        while unvisited:
            # posortuj nieodwiedzone wg odległości
            neighbors = sorted(
                [(city, self._distance(current, city)) for city in unvisited],
                key=lambda x: x[1]
            )
            # ogranicz do RCL: rcl_size najbliższych
            rcl = [city for city, _ in neighbors[:rcl_size]]
            next_city = random.choice(rcl)
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        route.append(depot)
        return route

    def _two_opt(self, route):
        """
        Lokalna poprawa 2-opt dla jednej trasy.
        """
        best = route
        best_dist = self._route_distance(best)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i+1, len(best) - 1):
                    candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    cand_dist = self._route_distance(candidate)
                    if cand_dist < best_dist:
                        best = candidate
                        best_dist = cand_dist
                        improved = True
                        break
                if improved:
                    break
        return best

    def _route_distance(self, route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += a.distance_to(b)
        return d

    def _total_distance(self, routes):
        return sum(self._route_distance(r) for r in routes)

    def solve(self, csv_path, config_path, output_file_path):
        logger.info("Started GRASP VRP algorithm")
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        depot = data.depot
        cities = data.cities
        vehicles = data.vehicles

        chunks = self._split_cities(cities, vehicles)

        best_solution = None
        best_distance = float('inf')

        for iteration in range(self.max_iter):
            logger.debug("GRASP iteration %d/%d", iteration+1, self.max_iter)
            rcl_size = self.neighborhood_size

            initial_routes = [
                self._construct_route(chunk, depot, rcl_size)
                for chunk in chunks
            ]

            improved_routes = [
                self._two_opt(route)
                for route in initial_routes
            ]

            total_dist = self._total_distance(improved_routes)
            if total_dist < best_distance:
                best_distance = total_dist
                best_solution = deepcopy(improved_routes)
                logger.info(" New best at iter %d: %.2f", iteration+1, best_distance)

        processing_time = time.time() - start_time
        total_km = best_distance / 1000.0

        logger.info("GRASP completed in %.2f s, distance %.2f km", processing_time, total_km)
        self.save_results_to_file(
            total_km,
            processing_time,
            best_solution,
            self.vehicle_info,
            output_file_path
        )
        logger.info("Results saved to %s", output_file_path)
        return best_solution
