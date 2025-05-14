import time
from copy import deepcopy

from src.algorithms.tools.vrp_tools import VRPInstanceLoader
from src.utils.logger_config import logger


class ClarkeWrightSavingsVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info):
        super().__init__()
        self.vehicle_info = vehicle_info
        logger.info("Initialized ClarkeWrightSavingsVRP algorithm")

    def _distance(self, city1, city2):
        return city1.distance_to(city2)

    def _calculate_savings(self, cities, depot):
        savings = []
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                c1 = cities[i]
                c2 = cities[j]
                saving = (self._distance(depot, c1) +
                          self._distance(depot, c2) -
                          self._distance(c1, c2))
                savings.append(((c1.city_id, c2.city_id), saving))
        savings.sort(key=lambda x: x[1], reverse=True)
        return savings

    def _initialize_routes(self, cities, depot):
        routes = {}
        for city in cities:
            routes[city.city_id] = [depot, city, depot]
        return routes

    def _merge_routes(self, routes, savings_list, max_vehicles):
        route_mapping = {city_id: city_id for city_id in routes}
        merged_routes = deepcopy(routes)

        for (i, j), saving in savings_list:
            if i not in route_mapping or j not in route_mapping:
                continue

            route_i_key = route_mapping[i]
            route_j_key = route_mapping[j]

            if route_i_key == route_j_key:
                continue

            route_i = merged_routes[route_i_key]
            route_j = merged_routes[route_j_key]

            # Sprawdzenie, czy połączenie tras minimalizuje odległość
            if route_i[-2].city_id == i and route_j[1].city_id == j:
                new_route = route_i[:-1] + route_j[1:]
                new_key = min(route_i_key, route_j_key)

                del merged_routes[route_i_key]
                del merged_routes[route_j_key]
                merged_routes[new_key] = new_route

                for city in new_route[1:-1]:
                    route_mapping[city.city_id] = new_key

            # Przerwanie, jeśli liczba tras osiągnie maksymalną liczbę pojazdów
            if len(merged_routes) <= max_vehicles:
                break

        # Łączenie pozostałych tras, jeśli liczba tras przekracza liczbę pojazdów
        while len(merged_routes) > max_vehicles:
            keys = list(merged_routes.keys())
            route1 = merged_routes.pop(keys[0])
            route2 = merged_routes.pop(keys[1])
            new_route = route1[:-1] + route2[1:]
            merged_routes[min(keys[0], keys[1])] = new_route

        return list(merged_routes.values())

    def _calculate_total_distance(self, routes):
        total = 0
        for route in routes:
            for i in range(len(route) - 1):
                total += route[i].distance_to(route[i + 1])
        return total

    def solve(self, csv_path, config_path, output_file_path):
        logger.info("Started solving VRP using Clarke-Wright Savings")
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        logger.info("Loaded dataset with %d cities and %d vehicles", len(data.cities), data.vehicles)

        initial_routes = self._initialize_routes(data.cities, data.depot)

        savings = self._calculate_savings(data.cities, data.depot)

        merged_routes = self._merge_routes(initial_routes, savings, data.vehicles)
        total_distance = self._calculate_total_distance(merged_routes)
        processing_time = time.time() - start_time
        total_distance_km = total_distance / 1000

        logger.info("Optimization completed in %.2f seconds", processing_time)
        logger.info("Total distance: %.2f km", total_distance_km)

        self.save_results_to_file(
            total_distance_km,
            processing_time,
            merged_routes,
            self.vehicle_info,
            output_file_path
        )

        logger.info("Results saved to %s", output_file_path)
        return merged_routes
