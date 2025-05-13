import math
import random
import time
from copy import deepcopy

from src.algorithms.tools.vrp_tools import VRPInstanceLoader
from src.utils.logger_config import logger


class AntColonyVRP(VRPInstanceLoader):
    def __init__(
        self,
        vehicle_info,
        num_ants=20,
        num_iterations=100,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        q=100,
    ):
        super().__init__()
        self.vehicle_info = vehicle_info
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # feromone importance
        self.beta = beta  # heuristic importance
        self.rho = rho  # pheromone evaporation rate
        self.q = q  # pheromone deposit factor

    def _init_pheromone_matrix(self, n):
        return [[1.0 for _ in range(n)] for _ in range(n)]

    def _distance_matrix(self, cities):
        n = len(cities)
        return [[cities[i].distance_to(cities[j]) for j in range(n)] for i in range(n)]

    def _calculate_probabilities(
        self, current_city_idx, unvisited, pheromone, distance
    ):
        probabilities = []
        denom = 0.0
        for j in unvisited:
            tau = pheromone[current_city_idx][j] ** self.alpha
            eta = (1.0 / distance[current_city_idx][j]) ** self.beta
            denom += tau * eta

        for j in unvisited:
            tau = pheromone[current_city_idx][j] ** self.alpha
            eta = (1.0 / distance[current_city_idx][j]) ** self.beta
            prob = (tau * eta) / denom if denom != 0 else 0
            probabilities.append((j, prob))

        return probabilities

    def _select_next_city(self, probabilities):
        r = random.random()
        cumulative = 0.0
        for city, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return city
        return probabilities[-1][0]  # fallback

    def _build_solution(self, pheromone, distance, depot_idx, cities, vehicles):
        n = len(cities)
        city_indices = list(range(n))
        solutions = []

        for _ in range(vehicles):
            route = []
            current_city = depot_idx
            unvisited = set(city_indices) - {depot_idx}

            while unvisited:
                probabilities = self._calculate_probabilities(
                    current_city, unvisited, pheromone, distance
                )
                next_city = self._select_next_city(probabilities)
                route.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
                if len(solutions) + 1 == vehicles and not unvisited:
                    break

            solutions.append(route)
        return solutions

    def _route_distance(self, route, distance, depot_idx):
        total = 0
        current = depot_idx
        for city in route:
            total += distance[current][city]
            current = city
        total += distance[current][depot_idx]
        return total

    def _update_pheromones(self, pheromone, solutions, distance, depot_idx):
        n = len(pheromone)
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= 1 - self.rho

        for solution in solutions:
            for route in solution:
                d = self._route_distance(route, distance, depot_idx)
                for i in range(len(route)):
                    from_city = depot_idx if i == 0 else route[i - 1]
                    to_city = route[i]
                    pheromone[from_city][to_city] += self.q / d
                    pheromone[to_city][from_city] += self.q / d
                pheromone[route[-1]][depot_idx] += self.q / d
                pheromone[depot_idx][route[-1]] += self.q / d

    def _simulate_once(self, cities, depot_idx, vehicles, max_iter=5):
        n = len(cities)
        pheromone = self._init_pheromone_matrix(n)
        distance = self._distance_matrix(cities)

        best_distance = float("inf")

        for _ in range(max_iter):
            all_solutions = []
            for _ in range(self.num_ants):
                solution = self._build_solution(pheromone, distance, depot_idx, cities, vehicles)
                all_solutions.append(solution)

                total_dist = sum(
                    self._route_distance(route, distance, depot_idx) for route in solution
                )
                if total_dist < best_distance:
                    best_distance = total_dist

            self._update_pheromones(pheromone, all_solutions, distance, depot_idx)

        return best_distance

    def random_search_params(self, cities, depot_idx, vehicles, attempts=10):
        n = len(cities)

        best_config = None
        best_score = float("inf")

        for _ in range(attempts):
            alpha = random.uniform(0.5, 3.0)
            beta = random.uniform(2.0, 5.0)
            rho = random.uniform(0.1, 0.9)
            q = random.uniform(10, 500)
            num_ants = random.randint(10, min(n, 100))
            num_iterations = random.randint(50, 300)

            candidate = AntColonyVRP(
                vehicle_info=self.vehicle_info,
                num_ants=num_ants,
                num_iterations=num_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                q=q,
            )

            try:
                score = candidate._simulate_once(cities, depot_idx, vehicles)
                if score < best_score:
                    best_score = score
                    best_config = (alpha, beta, rho, q, num_ants, num_iterations)
            except Exception as e:
                logger.warning(f"Random config failed: {e}")
                continue

        if best_config:
            self.alpha, self.beta, self.rho, self.q, self.num_ants, self.num_iterations = best_config
            logger.info(
                "Random search selected: alpha=%.2f, beta=%.2f, rho=%.2f, q=%.1f, ants=%d, iters=%d",
                self.alpha, self.beta, self.rho, self.q, self.num_ants, self.num_iterations
            )
        else:
            logger.warning("Random search failed to find a good configuration.")

    def solve(self, csv_path, config_path, output_file_path):
        logger.info("Started solving VRP using Ant Colony Optimization")
        start_time = time.time()

        data = self.load_dataset(csv_path, config_path)
        cities = [data.depot] + data.cities
        depot_idx = 0

        self.random_search_params(cities, depot_idx, data.vehicles)

        pheromone = self._init_pheromone_matrix(len(cities))
        distance = self._distance_matrix(cities)

        best_solution = None
        best_distance = float("inf")

        for iteration in range(self.num_iterations):
            logger.debug("Iteration %d", iteration + 1)
            all_solutions = []

            for _ in range(self.num_ants):
                solution = self._build_solution(
                    pheromone, distance, depot_idx, cities, data.vehicles
                )
                all_solutions.append(solution)

                total_dist = sum(
                    self._route_distance(route, distance, depot_idx) for route in solution
                )
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_solution = deepcopy(solution)
                    logger.info("New best solution with distance: %.2f", best_distance)

            self._update_pheromones(pheromone, all_solutions, distance, depot_idx)

        end_time = time.time()
        total_distance_km = best_distance / 1000
        processing_time = end_time - start_time
        logger.info("Optimization completed in %.2f seconds", processing_time)
        logger.info("Total distance: %.2f km", total_distance_km)

        routes = []
        for route in best_solution:
            full_route = [cities[0]] + [cities[i] for i in route] + [cities[0]]
            routes.append(full_route)

        self.save_results_to_file(
            total_distance_km,
            processing_time,
            routes,
            self.vehicle_info,
            output_file_path,
        )
        logger.info("Results saved to %s", output_file_path)
        return routes