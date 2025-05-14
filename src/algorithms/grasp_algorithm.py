import time
import random
from copy import deepcopy

from src.algorithms.tools.vrp_tools import VRPInstanceLoader
from src.utils.logger_config import logger


class GRASPVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info, max_iter=100, neighborhood_size=5):
        super().__init__()
        self.vehicle_info = vehicle_info
        self.max_iter = max_iter  # liczba prób GRASP
        self.neighborhood_size = neighborhood_size  # rozmiar sąsiedztwa w heurystyce greedy
        logger.info("Initialized GRASP VRP algorithm")

    def _distance(self, city1, city2):
        return city1.distance_to(city2)

    def _greedy_solution(self, cities, depot, max_vehicles):
        """
        Budowanie rozwiązania heurystyką greedy dla wielu pojazdów.
        Dzielimy miasta między max_vehicles tras, każda zaczyna i kończy się w depozycie.
        """
        unvisited = set(cities)
        routes = [[] for _ in range(max_vehicles)]  # lista tras (pojedynczych list miast)
        # Start każdej trasy od depozytu
        for route in routes:
            route.append(depot)

        # Przypisz miasta kolejno do tras, korzystając z greedy + losowość w ramach neighborhood_size
        vehicle_index = 0
        current_city = depot

        while unvisited:
            current_route = routes[vehicle_index]
            current_city = current_route[-1]

            # Znajdź najbliższych sąsiadów wśród nieodwiedzonych
            nearest_neighbors = sorted(
                [(city, self._distance(current_city, city)) for city in unvisited],
                key=lambda x: x[1]
            )[:self.neighborhood_size]

            if not nearest_neighbors:
                break

            # Wybierz losowo spośród najbliższych
            next_city = random.choice([city for city, _ in nearest_neighbors])
            current_route.append(next_city)
            unvisited.remove(next_city)

            # Przełącz pojazd (runda po pojazdach), aby równomiernie rozdzielić miasta
            vehicle_index = (vehicle_index + 1) % max_vehicles

        # Do każdej trasy dodaj depozyt na koniec (powrót)
        for route in routes:
            route.append(depot)

        return routes

    def _two_opt(self, route):
        """
        Lokalna poprawa trasy 2-opt.
        """
        best_route = route
        best_distance = self._calculate_total_distance([best_route])
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    new_distance = self._calculate_total_distance([new_route])

                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break  # Po znalezieniu poprawy restartuj pętlę
                if improved:
                    break

        return best_route

    def _calculate_total_distance(self, routes):
        total_distance = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += route[i].distance_to(route[i + 1])
        return total_distance

    def solve(self, csv_path, config_path, output_file_path):
        logger.info("Started GRASP VRP algorithm")
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        logger.info("Loaded dataset with %d cities and %d vehicles", len(data.cities), data.vehicles)

        best_solution = None
        best_distance = float('inf')

        for iteration in range(self.max_iter):
            logger.info("Iteration %d/%d", iteration + 1, self.max_iter)

            # Krok 1: Zbudowanie rozwiązania za pomocą heurystyki greedy z wieloma pojazdami
            initial_routes = self._greedy_solution(data.cities, data.depot, data.vehicles)

            # Krok 2: Lokalna poprawa każdej trasy osobno
            improved_routes = []
            for route in initial_routes:
                improved_route = self._two_opt(route)
                improved_routes.append(improved_route)

            # Krok 3: Sprawdzanie, czy poprawione rozwiązanie jest lepsze niż najlepsze znalezione dotychczas
            improved_distance = self._calculate_total_distance(improved_routes)
            if improved_distance < best_distance:
                best_solution = improved_routes
                best_distance = improved_distance

        # Zakończenie algorytmu: zapisanie wyników
        processing_time = time.time() - start_time
        total_distance_km = best_distance / 1000

        logger.info("Optimization completed in %.2f seconds", processing_time)
        logger.info("Best solution distance: %.2f km", total_distance_km)
        logger.critical("Number of routes (vehicles used): %d", len(best_solution))

        self.save_results_to_file(
            total_distance_km,
            processing_time,
            best_solution,
            self.vehicle_info,
            output_file_path
        )

        logger.info("Results saved to %s", output_file_path)
        return best_solution
