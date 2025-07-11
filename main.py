import json
import sys
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
from sklearn.cluster import KMeans

# ========================= 데이터 구조 정의 =========================

@dataclass
class Box:
    """박스 정보를 저장하는 클래스"""
    order_number: int
    box_id: str
    destination: str
    width: float
    length: float
    height: float
    volume: float

    def __post_init__(self):
        self.volume = self.width * self.length * self.height

@dataclass
class Location:
    """위치 정보를 저장하는 클래스"""
    id: str
    longitude: float
    latitude: float

@dataclass
class Vehicle:
    """차량 정보를 저장하는 클래스"""
    max_width: float = 160.0  # cm
    max_length: float = 280.0  # cm
    max_height: float = 180.0  # cm
    max_volume: float = 160.0 * 280.0 * 180.0
    fixed_cost: int = 150000  # 고정비 150,000원
    fuel_cost_per_km: int = 500  # 500원/km

@dataclass
class Cluster:
    """클러스터 정보를 저장하는 클래스"""
    id: int
    destinations: List[str]
    orders: List[Box]
    total_volume: float
    center_lat: float
    center_lon: float
    estimated_vehicles: int

@dataclass
class Route:
    """라우팅 정보를 저장하는 클래스"""
    cluster_id: int
    destinations: List[str]  # 방문 순서대로 정렬된 목적지
    total_distance: int
    route_cost: int

@dataclass
class PackedBox:
    """적재된 박스 정보"""
    box: Box
    x: float
    y: float
    z: float
    stacking_order: int  # 적재 순서 (LIFO를 위해)

@dataclass
class VehiclePlan:
    """차량별 적재 및 배송 계획"""
    vehicle_id: int
    cluster_id: int
    route: List[str]  # 배송 순서
    packed_boxes: List[PackedBox]
    total_volume: float
    routing_cost: int
    unloading_cost: int
    total_cost: int

# ========================= 데이터 전처리 클래스 =========================

class DataPreprocessor:
    """데이터 전처리를 담당하는 클래스"""

    def __init__(self):
        self.depot: Location = None
        self.destinations: Dict[str, Location] = {}
        self.orders: List[Box] = []
        self.distance_matrix: Dict[Tuple[str, str], int] = {}
        self.orders_by_destination: Dict[str, List[Box]] = defaultdict(list)
        self.vehicle = Vehicle()

    def load_data(self, data_file: str, distance_file: str):
        """데이터 파일들을 로드하는 메인 함수"""
        self._load_json_data(data_file)
        self._load_distance_matrix(distance_file)
        self._preprocess_data()

    def _load_json_data(self, data_file: str):
        """JSON 파일에서 데이터를 로드"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Depot 정보 로드
        depot_data = data['depot']
        self.depot = Location(
            id="Depot",
            longitude=depot_data['location']['longitude'],
            latitude=depot_data['location']['latitude']
        )

        # 목적지 정보 로드
        for dest_data in data['destinations']:
            dest_id = dest_data['destination_id']
            self.destinations[dest_id] = Location(
                id=dest_id,
                longitude=dest_data['location']['longitude'],
                latitude=dest_data['location']['latitude']
            )

        # 주문 정보 로드
        for order_data in data['orders']:
            box = Box(
                order_number=order_data['order_number'],
                box_id=order_data['box_id'],
                destination=order_data['destination'],
                width=order_data['dimension']['width'],
                length=order_data['dimension']['length'],
                height=order_data['dimension']['height'],
                volume=0  # __post_init__에서 계산됨
            )
            self.orders.append(box)

    def _load_distance_matrix(self, distance_file: str):
        """거리 매트릭스 파일을 로드"""
        with open(distance_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header = lines[0].strip().split('\t')
        locations = header[1:]

        for i, line in enumerate(lines[1:], 0):
            parts = line.strip().split('\t')
            if len(parts) <= 1:
                continue

            from_location = parts[0]

            for j, distance_str in enumerate(parts[1:], 0):
                if j >= len(locations):
                    break

                to_location = locations[j]

                try:
                    distance = int(distance_str)
                    self.distance_matrix[(from_location, to_location)] = distance
                except ValueError:
                    continue

    def _preprocess_data(self):
        """데이터 전처리 수행"""
        self._group_orders_by_destination()
        self._analyze_box_sizes()
        self._validate_data()

    def _group_orders_by_destination(self):
        """목적지별로 주문을 그룹화"""
        for order in self.orders:
            self.orders_by_destination[order.destination].append(order)

    def _analyze_box_sizes(self):
        """박스 크기별 분석 및 통계"""
        size_groups = defaultdict(int)
        total_volume = 0

        for order in self.orders:
            if (order.width, order.length, order.height) == (30, 40, 30):
                size_groups['small'] += 1
            elif (order.width, order.length, order.height) == (30, 50, 40):
                size_groups['medium'] += 1
            elif (order.width, order.length, order.height) == (50, 60, 50):
                size_groups['large'] += 1
            else:
                size_groups['custom'] += 1

            total_volume += order.volume

        estimated_vehicles = math.ceil(total_volume / self.vehicle.max_volume)

    def _validate_data(self):
        """데이터 유효성 검증"""
        errors = []

        for order in self.orders:
            if order.destination not in self.destinations:
                errors.append(f"주문 {order.order_number}: 존재하지 않는 목적지 {order.destination}")

        for order in self.orders:
            if (order.width > self.vehicle.max_width or
                order.length > self.vehicle.max_length or
                order.height > self.vehicle.max_height):
                errors.append(f"주문 {order.order_number}: 박스가 차량 적재함보다 큼")

    def get_distance(self, from_location: str, to_location: str) -> int:
        """두 위치 간의 거리를 반환"""
        distance = self.distance_matrix.get((from_location, to_location), None)
        if distance is None:
            return 999999
        return distance

# ========================= 세밀 조정된 클러스터링 클래스 =========================

class ClusteringManager:
    """원본 기반 세밀 조정된 클러스터링"""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.clusters: List[Cluster] = []
        self.destination_to_cluster: Dict[str, int] = {}
        self.vehicle_capacity = preprocessor.vehicle.max_volume

    def create_clusters(self, max_cluster_volume_ratio: float = 0.82, use_kmeans: bool = True):
        """세밀 조정된 클러스터 생성 (82% 활용률로 소폭 상향)"""
        dest_info = self._analyze_destinations()

        if use_kmeans:
            initial_clusters = self._improved_kmeans_clustering(dest_info)
        else:
            initial_clusters = self._geographic_clustering(dest_info)

        # 부피 제약 조건 적용
        volume_adjusted_clusters = self._adjust_for_volume_constraints(
            initial_clusters, max_cluster_volume_ratio
        )

        # 작은 클러스터 통합 (핵심 개선)
        final_clusters = self._consolidate_small_clusters(volume_adjusted_clusters)

        self._finalize_clusters(final_clusters)

    def _analyze_destinations(self) -> Dict[str, Dict]:
        """목적지별 주문 정보 분석"""
        dest_info = {}

        for dest_id, orders in self.preprocessor.orders_by_destination.items():
            location = self.preprocessor.destinations[dest_id]
            total_volume = sum(order.volume for order in orders)

            box_sizes = {'small': 0, 'medium': 0, 'large': 0}
            for order in orders:
                if (order.width, order.length, order.height) == (30, 40, 30):
                    box_sizes['small'] += 1
                elif (order.width, order.length, order.height) == (30, 50, 40):
                    box_sizes['medium'] += 1
                elif (order.width, order.length, order.height) == (50, 60, 50):
                    box_sizes['large'] += 1

            dest_info[dest_id] = {
                'location': location,
                'orders': orders,
                'order_count': len(orders),
                'total_volume': total_volume,
                'box_sizes': box_sizes,
                'priority': self._calculate_destination_priority(orders, total_volume)
            }

        return dest_info

    def _calculate_destination_priority(self, orders: List[Box], total_volume: float) -> float:
        """목적지 우선순위 계산"""
        order_count = len(orders)
        volume_score = total_volume / 100000

        large_box_ratio = sum(1 for order in orders
                             if (order.width, order.length, order.height) == (50, 60, 50)) / order_count

        return order_count * 0.4 + volume_score * 0.4 + large_box_ratio * 0.2

    def _improved_kmeans_clustering(self, dest_info: dict) -> list:
        """개선된 K-means 클러스터링 (차량 수 최소화 중심)"""
        dest_ids = list(dest_info.keys())
        coords = np.array([
            [dest_info[d]['location'].latitude, dest_info[d]['location'].longitude]
            for d in dest_ids
        ])

        # 총 부피 기반으로 더 적극적인 클러스터 수 계산
        total_volume = sum(dest_info[d]['total_volume'] for d in dest_ids)

        # 82% 활용률 기준으로 클러스터 수 계산 (원본의 80%에서 소폭 상향)
        min_clusters = int(np.ceil(total_volume / (self.vehicle_capacity * 0.82)))

        # 최대 클러스터 수를 원본보다 줄여서 차량 수 감소 유도
        max_clusters = min(len(dest_ids), 10)  # 12 → 10으로 감소
        n_clusters = max(min_clusters, 4)  # 최소 4개 클러스터
        n_clusters = min(n_clusters, max_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(dest_ids[idx])

        return clusters

    def _geographic_clustering(self, dest_info: Dict) -> List[List[str]]:
        """지리적 근접성 기반 클러스터링"""
        unassigned = set(dest_info.keys())
        clusters = []

        while unassigned:
            center_dest = max(unassigned, key=lambda d: dest_info[d]['priority'])
            cluster = [center_dest]
            unassigned.remove(center_dest)

            center_location = dest_info[center_dest]['location']

            candidates = list(unassigned)
            candidates.sort(key=lambda d: self._calculate_distance(
                center_location.latitude, center_location.longitude,
                dest_info[d]['location'].latitude, dest_info[d]['location'].longitude
            ))

            cluster_volume = dest_info[center_dest]['total_volume']
            max_cluster_volume = self.vehicle_capacity * 0.75  # 원본보다 소폭 상향

            for candidate in candidates:
                candidate_volume = dest_info[candidate]['total_volume']

                if (cluster_volume + candidate_volume <= max_cluster_volume and
                    len(cluster) < 9):  # 클러스터 크기 제한 소폭 증가

                    distance = self._calculate_distance(
                        center_location.latitude, center_location.longitude,
                        dest_info[candidate]['location'].latitude, dest_info[candidate]['location'].longitude
                    )

                    if distance <= 0.06:  # 거리 제약 소폭 완화
                        cluster.append(candidate)
                        cluster_volume += candidate_volume
                        unassigned.remove(candidate)

            clusters.append(cluster)

        return clusters

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 위치 간의 유클리드 거리 계산"""
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    def _adjust_for_volume_constraints(self, clusters: List[List[str]], max_ratio: float) -> List[List[str]]:
        """부피 제약 조건에 따른 클러스터 조정"""
        adjusted_clusters = []
        max_volume = self.vehicle_capacity * max_ratio

        for cluster_destinations in clusters:
            cluster_volume = sum(
                sum(order.volume for order in self.preprocessor.orders_by_destination[dest])
                for dest in cluster_destinations
            )

            if cluster_volume <= max_volume:
                adjusted_clusters.append(cluster_destinations)
            else:
                split_clusters = self._split_large_cluster(cluster_destinations, max_volume)
                adjusted_clusters.extend(split_clusters)

        return adjusted_clusters

    def _consolidate_small_clusters(self, clusters: List[List[str]]) -> List[List[str]]:
        """작은 클러스터들을 통합하여 차량 수 최소화 (핵심 개선)"""
        consolidated = []
        small_clusters = []

        # 50% 미만 활용률인 클러스터를 작은 클러스터로 분류
        for cluster in clusters:
            cluster_volume = sum(
                sum(order.volume for order in self.preprocessor.orders_by_destination[dest])
                for dest in cluster
            )
            utilization = cluster_volume / self.vehicle_capacity

            if utilization < 0.50:  # 50% 미만
                small_clusters.append((cluster, cluster_volume))
            else:
                consolidated.append(cluster)

        # 작은 클러스터들을 통합
        while len(small_clusters) > 1:
            # 가장 작은 두 클러스터를 찾아서 통합
            small_clusters.sort(key=lambda x: x[1])  # 부피순 정렬

            cluster1, volume1 = small_clusters.pop(0)
            cluster2, volume2 = small_clusters.pop(0)

            # 통합 가능한지 확인
            combined_volume = volume1 + volume2
            if combined_volume <= self.vehicle_capacity * 0.82:
                # 통합
                combined_cluster = cluster1 + cluster2
                small_clusters.append((combined_cluster, combined_volume))
            else:
                # 통합 불가능하면 각각 보존
                consolidated.append(cluster1)
                consolidated.append(cluster2)
                break

        # 남은 작은 클러스터 추가
        for cluster, _ in small_clusters:
            consolidated.append(cluster)

        return consolidated

    def _split_large_cluster(self, destinations: List[str], max_volume: float) -> List[List[str]]:
        """큰 클러스터를 분할"""
        dest_volumes = []
        for dest in destinations:
            orders = self.preprocessor.orders_by_destination[dest]
            total_volume = sum(order.volume for order in orders)
            dest_volumes.append((dest, total_volume))

        dest_volumes.sort(key=lambda x: x[1], reverse=True)

        split_clusters = []
        current_cluster = []
        current_volume = 0

        for dest, volume in dest_volumes:
            if current_volume + volume <= max_volume:
                current_cluster.append(dest)
                current_volume += volume
            else:
                if current_cluster:
                    split_clusters.append(current_cluster)
                current_cluster = [dest]
                current_volume = volume

        if current_cluster:
            split_clusters.append(current_cluster)

        return split_clusters

    def _finalize_clusters(self, cluster_destinations: List[List[str]]):
        """최종 클러스터 정보 생성 및 저장"""
        self.clusters = []

        for i, destinations in enumerate(cluster_destinations):
            all_orders = []
            for dest in destinations:
                all_orders.extend(self.preprocessor.orders_by_destination[dest])

            if destinations:
                center_lat = sum(self.preprocessor.destinations[dest].latitude
                               for dest in destinations) / len(destinations)
                center_lon = sum(self.preprocessor.destinations[dest].longitude
                               for dest in destinations) / len(destinations)
            else:
                center_lat = center_lon = 0

            total_volume = sum(order.volume for order in all_orders)
            estimated_vehicles = max(1, math.ceil(total_volume / self.vehicle_capacity))

            cluster = Cluster(
                id=i,
                destinations=destinations,
                orders=all_orders,
                total_volume=total_volume,
                center_lat=center_lat,
                center_lon=center_lon,
                estimated_vehicles=estimated_vehicles
            )

            self.clusters.append(cluster)

            for dest in destinations:
                self.destination_to_cluster[dest] = i

# ========================= 라우팅 최적화 클래스 (원본 유지) =========================

class RoutingOptimizer:
    """라우팅 최적화를 담당하는 클래스"""

    def __init__(self, preprocessor: DataPreprocessor, clustering_manager: ClusteringManager):
        self.preprocessor = preprocessor
        self.clustering_manager = clustering_manager
        self.routes: List[Route] = []

    def optimize_routes(self):
        """모든 클러스터에 대해 라우팅 최적화 수행"""
        self.routes = []
        for cluster in self.clustering_manager.clusters:
            if len(cluster.destinations) == 1:
                route = self._create_simple_route(cluster)
            else:
                route = self._solve_tsp_for_cluster(cluster)

            self.routes.append(route)

    def _create_simple_route(self, cluster: Cluster) -> Route:
        """단일 목적지 클러스터에 대한 간단한 경로 생성"""
        destination = cluster.destinations[0]

        depot_to_dest = self.preprocessor.get_distance("Depot", destination)
        dest_to_depot = self.preprocessor.get_distance(destination, "Depot")
        total_distance = depot_to_dest + dest_to_depot

        route_cost = self.preprocessor.vehicle.fixed_cost + (total_distance * self.preprocessor.vehicle.fuel_cost_per_km)

        return Route(
            cluster_id=cluster.id,
            destinations=[destination],
            total_distance=total_distance,
            route_cost=route_cost
        )

    def _solve_tsp_for_cluster(self, cluster: Cluster) -> Route:
        """클러스터에 대한 TSP 문제 해결"""
        destinations = cluster.destinations

        if len(destinations) <= 3:
            best_route = self._brute_force_tsp(destinations)
        else:
            best_route = self._nearest_neighbor_tsp(destinations)
            best_route = self._two_opt_improvement(best_route)

        total_distance = self._calculate_route_distance(best_route)
        route_cost = self.preprocessor.vehicle.fixed_cost + (total_distance * self.preprocessor.vehicle.fuel_cost_per_km)

        return Route(
            cluster_id=cluster.id,
            destinations=best_route,
            total_distance=total_distance,
            route_cost=route_cost
        )

    def _brute_force_tsp(self, destinations: List[str]) -> List[str]:
        """소규모 TSP를 위한 완전탐색"""
        from itertools import permutations

        best_distance = float('inf')
        best_route = destinations[:]

        for perm in permutations(destinations):
            distance = self._calculate_route_distance(list(perm))
            if distance < best_distance:
                best_distance = distance
                best_route = list(perm)

        return best_route

    def _nearest_neighbor_tsp(self, destinations: List[str]) -> List[str]:
        """Nearest Neighbor 휴리스틱"""
        if not destinations:
            return []

        best_start = destinations[0]
        best_distance = self.preprocessor.get_distance("Depot", best_start)

        for dest in destinations[1:]:
            distance = self.preprocessor.get_distance("Depot", dest)
            if distance < best_distance:
                best_distance = distance
                best_start = dest

        route = [best_start]
        unvisited = set(destinations) - {best_start}
        current = best_start

        while unvisited:
            next_dest = None
            min_distance = 999999

            for dest in unvisited:
                distance = self.preprocessor.get_distance(current, dest)
                if distance < min_distance:
                    min_distance = distance
                    next_dest = dest

            if next_dest is None:
                next_dest = list(unvisited)[0]

            route.append(next_dest)
            unvisited.remove(next_dest)
            current = next_dest

        return route

    def _two_opt_improvement(self, route: List[str]) -> List[str]:
        """2-opt 알고리즘으로 경로 개선"""
        if len(route) < 4:
            return route

        best_route = route[:]
        best_distance = self._calculate_route_distance(best_route)
        improved = True

        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route)

                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        route = new_route
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def _calculate_route_distance(self, destinations: List[str]) -> int:
        """경로의 총 거리 계산"""
        if not destinations:
            return 0

        total_distance = 0

        distance = self.preprocessor.get_distance("Depot", destinations[0])
        if distance >= 999999:
            return 999999
        total_distance += distance

        for i in range(len(destinations) - 1):
            distance = self.preprocessor.get_distance(destinations[i], destinations[i+1])
            if distance >= 999999:
                return 999999
            total_distance += distance

        distance = self.preprocessor.get_distance(destinations[-1], "Depot")
        if distance >= 999999:
            return 999999
        total_distance += distance

        return total_distance

# ========================= 원본 기반 적재 최적화 클래스 =========================

class PackingOptimizer:
    """원본 기반 세밀 조정된 3D 빈 패킹"""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.vehicle = preprocessor.vehicle
        self.shuffling_cost = 500

    def optimize_packing_for_route(self, route: Route) -> List[VehiclePlan]:
        """경로에 대한 적재 최적화 수행"""
        all_orders = []
        for dest in route.destinations:
            all_orders.extend(self.preprocessor.orders_by_destination[dest])

        vehicle_plans = self._create_vehicle_plans(route, all_orders)
        return vehicle_plans

    def _create_vehicle_plans(self, route: Route, orders: List[Box]) -> List[VehiclePlan]:
        """차량별 적재 계획 생성 (원본 기반, 소폭 개선)"""
        vehicle_plans = []

        orders_by_dest = defaultdict(list)
        for order in orders:
            orders_by_dest[order.destination].append(order)

        optimized_loading_order = self._optimize_loading_order(route.destinations, orders_by_dest)

        remaining_orders = optimized_loading_order[:]
        vehicle_id = 0

        while remaining_orders:
            # 82% 활용률 목표로 더 많은 박스 적재 시도
            best_combination = []
            best_count = 0

            # 더 적극적으로 박스 수를 늘려가며 시도
            for i in range(len(remaining_orders), 0, -1):
                candidate_boxes = remaining_orders[:i]

                # 부피 제약 확인 (82% 이하)
                total_volume = sum(box.volume for box in candidate_boxes)
                if total_volume > self.vehicle.max_volume * 0.82:
                    continue

                try:
                    packed_boxes = self._pack_boxes_3d_improved(candidate_boxes)
                    # 높이 제한 확인
                    if all(pb.z + pb.box.height <= self.vehicle.max_height + 1e-6 for pb in packed_boxes):
                        best_combination = candidate_boxes
                        best_count = i
                        break
                except:
                    continue

            if not best_combination:
                # 한 박스도 적재 불가능한 경우 최소 1개는 적재
                best_combination = [remaining_orders[0]]
                best_count = 1

            plan = self._create_single_vehicle_plan(vehicle_id, route, best_combination)
            vehicle_plans.append(plan)
            vehicle_id += 1
            remaining_orders = remaining_orders[best_count:]

        return vehicle_plans

    def _optimize_loading_order(self, destinations: List[str], orders_by_dest: Dict[str, List[Box]]) -> List[Box]:
        """원본 적재 순서 최적화 유지"""
        loading_order = []
        reversed_destinations = destinations[::-1]

        for dest in reversed_destinations:
            dest_orders = orders_by_dest.get(dest, [])
            if not dest_orders:
                continue

            dest_orders_sorted = sorted(dest_orders, key=lambda box: (
                -box.volume,
                -box.height,
                -box.width * box.length
            ))

            loading_order.extend(dest_orders_sorted)

        return loading_order

    def _pack_boxes_3d_improved(self, boxes: List[Box]) -> List[PackedBox]:
        """개선된 3D 빈 패킹 (겹침 방지 강화)"""
        packed_boxes = []

        sorted_boxes = sorted(boxes, key=lambda b: (
            -(b.width * b.length),
            -b.volume,
            b.height,
            b.box_id
        ))

        occupied_spaces = []

        for i, box in enumerate(sorted_boxes):
            position = self._find_best_position_safe(box, occupied_spaces)

            packed_box = PackedBox(
                box=box,
                x=position[0],
                y=position[1],
                z=position[2],
                stacking_order=i
            )
            packed_boxes.append(packed_box)

            occupied_space = {
                'x1': position[0],
                'y1': position[1],
                'z1': position[2],
                'x2': position[0] + box.width,
                'y2': position[1] + box.length,
                'z2': position[2] + box.height,
                'destination': box.destination,
                'box_id': box.box_id
            }
            occupied_spaces.append(occupied_space)

        return packed_boxes

    def _find_best_position_safe(self, box: Box, occupied_spaces: List[Dict]) -> Tuple[float, float, float]:
        """안전한 위치 찾기 (겹침 방지 강화)"""
        max_width = self.vehicle.max_width
        max_length = self.vehicle.max_length
        max_height = self.vehicle.max_height

        candidates = []
        step_size = 5  # 더 세밀한 탐색

        # 바닥부터 차근차근 탐색
        for z in range(0, int(max_height - box.height + 1), step_size):
            for y in range(0, int(max_length - box.length + 1), step_size):
                for x in range(0, int(max_width - box.width + 1), step_size):
                    if (x + box.width <= max_width and
                        y + box.length <= max_length and
                        z + box.height <= max_height):

                        if not self._check_overlap_strict_safe(x, y, z, box, occupied_spaces):
                            candidates.append((x, y, z))

        if not candidates:
            # 모서리 기반 탐색
            candidates = self._find_corner_positions_safe(box, occupied_spaces, max_width, max_length, max_height)

        if not candidates:
            return (0.0, 0.0, 0.0)

        candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        return (float(candidates[0][0]), float(candidates[0][1]), float(candidates[0][2]))

    def _find_corner_positions_safe(self, box: Box, occupied_spaces: List[Dict], max_width: float, max_length: float, max_height: float) -> List[Tuple[float, float, float]]:
        """안전한 모서리 위치 탐색"""
        candidates = [(0, 0, 0)]

        for space in occupied_spaces:
            edge_positions = [
                (space['x2'], space['y1'], space['z1']),
                (space['x1'], space['y2'], space['z1']),
                (space['x1'], space['y1'], space['z2']),
            ]
            candidates.extend(edge_positions)

        valid_candidates = []
        for x, y, z in candidates:
            if (x + box.width <= max_width and
                y + box.length <= max_length and
                z + box.height <= max_height):
                if not self._check_overlap_strict_safe(x, y, z, box, occupied_spaces):
                    valid_candidates.append((x, y, z))

        return valid_candidates

    def _check_overlap_strict_safe(self, x: float, y: float, z: float, box: Box, occupied_spaces: List[Dict]) -> bool:
        """강화된 겹침 검사 (더 큰 마진)"""
        box_x2 = x + box.width
        box_y2 = y + box.length
        box_z2 = z + box.height

        tolerance = 0.1  # 더 큰 마진으로 겹침 방지

        for space in occupied_spaces:
            x_overlap = (box_x2 > space['x1'] + tolerance and x < space['x2'] - tolerance)
            y_overlap = (box_y2 > space['y1'] + tolerance and y < space['y2'] - tolerance)
            z_overlap = (box_z2 > space['z1'] + tolerance and z < space['z2'] - tolerance)
            if x_overlap and y_overlap and z_overlap:
                return True
        return False

    def _create_single_vehicle_plan(self, vehicle_id: int, route: Route, boxes: List[Box]) -> VehiclePlan:
        """단일 차량에 대한 적재 계획 생성"""
        packed_boxes = self._pack_boxes_3d_improved(boxes)
        unloading_cost = self._calculate_unloading_cost(packed_boxes, route.destinations)

        vehicle_destinations = []
        box_destinations = {box.destination for box in boxes}
        for dest in route.destinations:
            if dest in box_destinations:
                vehicle_destinations.append(dest)

        vehicle_routing_cost = self._calculate_vehicle_routing_cost(
            route.route_cost, len(boxes), sum(len(self.preprocessor.orders_by_destination[dest])
                                            for dest in route.destinations)
        )

        return VehiclePlan(
            vehicle_id=vehicle_id,
            cluster_id=route.cluster_id,
            route=vehicle_destinations,
            packed_boxes=packed_boxes,
            total_volume=sum(box.volume for box in boxes),
            routing_cost=vehicle_routing_cost,
            unloading_cost=unloading_cost,
            total_cost=vehicle_routing_cost + unloading_cost
        )

    def _calculate_unloading_cost(self, packed_boxes: List[PackedBox], delivery_order: List[str]) -> int:
        """하차 비용 계산"""
        total_shuffling = 0
        remaining_boxes = packed_boxes[:]

        for dest in delivery_order:
            dest_boxes = [pb for pb in remaining_boxes if pb.box.destination == dest]

            for target_box in dest_boxes:
                shuffling_count = self._calculate_shuffling_for_box(target_box, remaining_boxes)
                total_shuffling += shuffling_count
                remaining_boxes.remove(target_box)

        return total_shuffling * self.shuffling_cost

    def _calculate_shuffling_for_box(self, target_box: PackedBox, remaining_boxes: List[PackedBox]) -> int:
        """셔플링 횟수 계산"""
        shuffling_count = 0
        truck_exit_y = self.vehicle.max_length

        for other_box in remaining_boxes:
            if other_box == target_box:
                continue

            if self._is_box_above(other_box, target_box):
                shuffling_count += 1
                continue

            if self._is_blocking_exit_path(other_box, target_box, truck_exit_y):
                shuffling_count += 1

        return shuffling_count

    def _is_box_above(self, blocker: PackedBox, target: PackedBox) -> bool:
        """박스가 위에 있는지 확인"""
        x_overlap = (blocker.x < target.x + target.box.width and
                    blocker.x + blocker.box.width > target.x)
        y_overlap = (blocker.y < target.y + target.box.length and
                    blocker.y + blocker.box.length > target.y)
        z_above = blocker.z >= target.z + target.box.height

        return x_overlap and y_overlap and z_above

    def _is_blocking_exit_path(self, blocker: PackedBox, target: PackedBox, exit_y: float) -> bool:
        """출구 경로 차단 확인"""
        target_exit_path_start_y = target.y + target.box.length

        x_overlap = (blocker.x < target.x + target.box.width and
                    blocker.x + blocker.box.width > target.x)
        z_overlap = (blocker.z < target.z + target.box.height and
                    blocker.z + blocker.box.height > target.z)

        y_blocking = (blocker.y < exit_y and
                     blocker.y + blocker.box.length > target_exit_path_start_y)

        return x_overlap and z_overlap and y_blocking

    def _calculate_vehicle_routing_cost(self, total_route_cost: int, vehicle_boxes: int, total_boxes: int) -> int:
        """차량별 라우팅 비용 계산"""
        if total_boxes == 0:
            return 0

        if total_route_cost >= 999999:
            return 999999

        ratio = vehicle_boxes / total_boxes
        cost = total_route_cost * ratio

        if cost >= 999999:
            return 999999

        return int(cost)

# ========================= 통합 최적화 및 결과 출력 클래스 =========================

class IntegratedOptimizer:
    """라우팅과 적재를 통합 최적화하는 클래스"""

    def __init__(self, preprocessor: DataPreprocessor, clustering_manager: ClusteringManager,
                 routing_optimizer: RoutingOptimizer):
        self.preprocessor = preprocessor
        self.clustering_manager = clustering_manager
        self.routing_optimizer = routing_optimizer
        self.packing_optimizer = PackingOptimizer(preprocessor)
        self.final_vehicle_plans: List[VehiclePlan] = []

    def optimize_integrated_solution(self):
        """라우팅과 적재를 통합하여 최적화"""
        self.final_vehicle_plans = []
        global_vehicle_id = 0

        for route in self.routing_optimizer.routes:
            vehicle_plans = self.packing_optimizer.optimize_packing_for_route(route)

            for plan in vehicle_plans:
                plan.vehicle_id = global_vehicle_id
                global_vehicle_id += 1

            self.final_vehicle_plans.extend(vehicle_plans)

    def generate_output_file(self, filename: str = "Result.xlsx"):
        """결과를 Excel 파일로 출력"""
        output_data = []

        for plan in self.final_vehicle_plans:
            depot_start_row = {
                'Vehicle_ID': plan.vehicle_id,
                'Route_Order': 0,
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': '',
                'Latitude': '',
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            }
            output_data.append(depot_start_row)

            route_order = 1

            for dest in plan.route:
                dest_boxes = [pb for pb in plan.packed_boxes if pb.box.destination == dest]

                for packed_box in dest_boxes:
                    box = packed_box.box
                    dest_location = self.preprocessor.destinations[box.destination]

                    row_data = {
                        'Vehicle_ID': plan.vehicle_id,
                        'Route_Order': route_order,
                        'Destination': box.destination,
                        'Order_Number': box.order_number,
                        'Box_ID': box.box_id,
                        'Stacking_Order': packed_box.stacking_order,
                        'Lower_Left_X': round(packed_box.x, 2),
                        'Lower_Left_Y': round(packed_box.y, 2),
                        'Lower_Left_Z': round(packed_box.z, 2),
                        'Longitude': dest_location.longitude,
                        'Latitude': dest_location.latitude,
                        'Box_Width': round(box.width, 2),
                        'Box_Length': round(box.length, 2),
                        'Box_Height': round(box.height, 2)
                    }
                    output_data.append(row_data)

            route_order += 1

            depot_end_row = {
                'Vehicle_ID': plan.vehicle_id,
                'Route_Order': route_order,
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': '',
                'Latitude': '',
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            }
            output_data.append(depot_end_row)

        output_data.sort(key=lambda x: (
            x['Vehicle_ID'],
            x['Route_Order'],
            x['Stacking_Order'] if x['Stacking_Order'] != '' else -1
        ))

        try:
            import pandas as pd
            df = pd.DataFrame(output_data)

            column_order = [
                'Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
                'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
                'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'
            ]
            df = df[column_order]

            df.to_excel(filename, index=False)
        except ImportError:
            self._generate_csv_output(output_data, filename.replace('.xlsx', '.csv'))

    def _generate_csv_output(self, output_data: List[Dict], filename: str):
        """CSV 파일로 결과 출력"""
        import csv

        if not output_data:
            return

        fieldnames = [
            'Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
            'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
            'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in output_data:
                writer.writerow(row)

    def print_final_summary(self):
        """최종 요약 정보 출력"""
        total_routing_cost = sum(plan.routing_cost for plan in self.final_vehicle_plans)
        total_unloading_cost = sum(plan.unloading_cost for plan in self.final_vehicle_plans)
        total_cost = sum(plan.total_cost for plan in self.final_vehicle_plans)

        print(f"총 비용: {total_cost:,}원")

# ========================= 메인 시스템 클래스 =========================

class DeliveryOptimizationSystem:
    """전체 배송 최적화 시스템을 관리하는 클래스"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.clustering_manager = None
        self.routing_optimizer = None
        self.integrated_optimizer = None

    def run_optimization(self, data_file: str, distance_file: str):
        """전체 최적화 프로세스 실행"""
        self.preprocessor.load_data(data_file, distance_file)

        # 세밀 조정된 클러스터링 (82% 활용률)
        self.clustering_manager = ClusteringManager(self.preprocessor)
        self.clustering_manager.create_clusters(max_cluster_volume_ratio=0.82)

        self.routing_optimizer = RoutingOptimizer(self.preprocessor, self.clustering_manager)
        self.routing_optimizer.optimize_routes()

        self.integrated_optimizer = IntegratedOptimizer(
            self.preprocessor, self.clustering_manager, self.routing_optimizer
        )
        self.integrated_optimizer.optimize_integrated_solution()

        self.integrated_optimizer.generate_output_file("Result.xlsx")
        self.integrated_optimizer.print_final_summary()

# ========================= 실행 부분 =========================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    data_file = sys.argv[1]
    distance_file = sys.argv[2]

    system = DeliveryOptimizationSystem()
    system.run_optimization(data_file, distance_file)
