import json
import sys
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import math

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
        print("데이터 로딩 시작...")

        # JSON 데이터 로드
        self._load_json_data(data_file)

        # 거리 매트릭스 로드
        self._load_distance_matrix(distance_file)

        # 데이터 전처리
        self._preprocess_data()

        print(f"데이터 로딩 완료: {len(self.orders)}개 주문, {len(self.destinations)}개 목적지")

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

        # 첫 번째 줄에서 위치 ID들 추출
        header = lines[0].strip().split('\t')
        locations = header[1:]  # 첫 번째 열은 행 레이블이므로 제외

        # 거리 매트릭스 파싱
        for i, line in enumerate(lines[1:], 0):
            parts = line.strip().split('\t')
            if len(parts) <= 1:  # 빈 줄이나 잘못된 줄 건너뛰기
                continue

            from_location = parts[0]

            for j, distance_str in enumerate(parts[1:], 0):
                if j >= len(locations):  # 헤더보다 많은 데이터가 있는 경우 방지
                    break

                to_location = locations[j]

                # 숫자가 아닌 값 처리 (헤더 중복 등)
                try:
                    distance = int(distance_str)
                    self.distance_matrix[(from_location, to_location)] = distance
                except ValueError:
                    # 숫자가 아닌 경우 건너뛰기 (헤더 중복이나 잘못된 데이터)
                    continue

    def _preprocess_data(self):
        """데이터 전처리 수행"""
        print("데이터 전처리 시작...")

        # 1. 목적지별 주문 그룹화
        self._group_orders_by_destination()

        # 2. 박스 크기별 분류 및 통계
        self._analyze_box_sizes()

        # 3. 데이터 유효성 검증
        self._validate_data()

        print("데이터 전처리 완료")

    def _group_orders_by_destination(self):
        """목적지별로 주문을 그룹화"""
        for order in self.orders:
            self.orders_by_destination[order.destination].append(order)

        print(f"목적지별 주문 그룹화 완료: {len(self.orders_by_destination)}개 목적지")

    def _analyze_box_sizes(self):
        """박스 크기별 분석 및 통계"""
        size_groups = defaultdict(int)
        total_volume = 0

        for order in self.orders:
            # 박스 크기별 분류 (표준 박스 크기 기준)
            if (order.width, order.length, order.height) == (30, 40, 30):
                size_groups['small'] += 1
            elif (order.width, order.length, order.height) == (30, 50, 40):
                size_groups['medium'] += 1
            elif (order.width, order.length, order.height) == (50, 60, 50):
                size_groups['large'] += 1
            else:
                size_groups['custom'] += 1

            total_volume += order.volume

        print(f"박스 크기 분석:")
        print(f"  - 소형(30x40x30): {size_groups['small']}개")
        print(f"  - 중형(30x50x40): {size_groups['medium']}개")
        print(f"  - 대형(50x60x50): {size_groups['large']}개")
        print(f"  - 기타: {size_groups['custom']}개")
        print(f"  - 총 부피: {total_volume:,.0f} cm³")

        # 예상 필요 차량 수 계산
        estimated_vehicles = math.ceil(total_volume / self.vehicle.max_volume)
        print(f"  - 예상 필요 차량 수: {estimated_vehicles}대")

    def _validate_data(self):
        """데이터 유효성 검증"""
        errors = []

        # 1. 모든 주문의 목적지가 destinations에 존재하는지 확인
        for order in self.orders:
            if order.destination not in self.destinations:
                errors.append(f"주문 {order.order_number}: 존재하지 않는 목적지 {order.destination}")

        # 2. 박스 크기가 차량 적재함보다 큰 경우 확인
        for order in self.orders:
            if (order.width > self.vehicle.max_width or
                order.length > self.vehicle.max_length or
                order.height > self.vehicle.max_height):
                errors.append(f"주문 {order.order_number}: 박스가 차량 적재함보다 큼")

        if errors:
            print("데이터 유효성 검증 오류:")
            for error in errors[:10]:  # 처음 10개만 출력
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... 외 {len(errors) - 10}개 오류")
        else:
            print("데이터 유효성 검증 통과")

    def get_distance(self, from_location: str, to_location: str) -> int:
        """두 위치 간의 거리를 반환"""
        distance = self.distance_matrix.get((from_location, to_location), None)
        if distance is None:
            # 거리 정보가 없는 경우 큰 값으로 설정 (하지만 무한대는 아님)
            return 999999
        return distance

# ========================= 클러스터링 클래스 =========================

class ClusteringManager:
    """클러스터링을 담당하는 클래스"""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.clusters: List[Cluster] = []
        self.destination_to_cluster: Dict[str, int] = {}
        self.vehicle_capacity = preprocessor.vehicle.max_volume

    def create_clusters(self, max_cluster_volume_ratio: float = 0.8):
        """클러스터 생성 메인 함수"""
        print("클러스터링 시작...")

        # 1. 목적지별 주문 분석
        dest_info = self._analyze_destinations()

        # 2. 지리적 근접성 기반 초기 클러스터링
        initial_clusters = self._geographic_clustering(dest_info)

        # 3. 부피 제약 조건 적용
        volume_adjusted_clusters = self._adjust_for_volume_constraints(
            initial_clusters, max_cluster_volume_ratio
        )

        # 4. 박스 크기 분포 최적화
        final_clusters = self._optimize_box_distribution(volume_adjusted_clusters)

        # 5. 클러스터 정보 저장
        self._finalize_clusters(final_clusters)

        print(f"클러스터링 완료: {len(self.clusters)}개 클러스터 생성")

    def _analyze_destinations(self) -> Dict[str, Dict]:
        """목적지별 주문 정보 분석"""
        dest_info = {}

        for dest_id, orders in self.preprocessor.orders_by_destination.items():
            location = self.preprocessor.destinations[dest_id]
            total_volume = sum(order.volume for order in orders)

            # 박스 크기별 분류
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
        """목적지 우선순위 계산 (높을수록 우선)"""
        order_count = len(orders)
        volume_score = total_volume / 100000  # 정규화

        # 대형 박스가 많을수록 높은 우선순위 (적재 효율성)
        large_box_ratio = sum(1 for order in orders
                             if (order.width, order.length, order.height) == (50, 60, 50)) / order_count

        return order_count * 0.4 + volume_score * 0.4 + large_box_ratio * 0.2

    def _geographic_clustering(self, dest_info: Dict) -> List[List[str]]:
        """지리적 근접성 기반 클러스터링"""
        print("  지리적 클러스터링 수행...")

        unassigned = set(dest_info.keys())
        clusters = []

        while unassigned:
            # 가장 높은 우선순위의 목적지를 클러스터 중심으로 선택
            center_dest = max(unassigned, key=lambda d: dest_info[d]['priority'])
            cluster = [center_dest]
            unassigned.remove(center_dest)

            center_location = dest_info[center_dest]['location']

            # 중심점 주변의 근접한 목적지들을 클러스터에 추가
            candidates = list(unassigned)
            candidates.sort(key=lambda d: self._calculate_distance(
                center_location.latitude, center_location.longitude,
                dest_info[d]['location'].latitude, dest_info[d]['location'].longitude
            ))

            cluster_volume = dest_info[center_dest]['total_volume']
            max_cluster_volume = self.vehicle_capacity * 0.7  # 여유 공간 확보

            for candidate in candidates:
                candidate_volume = dest_info[candidate]['total_volume']

                # 부피 제약과 거리 제약 확인
                if (cluster_volume + candidate_volume <= max_cluster_volume and
                    len(cluster) < 8):  # 한 클러스터당 최대 8개 목적지

                    # 거리 제약 확인 (클러스터 중심에서 너무 멀지 않은지)
                    distance = self._calculate_distance(
                        center_location.latitude, center_location.longitude,
                        dest_info[candidate]['location'].latitude, dest_info[candidate]['location'].longitude
                    )

                    if distance <= 0.05:  # 약 5km 이내 (위경도 차이 기준)
                        cluster.append(candidate)
                        cluster_volume += candidate_volume
                        unassigned.remove(candidate)

            clusters.append(cluster)

        print(f"    초기 클러스터 {len(clusters)}개 생성")
        return clusters

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 위치 간의 유클리드 거리 계산"""
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    def _adjust_for_volume_constraints(self, clusters: List[List[str]],
                                     max_ratio: float) -> List[List[str]]:
        """부피 제약 조건에 따른 클러스터 조정"""
        print("  부피 제약 조건 적용...")

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

        print(f"    부피 조정 후 클러스터 {len(adjusted_clusters)}개")
        return adjusted_clusters

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

    def _optimize_box_distribution(self, clusters: List[List[str]]) -> List[List[str]]:
        """박스 크기 분포 최적화"""
        print("  박스 크기 분포 최적화...")

        cluster_scores = []
        for i, cluster_destinations in enumerate(clusters):
            large_box_count = 0
            total_orders = 0

            for dest in cluster_destinations:
                orders = self.preprocessor.orders_by_destination[dest]
                total_orders += len(orders)
                large_box_count += sum(1 for order in orders
                                     if (order.width, order.length, order.height) == (50, 60, 50))

            large_box_ratio = large_box_count / max(total_orders, 1)
            cluster_scores.append((i, large_box_ratio))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        optimized_clusters = [clusters[i] for i, _ in cluster_scores]

        return optimized_clusters

    def _finalize_clusters(self, cluster_destinations: List[List[str]]):
        """최종 클러스터 정보 생성 및 저장"""
        self.clusters = []

        for i, destinations in enumerate(cluster_destinations):
            all_orders = []
            for dest in destinations:
                all_orders.extend(self.preprocessor.orders_by_destination[dest])

            # 클러스터 중심점 계산
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

# ========================= 라우팅 최적화 클래스 =========================

class RoutingOptimizer:
    """라우팅 최적화를 담당하는 클래스"""

    def __init__(self, preprocessor: DataPreprocessor, clustering_manager: ClusteringManager):
        self.preprocessor = preprocessor
        self.clustering_manager = clustering_manager
        self.routes: List[Route] = []

    def optimize_routes(self):
        """모든 클러스터에 대해 라우팅 최적화 수행"""
        print("라우팅 최적화 시작...")

        self.routes = []
        for cluster in self.clustering_manager.clusters:
            if len(cluster.destinations) == 1:
                # 목적지가 하나인 경우 간단한 경로
                route = self._create_simple_route(cluster)
            else:
                # 여러 목적지인 경우 TSP 최적화
                route = self._solve_tsp_for_cluster(cluster)

            self.routes.append(route)

        print(f"라우팅 최적화 완료: {len(self.routes)}개 경로 생성")

    def _create_simple_route(self, cluster: Cluster) -> Route:
        """단일 목적지 클러스터에 대한 간단한 경로 생성"""
        destination = cluster.destinations[0]

        # Depot -> 목적지 -> Depot
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
            # 목적지가 적은 경우 완전탐색
            best_route = self._brute_force_tsp(destinations)
        else:
            # 목적지가 많은 경우 휴리스틱 알고리즘
            best_route = self._nearest_neighbor_tsp(destinations)
            # 2-opt 개선 적용
            best_route = self._two_opt_improvement(best_route)

        # 총 거리 및 비용 계산
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

        # 모든 순열에 대해 거리 계산
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

        # 시작점을 depot에서 가장 가까운 목적지로 선택
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
            # 현재 위치에서 가장 가까운 미방문 목적지 선택
            next_dest = None
            min_distance = 999999

            for dest in unvisited:
                distance = self.preprocessor.get_distance(current, dest)
                if distance < min_distance:
                    min_distance = distance
                    next_dest = dest

            if next_dest is None:  # 모든 거리가 무한대인 경우
                next_dest = list(unvisited)[0]  # 첫 번째 목적지 선택

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
                    # 2-opt 스왑 수행
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
        """경로의 총 거리 계산 (Depot 시작/종료 포함)"""
        if not destinations:
            return 0

        total_distance = 0

        # Depot에서 첫 번째 목적지
        distance = self.preprocessor.get_distance("Depot", destinations[0])
        if distance >= 999999:  # 거리 정보가 없는 경우
            return 999999
        total_distance += distance

        # 목적지 간 이동
        for i in range(len(destinations) - 1):
            distance = self.preprocessor.get_distance(destinations[i], destinations[i+1])
            if distance >= 999999:  # 거리 정보가 없는 경우
                return 999999
            total_distance += distance

        # 마지막 목적지에서 Depot으로 복귀
        distance = self.preprocessor.get_distance(destinations[-1], "Depot")
        if distance >= 999999:  # 거리 정보가 없는 경우
            return 999999
        total_distance += distance

        return total_distance

    def print_routing_summary(self):
        """라우팅 요약 정보 출력"""
        print("\n=== 라우팅 최적화 요약 ===")
        total_cost = 0
        total_distance = 0

        for route in self.routes:
            print(f"클러스터 {route.cluster_id}:")
            print(f"  - 목적지 순서: {' -> '.join(route.destinations)}")
            print(f"  - 총 거리: {route.total_distance} km")
            print(f"  - 경로 비용: {route.route_cost:,}원")
            total_cost += route.route_cost
            total_distance += route.total_distance

# ========================= 적재 최적화 클래스 =========================

class PackingOptimizer:
    """3D 빈 패킹 및 적재 최적화를 담당하는 클래스"""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.vehicle = preprocessor.vehicle
        self.shuffling_cost = 500  # 셔플링 비용 500원

    def optimize_packing_for_route(self, route: Route) -> List[VehiclePlan]:
        """경로에 대한 적재 최적화 수행"""
        cluster = None
        for c in self.preprocessor.orders_by_destination:
            if any(dest in route.destinations for dest in [c]):
                cluster = c
                break

        # 경로의 모든 주문 수집
        all_orders = []
        for dest in route.destinations:
            all_orders.extend(self.preprocessor.orders_by_destination[dest])

        # 배송 순서 기반으로 적재 계획 생성
        vehicle_plans = self._create_vehicle_plans(route, all_orders)

        return vehicle_plans

    def _create_vehicle_plans(self, route: Route, orders: List[Box]) -> List[VehiclePlan]:
        """차량별 적재 계획 생성 (셔플링 최소화 고려)"""
        vehicle_plans = []

        # 목적지별 주문 그룹화
        orders_by_dest = defaultdict(list)
        for order in orders:
            orders_by_dest[order.destination].append(order)

        # 배송 순서에 따른 최적 적재 순서 결정 (LIFO + 셔플링 최소화)
        optimized_loading_order = self._optimize_loading_order(route.destinations, orders_by_dest)

        current_vehicle_boxes = []
        current_volume = 0
        vehicle_id = 0

        for order in optimized_loading_order:
            # 현재 차량에 적재 가능한지 확인
            if current_volume + order.volume <= self.vehicle.max_volume:
                current_vehicle_boxes.append(order)
                current_volume += order.volume
            else:
                # 현재 차량으로 적재 계획 생성
                if current_vehicle_boxes:
                    plan = self._create_single_vehicle_plan(
                        vehicle_id, route, current_vehicle_boxes
                    )
                    vehicle_plans.append(plan)
                    vehicle_id += 1

                # 새 차량 시작
                current_vehicle_boxes = [order]
                current_volume = order.volume

        # 마지막 차량 처리
        if current_vehicle_boxes:
            plan = self._create_single_vehicle_plan(
                vehicle_id, route, current_vehicle_boxes
            )
            vehicle_plans.append(plan)

        return vehicle_plans

    def _optimize_loading_order(self, destinations: List[str], orders_by_dest: Dict[str, List[Box]]) -> List[Box]:
        """셔플링을 최소화하는 적재 순서 최적화"""
        # LIFO 원칙: 마지막에 배송할 것을 먼저 적재
        # 추가 최적화: 같은 목적지 내에서 큰 박스를 아래(먼저), 작은 박스를 위에(나중에)

        loading_order = []
        reversed_destinations = destinations[::-1]  # 배송 순서의 역순

        for dest in reversed_destinations:
            dest_orders = orders_by_dest.get(dest, [])
            if not dest_orders:
                continue

            # 같은 목적지 내에서 크기별 최적화
            # 1. 대형 박스를 먼저 (바닥에 배치)
            # 2. 소형 박스를 나중에 (위쪽에 배치)
            dest_orders_sorted = sorted(dest_orders, key=lambda box: (
                -box.volume,  # 부피 큰 것부터 (음수로 역순)
                -box.height,  # 높이 큰 것부터
                -box.width * box.length  # 바닥 면적 큰 것부터
            ))

            loading_order.extend(dest_orders_sorted)

        return loading_order

    def _create_single_vehicle_plan(self, vehicle_id: int, route: Route,
                                   boxes: List[Box]) -> VehiclePlan:
        """단일 차량에 대한 적재 계획 생성"""
        # 3D 빈 패킹 수행
        packed_boxes = self._pack_boxes_3d(boxes)

        # 하차 비용 계산 (셔플링 비용)
        unloading_cost = self._calculate_unloading_cost(packed_boxes, route.destinations)

        # 이 차량이 방문해야 할 목적지만 필터링
        vehicle_destinations = []
        box_destinations = {box.destination for box in boxes}
        for dest in route.destinations:
            if dest in box_destinations:
                vehicle_destinations.append(dest)

        # 차량별 라우팅 비용 계산 (비례 배분)
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

    def _pack_boxes_3d(self, boxes: List[Box]) -> List[PackedBox]:
        """개선된 3D 빈 패킹 알고리즘"""
        packed_boxes = []

        # 박스를 안정성을 고려하여 정렬
        # 1. 바닥 면적이 큰 것부터 (안정적인 기반)
        # 2. 부피가 큰 것부터
        # 3. 높이가 낮은 것부터 (안정적인 적재)
        sorted_boxes = sorted(boxes, key=lambda b: (
            -(b.width * b.length),  # 바닥 면적 큰 것부터
            -b.volume,              # 부피 큰 것부터
            b.height,               # 높이 낮은 것부터
            b.box_id                # 일관된 정렬을 위한 ID
        ))

        # 이미 배치된 박스들의 위치 추적 (더 정확한 추적)
        occupied_spaces = []

        for i, box in enumerate(sorted_boxes):
            # 최적 위치 찾기
            position = self._find_best_position(box, occupied_spaces)

            # 위치 유효성 재확인
            if not self._validate_position(position, box, occupied_spaces):
                print(f"경고: 박스 {box.box_id}의 위치 ({position})가 유효하지 않습니다.")
                # 강제로 빈 공간 찾기
                position = self._find_safe_position(box, occupied_spaces)

            packed_box = PackedBox(
                box=box,
                x=position[0],
                y=position[1],
                z=position[2],
                stacking_order=i
            )
            packed_boxes.append(packed_box)

            # 점유 공간 정확히 추가
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

            # 배치 후 겹침 재검증
            if self._has_overlaps(occupied_spaces):
                print(f"경고: 박스 {box.box_id} 배치 후 겹침 발생!")

        return packed_boxes

    def _validate_position(self, position: Tuple[float, float, float], box: Box,
                          occupied_spaces: List[Dict]) -> bool:
        """위치 유효성 검증"""
        x, y, z = position

        # 차량 경계 확인
        if (x + box.width > self.vehicle.max_width or
            y + box.length > self.vehicle.max_length or
            z + box.height > self.vehicle.max_height):
            return False

        # 겹침 확인
        if self._check_overlap_strict(x, y, z, box, occupied_spaces):
            return False

        return True

    def _find_safe_position(self, box: Box, occupied_spaces: List[Dict]) -> Tuple[float, float, float]:
        """안전한 위치 강제 탐색 (최후의 수단) - cm 단위"""
        max_width = self.vehicle.max_width   # 160cm
        max_length = self.vehicle.max_length # 280cm
        max_height = self.vehicle.max_height # 180cm

        # 더 작은 단위로 전체 공간 스캔 (cm 단위)
        step = 5  # 5cm 단위

        for z in range(0, int(max_height - box.height + 1), step):
            for y in range(0, int(max_length - box.length + 1), step):
                for x in range(0, int(max_width - box.width + 1), step):
                    if not self._check_overlap_strict(x, y, z, box, occupied_spaces):
                        return (float(x), float(y), float(z))

        # 정말 공간이 없다면 위쪽으로 쌓기
        if occupied_spaces:
            max_z = max(space['z2'] for space in occupied_spaces)
            return (0.0, 0.0, float(max_z))

        return (0.0, 0.0, 0.0)

    def _has_overlaps(self, occupied_spaces: List[Dict]) -> bool:
        """전체 배치에서 겹침이 있는지 확인"""
        for i in range(len(occupied_spaces)):
            for j in range(i + 1, len(occupied_spaces)):
                space1 = occupied_spaces[i]
                space2 = occupied_spaces[j]

                x_overlap = space1['x1'] < space2['x2'] and space1['x2'] > space2['x1']
                y_overlap = space1['y1'] < space2['y2'] and space1['y2'] > space2['y1']
                z_overlap = space1['z1'] < space2['z2'] and space1['z2'] > space2['z1']

                if x_overlap and y_overlap and z_overlap:
                    print(f"겹침 발견: {space1['box_id']} ↔ {space2['box_id']}")
                    return True

        return False

    def _find_best_position(self, box: Box, occupied_spaces: List[Dict]) -> Tuple[float, float, float]:
        """박스에 대한 최적 위치 찾기 (cm 단위 유지)"""
        max_width = self.vehicle.max_width   # 160cm
        max_length = self.vehicle.max_length # 280cm
        max_height = self.vehicle.max_height # 180cm

        # 격자 단위로 위치 후보 생성 (cm 단위로 탐색)
        step_size = 10  # 10cm 단위로 탐색
        candidates = []

        # 바닥부터 시작
        for z in range(0, int(max_height), step_size):
            for y in range(0, int(max_length), step_size):
                for x in range(0, int(max_width), step_size):
                    # 박스가 차량 경계를 넘지 않는지 확인
                    if (x + box.width <= max_width and
                        y + box.length <= max_length and
                        z + box.height <= max_height):

                        # 다른 박스와 겹치지 않는지 확인
                        if not self._check_overlap_strict(x, y, z, box, occupied_spaces):
                            candidates.append((x, y, z))

        if not candidates:
            # 격자 탐색으로 위치를 찾지 못한 경우, 기존 박스 모서리 기반 탐색
            candidates = self._find_corner_positions(box, occupied_spaces, max_width, max_length, max_height)

        if not candidates:
            print(f"경고: 박스 {box.box_id}에 대한 유효한 위치를 찾을 수 없습니다. 기본 위치 (0,0,0) 사용")
            return (0.0, 0.0, 0.0)

        # 가장 아래쪽, 뒤쪽, 왼쪽 우선으로 정렬 (안정적인 적재)
        candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        return (float(candidates[0][0]), float(candidates[0][1]), float(candidates[0][2]))

    def _find_corner_positions(self, box: Box, occupied_spaces: List[Dict],
                              max_width: float, max_length: float, max_height: float) -> List[Tuple[float, float, float]]:
        """기존 박스 모서리 기반 위치 후보 생성"""
        candidates = [(0, 0, 0)]  # 바닥 모서리

        # 기존 박스들의 모서리 위치들을 후보로 추가
        for space in occupied_spaces:
            edge_positions = [
                (space['x2'], space['y1'], space['z1']),  # 오른쪽
                (space['x1'], space['y2'], space['z1']),  # 뒤쪽
                (space['x1'], space['y1'], space['z2']),  # 위쪽
                (space['x2'], space['y2'], space['z1']),  # 오른쪽 뒤
                (space['x2'], space['y1'], space['z2']),  # 오른쪽 위
                (space['x1'], space['y2'], space['z2']),  # 뒤쪽 위
                (space['x2'], space['y2'], space['z2']),  # 오른쪽 뒤 위
            ]
            candidates.extend(edge_positions)

        # 유효한 위치만 필터링
        valid_candidates = []
        for x, y, z in candidates:
            if (x + box.width <= max_width and
                y + box.length <= max_length and
                z + box.height <= max_height):

                if not self._check_overlap_strict(x, y, z, box, occupied_spaces):
                    valid_candidates.append((x, y, z))

        return valid_candidates

    def _check_overlap_strict(self, x: float, y: float, z: float, box: Box,
                             occupied_spaces: List[Dict]) -> bool:
        """엄격한 겹침 검사 (부동소수점 오차 고려)"""
        box_x2 = x + box.width
        box_y2 = y + box.length
        box_z2 = z + box.height

        tolerance = 0.01  # 1cm 허용 오차

        for space in occupied_spaces:
            # 겹침 조건: 모든 축에서 겹치는 경우
            x_overlap = (box_x2 > space['x1'] + tolerance and
                        x < space['x2'] - tolerance)
            y_overlap = (box_y2 > space['y1'] + tolerance and
                        y < space['y2'] - tolerance)
            z_overlap = (box_z2 > space['z1'] + tolerance and
                        z < space['z2'] - tolerance)

            if x_overlap and y_overlap and z_overlap:
                return True

        return False

    def _calculate_unloading_cost(self, packed_boxes: List[PackedBox],
                                 delivery_order: List[str]) -> int:
        """하차 비용 계산 (셔플링 횟수 기반)"""
        total_shuffling = 0

        # 배송 순서대로 하차 시뮬레이션
        remaining_boxes = packed_boxes[:]

        for dest in delivery_order:
            # 해당 목적지의 박스들 찾기
            dest_boxes = [pb for pb in remaining_boxes if pb.box.destination == dest]

            for target_box in dest_boxes:
                # 이 박스를 출구로 꺼내기 위해 옮겨야 할 박스 수 계산
                shuffling_count = self._calculate_shuffling_for_box(target_box, remaining_boxes)
                total_shuffling += shuffling_count
                remaining_boxes.remove(target_box)

        return total_shuffling * self.shuffling_cost

    def _calculate_shuffling_for_box(self, target_box: PackedBox, remaining_boxes: List[PackedBox]) -> int:
        """특정 박스를 꺼내기 위한 셔플링 횟수 계산"""
        shuffling_count = 0

        # 출구는 Y=280cm (차량 뒤쪽)
        truck_exit_y = self.vehicle.max_length  # 280cm

        for other_box in remaining_boxes:
            if other_box == target_box:
                continue

            # 1. 위에 적재된 박스 체크 (Z축 방향)
            if self._is_box_above(other_box, target_box):
                shuffling_count += 1
                continue

            # 2. 출구 방향으로 직선 경로를 가로막는 박스 체크 (Y축 방향)
            if self._is_blocking_exit_path(other_box, target_box, truck_exit_y):
                shuffling_count += 1

        return shuffling_count

    def _is_box_above(self, blocker: PackedBox, target: PackedBox) -> bool:
        """blocker가 target 위에 적재되어 있는지 확인"""
        # XY 평면에서 겹치고, blocker가 target보다 위에 있는 경우
        x_overlap = (blocker.x < target.x + target.box.width and
                    blocker.x + blocker.box.width > target.x)
        y_overlap = (blocker.y < target.y + target.box.length and
                    blocker.y + blocker.box.length > target.y)
        z_above = blocker.z >= target.z + target.box.height

        return x_overlap and y_overlap and z_above

    def _is_blocking_exit_path(self, blocker: PackedBox, target: PackedBox, exit_y: float) -> bool:
        """blocker가 target의 출구 방향 직선 경로를 막고 있는지 확인"""
        # target이 출구로 나가는 직선 경로 (Y축 방향)
        target_exit_path_start_y = target.y + target.box.length

        # blocker가 target의 출구 경로에 있는지 확인
        # 1. XZ 좌표가 겹치는지 확인
        x_overlap = (blocker.x < target.x + target.box.width and
                    blocker.x + blocker.box.width > target.x)
        z_overlap = (blocker.z < target.z + target.box.height and
                    blocker.z + blocker.box.height > target.z)

        # 2. Y축에서 target과 출구 사이에 있는지 확인
        y_blocking = (blocker.y < exit_y and
                     blocker.y + blocker.box.length > target_exit_path_start_y)

        return x_overlap and z_overlap and y_blocking

    def _calculate_vehicle_routing_cost(self, total_route_cost: int,
                                      vehicle_boxes: int, total_boxes: int) -> int:
        """차량별 라우팅 비용 계산 (박스 수 비례)"""
        if total_boxes == 0:
            return 0

        # 무한대나 너무 큰 값 처리
        if total_route_cost >= 999999:
            return 999999

        ratio = vehicle_boxes / total_boxes
        cost = total_route_cost * ratio

        # 결과가 너무 크지 않은지 확인
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
        print("통합 최적화 시작...")

        self.final_vehicle_plans = []
        global_vehicle_id = 0  # 전역 차량 ID 카운터

        for route in self.routing_optimizer.routes:
            # 각 경로에 대해 적재 최적화 수행
            vehicle_plans = self.packing_optimizer.optimize_packing_for_route(route)

            # 전역 차량 ID 할당
            for plan in vehicle_plans:
                plan.vehicle_id = global_vehicle_id
                global_vehicle_id += 1

            self.final_vehicle_plans.extend(vehicle_plans)

        print(f"통합 최적화 완료: {len(self.final_vehicle_plans)}대 차량 계획 생성")

    def generate_output_file(self, filename: str = "Result.xlsx"):
        """결과를 Excel 파일로 출력"""
        print(f"결과 파일 생성: {filename}")

        # Excel 출력을 위한 데이터 준비
        output_data = []

        for plan in self.final_vehicle_plans:
            # 차량 시작 - Depot 행 추가 (차량당 한 번만)
            depot_start_row = {
                'Vehicle_ID': plan.vehicle_id,
                'Route_Order': 0,  # Depot은 0번으로 시작
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': '',  # 좌표 정보 제거
                'Latitude': '',   # 좌표 정보 제거
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            }
            output_data.append(depot_start_row)

            # 차량의 배송 순서대로 목적지 정렬
            route_order = 1

            for dest in plan.route:
                # 해당 목적지의 박스들 찾기
                dest_boxes = [pb for pb in plan.packed_boxes if pb.box.destination == dest]

                for packed_box in dest_boxes:
                    box = packed_box.box
                    dest_location = self.preprocessor.destinations[box.destination]

                    # 프로젝트 요구사항에 맞는 컬럼 형식으로 데이터 생성
                    # cm 단위 그대로 유지 (변환하지 않음)
                    row_data = {
                        'Vehicle_ID': plan.vehicle_id,
                        'Route_Order': route_order,
                        'Destination': box.destination,
                        'Order_Number': box.order_number,
                        'Box_ID': box.box_id,
                        'Stacking_Order': packed_box.stacking_order,
                        'Lower_Left_X': round(packed_box.x, 2),  # cm 단위 유지
                        'Lower_Left_Y': round(packed_box.y, 2),  # cm 단위 유지
                        'Lower_Left_Z': round(packed_box.z, 2),  # cm 단위 유지
                        'Longitude': dest_location.longitude,
                        'Latitude': dest_location.latitude,
                        'Box_Width': round(box.width, 2),   # cm 단위 유지
                        'Box_Length': round(box.length, 2), # cm 단위 유지
                        'Box_Height': round(box.height, 2)  # cm 단위 유지
                    }
                    output_data.append(row_data)

                route_order += 1

            # 차량 종료 - Depot 행 추가 (차량당 한 번만)
            depot_end_row = {
                'Vehicle_ID': plan.vehicle_id,
                'Route_Order': route_order,  # 마지막 배송 후 Depot 복귀
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': '',  # 좌표 정보 제거
                'Latitude': '',   # 좌표 정보 제거
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            }
            output_data.append(depot_end_row)

        # 정렬: Vehicle_ID -> Route_Order -> Stacking_Order 순으로
        output_data.sort(key=lambda x: (
            x['Vehicle_ID'],
            x['Route_Order'],
            x['Stacking_Order'] if x['Stacking_Order'] != '' else -1
        ))

        # pandas를 사용하여 Excel 파일 생성
        try:
            import pandas as pd
            df = pd.DataFrame(output_data)

            # 컬럼 순서를 프로젝트 요구사항에 맞게 조정
            column_order = [
                'Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
                'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
                'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'
            ]
            df = df[column_order]

            df.to_excel(filename, index=False)
            print(f"결과 파일 생성 완료: {filename}")
        except ImportError:
            print("pandas가 설치되어 있지 않습니다. CSV 파일로 출력합니다.")
            self._generate_csv_output(output_data, filename.replace('.xlsx', '.csv'))

    def _generate_csv_output(self, output_data: List[Dict], filename: str):
        """CSV 파일로 결과 출력"""
        import csv

        if not output_data:
            return

        # 컬럼 순서를 프로젝트 요구사항에 맞게 설정
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

        print(f"CSV 파일 생성 완료: {filename}")

    def print_final_summary(self):
        """최종 요약 정보 출력"""
        print("\n=== 최종 통합 최적화 결과 ===")

        total_routing_cost = sum(plan.routing_cost for plan in self.final_vehicle_plans)
        total_unloading_cost = sum(plan.unloading_cost for plan in self.final_vehicle_plans)
        total_cost = sum(plan.total_cost for plan in self.final_vehicle_plans)

        print(f"사용된 차량 수: {len(self.final_vehicle_plans)}대")
        print(f"총 라우팅 비용: {total_routing_cost:,}원")
        print(f"총 하차 비용: {total_unloading_cost:,}원")
        print(f"총 비용: {total_cost:,}원")

        # 차량별 상세 정보
        print("\n=== 차량별 상세 정보 ===")
        for plan in self.final_vehicle_plans:
            print(f"차량 {plan.vehicle_id}:")
            print(f"  - 클러스터: {plan.cluster_id}")
            print(f"  - 방문 목적지: {len(plan.route)}개")
            print(f"  - 적재 박스: {len(plan.packed_boxes)}개")
            print(f"  - 적재 부피: {plan.total_volume:,.0f} cm³ ({plan.total_volume/self.preprocessor.vehicle.max_volume*100:.1f}%)")
            print(f"  - 라우팅 비용: {plan.routing_cost:,}원")
            print(f"  - 하차 비용: {plan.unloading_cost:,}원")
            print(f"  - 총 비용: {plan.total_cost:,}원")

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
        print("=== 배송 최적화 시스템 시작 ===\n")

        # 1. 데이터 입력 및 전처리
        self.preprocessor.load_data(data_file, distance_file)

        # 2. 클러스터링
        self.clustering_manager = ClusteringManager(self.preprocessor)
        self.clustering_manager.create_clusters()

        # 3. 라우팅 최적화
        self.routing_optimizer = RoutingOptimizer(self.preprocessor, self.clustering_manager)
        self.routing_optimizer.optimize_routes()

        # 4. 통합 최적화 (라우팅 + 적재)
        self.integrated_optimizer = IntegratedOptimizer(
            self.preprocessor, self.clustering_manager, self.routing_optimizer
        )
        self.integrated_optimizer.optimize_integrated_solution()

        # 5. 결과 출력
        self.integrated_optimizer.generate_output_file("Result.xlsx")
        self.integrated_optimizer.print_final_summary()

        print("\n=== 배송 최적화 시스템 완료 ===")

# ========================= 실행 부분 =========================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python main.py data.json distance-data.txt")
        sys.exit(1)

    data_file = sys.argv[1]
    distance_file = sys.argv[2]

    # 배송 최적화 시스템 실행
    system = DeliveryOptimizationSystem()
    system.run_optimization(data_file, distance_file)
