import json
import sys
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
from itertools import combinations
import math

class Box:
    def __init__(self, order_number, box_id, destination, width, length, height):
        self.order_number = order_number
        self.box_id = box_id
        self.destination = destination
        self.width = width
        self.length = length
        self.height = height
        self.volume = width * length * height

class Vehicle:
    def __init__(self, width=160, length=280, height=180):
        self.width = width
        self.length = length
        self.height = height
        self.volume = width * length * height

class LoadingOptimizer:
    def __init__(self, vehicle_capacity):
        self.vehicle = Vehicle(vehicle_capacity['width'], vehicle_capacity['length'], vehicle_capacity['height'])

    def can_fit_boxes(self, boxes):
        """개선된 적재 가능성 검사"""
        total_volume = sum(box.volume for box in boxes)
        if total_volume > self.vehicle.volume:
            return False

        # 차원별 제약도 확인
        max_width = max(box.width for box in boxes)
        max_length = max(box.length for box in boxes)
        max_height = max(box.height for box in boxes)

        if (max_width > self.vehicle.width or
            max_length > self.vehicle.length or
            max_height > self.vehicle.height):
            return False

        return True

    def calculate_loading_positions(self, boxes):
        """박스들의 적재 위치를 계산 (개선된 3D 패킹)"""
        if not boxes:
            return []

        positions = []

        # 박스를 부피 순으로 정렬 (큰 것부터)
        sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)

        # 적재 공간 추적
        occupied_spaces = []

        for i, box in enumerate(sorted_boxes):
            # 적재 가능한 위치 찾기
            position = self._find_best_position(box, occupied_spaces)
            if position is None:
                # 적재 불가능한 경우 간단한 층별 적재로 시도
                return self._simple_layer_packing(boxes)

            positions.append({
                'box_id': box.box_id,
                'x': position[0],
                'y': position[1],
                'z': position[2],
                'stacking_order': i + 1
            })

            # 점유 공간 추가
            occupied_spaces.append({
                'x1': position[0], 'y1': position[1], 'z1': position[2],
                'x2': position[0] + box.width,
                'y2': position[1] + box.length,
                'z2': position[2] + box.height
            })

        return positions

    def _find_best_position(self, box, occupied_spaces):
        """박스에 대한 최적 위치 찾기 (더 세밀한 탐색)"""
        # 더 세밀한 단위로 위치 탐색 (5cm 단위)
        step = 5

        # 바닥부터 시작하여 층별로 탐색
        for z in range(0, self.vehicle.height - box.height + 1, step):
            for y in range(0, self.vehicle.length - box.length + 1, step):
                for x in range(0, self.vehicle.width - box.width + 1, step):
                    if self._is_position_valid(x, y, z, box, occupied_spaces):
                        return (x, y, z)
        return None

    def _is_position_valid(self, x, y, z, box, occupied_spaces):
        """위치가 유효한지 확인"""
        # 차량 경계 확인
        if (x + box.width > self.vehicle.width or
            y + box.length > self.vehicle.length or
            z + box.height > self.vehicle.height):
            return False

        # 다른 박스와 겹치는지 확인
        for space in occupied_spaces:
            if not (x + box.width <= space['x1'] or x >= space['x2'] or
                   y + box.length <= space['y1'] or y >= space['y2'] or
                   z + box.height <= space['z1'] or z >= space['z2']):
                return False

        return True

    def _simple_layer_packing(self, boxes):
        """개선된 층별 적재 (높이 제약 고려)"""
        positions = []

        # 박스를 높이와 부피를 모두 고려하여 정렬
        # 높이가 낮고, 면적이 큰 박스를 우선 배치
        sorted_boxes = sorted(boxes, key=lambda b: (b.height, -b.width * b.length))

        # 현재 층 정보
        layers = []  # 각 층의 정보를 저장

        for i, box in enumerate(sorted_boxes):
            placed = False

            # 기존 층에 배치 시도
            for layer_idx, layer in enumerate(layers):
                if layer['current_height'] + box.height <= self.vehicle.height:
                    position = self._find_position_in_layer_improved(
                        box, layer['positions'], layer['z'])

                    if position:
                        positions.append({
                            'box_id': box.box_id,
                            'x': position[0],
                            'y': position[1],
                            'z': position[2],
                            'stacking_order': i + 1
                        })

                        layer['positions'].append({
                            'x1': position[0], 'y1': position[1],
                            'x2': position[0] + box.width,
                            'y2': position[1] + box.length,
                            'height': box.height
                        })

                        # 층 높이 업데이트
                        if position[2] + box.height > layer['current_height']:
                            layer['current_height'] = position[2] + box.height

                        placed = True
                        break

            # 기존 층에 배치할 수 없으면 새 층 생성
            if not placed:
                # 새 층의 z 좌표 계산
                new_z = max([layer['current_height'] for layer in layers]) if layers else 0

                if new_z + box.height <= self.vehicle.height:
                    position = (0, 0, new_z)

                    positions.append({
                        'box_id': box.box_id,
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'stacking_order': i + 1
                    })

                    # 새 층 추가
                    layers.append({
                        'z': new_z,
                        'current_height': new_z + box.height,
                        'positions': [{
                            'x1': position[0], 'y1': position[1],
                            'x2': position[0] + box.width,
                            'y2': position[1] + box.length,
                            'height': box.height
                        }]
                    })
                    placed = True

            if not placed:
                print(f"박스 {box.box_id} 적재 실패: 높이 제약")
                break

        print(f"적재 완료: {len(positions)}/{len(boxes)} 박스, {len(layers)}개 층")

        # 모든 박스가 적재되지 않았더라도 부분 결과 반환
        return positions if positions else None

    def _find_position_in_layer_improved(self, box, layer_positions, layer_z):
        """개선된 층 내 위치 찾기"""
        # 5cm 단위로 탐색 (성능과 정확성의 균형)
        for y in range(0, self.vehicle.length - box.length + 1, 5):
            for x in range(0, self.vehicle.width - box.width + 1, 5):
                # 다른 박스와 겹치지 않는지 확인
                conflict = False
                for pos in layer_positions:
                    if not (x + box.width <= pos['x1'] or x >= pos['x2'] or
                           y + box.length <= pos['y1'] or y >= pos['y2']):
                        conflict = True
                        break

                if not conflict:
                    return (x, y, layer_z)

        return None

    def calculate_shuffling_cost(self, route_boxes, positions, destinations):
        """셔플링 비용 계산 - LIFO 방식"""
        shuffling_cost = 0

        # 목적지별로 박스들을 그룹화
        dest_boxes = {}
        for box in route_boxes:
            if box.destination not in dest_boxes:
                dest_boxes[box.destination] = []
            dest_boxes[box.destination].append(box)

        # 스택킹 순서에 따른 셔플링 계산
        remaining_boxes = set(box.box_id for box in route_boxes)

        for dest in destinations:
            if dest in dest_boxes:
                dest_box_ids = [box.box_id for box in dest_boxes[dest]]

                for box_id in dest_box_ids:
                    # 이 박스보다 나중에 적재된(스택킹 순서가 높은) 박스들의 수
                    box_position = next((pos for pos in positions if pos['box_id'] == box_id), None)
                    if box_position:
                        shuffles = sum(1 for pos in positions
                                     if pos['stacking_order'] > box_position['stacking_order']
                                     and pos['box_id'] in remaining_boxes)
                        shuffling_cost += shuffles * 500  # 셔플링 비용 500원
                        remaining_boxes.remove(box_id)

        return shuffling_cost

class RoutingOptimizer:
    def __init__(self, data):
        self.data = data
        self.distance_matrix = self._build_distance_matrix()
        self.depot_index = 0

    def _build_distance_matrix(self):
        """거리 행렬 구성"""
        locations = ['Depot'] + [f"D_{i:05d}" for i in range(1, 301)]
        n = len(locations)
        matrix = [[0] * n for _ in range(n)]

        # distance-data.txt 파일에서 거리 정보 읽기
        try:
            with open('distance-data.txt', 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or 'TIME_MIN' in line or line.startswith('#'):
                        continue  # 헤더나 빈 줄 스킵

                    parts = line.split('\t')
                    if len(parts) >= 3:
                        from_loc = parts[0].strip()
                        to_loc = parts[1].strip()

                        # 거리 값 파싱 (숫자가 아닌 경우 스킵)
                        try:
                            distance = float(parts[2])
                        except ValueError:
                            continue

                        # 인덱스 찾기
                        try:
                            from_idx = 0 if from_loc == 'Depot' else locations.index(from_loc)
                            to_idx = 0 if to_loc == 'Depot' else locations.index(to_loc)
                            matrix[from_idx][to_idx] = int(distance * 1000)  # 미터 단위로 변환
                        except ValueError:
                            # 위치를 찾을 수 없는 경우 스킵
                            continue
        except FileNotFoundError:
            print("거리 파일을 찾을 수 없습니다. 유클리드 거리를 사용합니다.")
            self._calculate_euclidean_distances(matrix)
        except Exception as e:
            print(f"거리 파일 읽기 오류: {e}. 유클리드 거리를 사용합니다.")
            self._calculate_euclidean_distances(matrix)

        return matrix

    def _calculate_euclidean_distances(self, matrix):
        """유클리드 거리 계산 (백업)"""
        depot_loc = self.data['depot']['location']
        locations = [depot_loc] + [dest['location'] for dest in self.data['destinations']]

        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    loc1 = locations[i]
                    loc2 = locations[j]
                    dist = math.sqrt((loc1['longitude'] - loc2['longitude'])**2 +
                                   (loc1['latitude'] - loc2['latitude'])**2)
                    matrix[i][j] = int(dist * 111000)  # 대략적인 미터 변환

    def solve_vrp(self, destination_demands, max_vehicles=50):
        """VRP 해결"""
        # 실제 방문해야 할 목적지만 필터링
        valid_destinations = []
        for dest in destination_demands.keys():
            try:
                dest_num = int(dest.split('_')[1])
                if 1 <= dest_num <= 300:
                    valid_destinations.append(dest_num)
            except (ValueError, IndexError):
                continue

        if not valid_destinations:
            return []

        # 거리 행렬 크기 조정 (Depot + 실제 목적지들)
        num_locations = len(valid_destinations) + 1

        manager = pywrapcp.RoutingIndexManager(num_locations, max_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            # 노드를 실제 거리 행렬 인덱스로 변환
            from_matrix_idx = 0 if from_node == 0 else valid_destinations[from_node - 1]
            to_matrix_idx = 0 if to_node == 0 else valid_destinations[to_node - 1]

            if (from_matrix_idx < len(self.distance_matrix) and
                to_matrix_idx < len(self.distance_matrix[0])):
                return self.distance_matrix[from_matrix_idx][to_matrix_idx]
            else:
                return 50000  # 기본 거리값

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 용량 제약
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node == 0:  # Depot
                return 0
            dest_id = f"D_{valid_destinations[from_node - 1]:05d}"
            return min(destination_demands.get(dest_id, 0), 8000000)  # 용량 제한

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # 모든 차량의 용량을 동일하게 설정
        vehicle_capacities = [8064000] * max_vehicles
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

        # 해결 매개변수
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 180  # 3분  제한

        try:
            solution = routing.SolveWithParameters(search_parameters)
            if solution:
                return self._extract_routes(manager, routing, solution, valid_destinations)
            else:
                print("VRP 해결 실패. 단순 알고리즘 사용...")
                return self._create_simple_routes(destination_demands)
        except Exception as e:
            print(f"VRP 해결 중 오류: {e}. 단순 알고리즘 사용...")
            return self._create_simple_routes(destination_demands)

    def _extract_routes(self, manager, routing, solution, valid_destinations):
        """솔루션에서 경로 추출"""
        routes = []
        route_count = 0  # 실제 경로 카운터

        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:  # Depot이 아닌 경우
                    dest_id = f"D_{valid_destinations[node - 1]:05d}"
                    route.append(dest_id)

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            if route:  # 빈 경로가 아닌 경우만 추가
                routes.append({
                    'vehicle_id': route_count,  # 0부터 시작하는 연속된 번호 사용
                    'destinations': route,
                    'distance': route_distance / 1000.0  # km 변환
                })
                route_count += 1

        return routes

    def _create_simple_routes(self, destination_demands):
        """간단한 탐욕 알고리즘으로 경로 생성"""
        routes = []
        unvisited = set(destination_demands.keys())
        vehicle_id = 0  # 0부터 시작

        while unvisited and vehicle_id < 50:
            route = []
            current_capacity = 0
            current_pos = 0  # Depot
            route_distance = 0

            while unvisited:
                best_dest = None
                best_distance = float('inf')

                # 가장 가까운 목적지 찾기
                for dest in unvisited:
                    try:
                        dest_num = int(dest.split('_')[1])
                        if dest_num < len(self.distance_matrix):
                            distance = self.distance_matrix[current_pos][dest_num]
                            demand = destination_demands[dest]

                            if current_capacity + demand <= 8064000 and distance < best_distance and distance > 0:
                                best_dest = dest
                                best_distance = distance
                    except (ValueError, IndexError):
                        continue

                if best_dest is None:
                    break

                route.append(best_dest)
                current_capacity += destination_demands[best_dest]
                current_pos = int(best_dest.split('_')[1])
                route_distance += best_distance / 1000.0  # km 변환
                unvisited.remove(best_dest)

            # 디팟으로 돌아가는 거리 추가
            if route and current_pos < len(self.distance_matrix):
                return_distance = self.distance_matrix[current_pos][0]
                route_distance += return_distance / 1000.0

            if route:
                routes.append({
                    'vehicle_id': vehicle_id,  # 0부터 시작
                    'destinations': route,
                    'distance': route_distance
                })
                print(f"차량 {vehicle_id} 경로 생성: {len(route)}개 목적지, {route_distance:.2f}km")

            vehicle_id += 1

        return routes

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        sys.exit(1)

    start_time = time.time()

    # 데이터 로드
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    # 박스 객체 생성
    boxes = []
    for order in data['orders']:
        box = Box(
            order['order_number'],
            order['box_id'],
            order['destination'],
            order['dimension']['width'],
            order['dimension']['length'],
            order['dimension']['height']
        )
        boxes.append(box)

    print(f"총 주문 수: {len(boxes)}")

    # 목적지별 박스 그룹화
    destination_boxes = {}
    destination_demands = {}

    for box in boxes:
        if box.destination not in destination_boxes:
            destination_boxes[box.destination] = []
        destination_boxes[box.destination].append(box)

        if box.destination not in destination_demands:
            destination_demands[box.destination] = 0
        # 부피를 세제곱센티미터 단위로 계산
        destination_demands[box.destination] += box.volume

    # 라우팅 최적화
    print("라우팅 최적화 시작...")
    routing_optimizer = RoutingOptimizer(data)
    routes = routing_optimizer.solve_vrp(destination_demands)

    print(f"생성된 경로 수: {len(routes)}")
    for route in routes[:3]:  # 처음 3개 경로만 출력
        print(f"차량 {route['vehicle_id']}: {len(route['destinations'])}개 목적지, {route['distance']:.2f}km")

    # 적재 최적화
    print("적재 최적화 시작...")
    loading_optimizer = LoadingOptimizer(data['vehicles'][0]['dimension'])

    results = []
    total_routing_cost = 0
    total_unloading_cost = 0
    processed_boxes = set()  # 처리된 박스 추적

    for route in routes:
        vehicle_id = route['vehicle_id']
        destinations = route['destinations']
        route_distance = route['distance']

        # 이 경로의 모든 박스들
        route_boxes = []
        for dest in destinations:
            route_boxes.extend(destination_boxes.get(dest, []))

        print(f"\n차량 {vehicle_id}: {len(route_boxes)}개 박스 처리 시작")

        # 적재 가능성 확인
        total_box_volume = sum(box.volume for box in route_boxes)
        if not loading_optimizer.can_fit_boxes(route_boxes):
            print(f"경고: 차량 {vehicle_id}에 모든 박스를 적재할 수 없습니다. (부피: {total_box_volume:,} > {loading_optimizer.vehicle.volume:,})")
            # 부피에 맞게 박스 수를 조정
            volume_ratio = loading_optimizer.vehicle.volume / total_box_volume
            max_boxes = int(len(route_boxes) * volume_ratio * 0.8)  # 80% 여유분
            route_boxes = route_boxes[:max_boxes]
            destinations = list(set(box.destination for box in route_boxes))
            print(f"차량 {vehicle_id}: 박스 수를 {len(route_boxes)}개로 조정")

        # 적재 위치 계산
        positions = loading_optimizer.calculate_loading_positions(route_boxes)
        if positions is None or len(positions) < len(route_boxes):
            print(f"경고: 차량 {vehicle_id}의 적재 계획 부분적 실패 ({len(positions) if positions else 0}/{len(route_boxes)})")

            if positions is None:
                # 완전 실패 시 더 적은 박스로 재시도
                route_boxes = route_boxes[:max(1, len(route_boxes)//4)]
                destinations = list(set(box.destination for box in route_boxes))
                positions = loading_optimizer.calculate_loading_positions(route_boxes)

                if positions is None:
                    print(f"차량 {vehicle_id}: 최종 적재 실패")
                    continue
            else:
                # 부분 성공 시 적재된 박스들만 사용
                loaded_box_ids = {pos['box_id'] for pos in positions}
                route_boxes = [box for box in route_boxes if box.box_id in loaded_box_ids]
                destinations = list(set(box.destination for box in route_boxes))
                print(f"차량 {vehicle_id}: {len(route_boxes)}개 박스로 진행")

        # 비용 계산
        fixed_cost = 150000  # 고정비
        fuel_cost = route_distance * 500  # 유류비 (500원/km)
        routing_cost = fixed_cost + fuel_cost

        print(f"차량 {vehicle_id}: 거리 {route_distance:.2f}km, 라우팅 비용 {routing_cost:,.0f}원")

        # 셔플링 비용 계산
        shuffling_cost = loading_optimizer.calculate_shuffling_cost(route_boxes, positions, destinations)

        print(f"차량 {vehicle_id}: 셔플링 비용 {shuffling_cost:,.0f}원")

        total_routing_cost += routing_cost
        total_unloading_cost += shuffling_cost

        # 각 박스별 결과 생성
        for i, dest in enumerate(destinations):
            dest_boxes = [box for box in route_boxes if box.destination == dest]
            for box in dest_boxes:
                position = next((pos for pos in positions if pos['box_id'] == box.box_id), None)
                if position:
                    results.append({
                        'Vehicle_ID': vehicle_id,
                        'Route_Order': i + 1,
                        'Destination': dest,
                        'Order_Number': box.order_number,
                        'Box_ID': box.box_id,
                        'Stacking_Order': position['stacking_order'],
                        'Lower_Left_X': position['x'],
                        'Lower_Left_Y': position['y'],
                        'Lower_Left_Z': position['z'],
                        'Longitude': next((d['location']['longitude'] for d in data['destinations']
                                         if d['destination_id'] == dest), 0),
                        'Latitude': next((d['location']['latitude'] for d in data['destinations']
                                        if d['destination_id'] == dest), 0),
                        'Box_Width': box.width / 100.0,  # 미터 단위로 변환
                        'Box_Length': box.length / 100.0,
                        'Box_Height': box.height / 100.0
                    })
                    processed_boxes.add(box.box_id)
                    print(f"  - 박스 {box.box_id} 처리 완료")

        print(f"차량 {vehicle_id} 완료: {len([box for box in route_boxes if box.box_id in processed_boxes])}개 박스 처리됨")

    # 처리되지 않은 박스들 확인
    unprocessed_boxes = [box for box in boxes if box.box_id not in processed_boxes]

    print(f"\n=== 처리 현황 ===")
    print(f"총 박스: {len(boxes)}")
    print(f"처리된 박스: {len(processed_boxes)}")
    print(f"미처리 박스: {len(unprocessed_boxes)}")

    if unprocessed_boxes:
        print(f"\n❌ 미처리 박스 {len(unprocessed_boxes)}개:")
        for i, box in enumerate(unprocessed_boxes[:10]):  # 처음 10개만 출력
            print(f"  {i+1}. {box.box_id} (목적지: {box.destination}, 크기: {box.width}x{box.length}x{box.height})")
        if len(unprocessed_boxes) > 10:
            print(f"  ... 및 {len(unprocessed_boxes) - 10}개 더")
    else:
        print("✅ 모든 박스가 처리되었습니다!")

    # 결과를 DataFrame으로 변환하고 정렬
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['Vehicle_ID', 'Route_Order', 'Stacking_Order'])

    # Excel 파일로 저장
    df.to_excel('Result.xlsx', index=False)

    # 결과 출력
    total_cost = total_routing_cost + total_unloading_cost
    elapsed_time = time.time() - start_time

    print(f"\n=== 최적화 결과 ===")
    print(f"총 주문 수: {len(boxes)}")
    print(f"처리된 주문 수: {len(processed_boxes)}")
    print(f"처리율: {len(processed_boxes)/len(boxes)*100:.1f}%")
    print(f"라우팅 비용: {total_routing_cost:,.0f}원")
    print(f"하차 비용: {total_unloading_cost:,.0f}원")
    print(f"총 비용: {total_cost:,.0f}원")
    print(f"실행 시간: {elapsed_time:.2f}초")
    print(f"결과 파일: Result.xlsx")

if __name__ == "__main__":
    main()
