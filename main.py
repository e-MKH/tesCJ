import json
import sys
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import time
import random

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from py3dbp import Packer, Bin, Item

class VehicleRoutingProblem:
    def __init__(self, data_file, distance_file):
        self.data = self.load_data(data_file)
        self.distance_matrix = self.load_distance_matrix(distance_file)
        self.depot_idx = 0
        self.vehicle_capacity = 160 * 280 * 180  # 8,064,000cm³
        self.max_vehicles = 12  # 트럭 수 제한 완화
        self.fixed_cost = 150_000
        self.fuel_cost_per_km = 500
        self.shuffling_cost = 500

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_distance_matrix(self, filename):
        df = pd.read_csv(filename, sep='\t')

        # 주문에 있는 모든 목적지 수집
        used_destinations = set()
        for order in self.data['orders']:
            used_destinations.add(order['destination'])

        locations = ['Depot'] + sorted(list(used_destinations))
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}

        n = len(locations)
        matrix = [[999999] * n for _ in range(n)]

        # 대각선은 0으로 설정
        for i in range(n):
            matrix[i][i] = 0

        # 거리 데이터 채우기
        for _, row in df.iterrows():
            origin = row['ORIGIN']
            destination = row['DESTINATION']
            distance = row['DISTANCE_METER']

            if origin in location_to_idx and destination in location_to_idx:
                i = location_to_idx[origin]
                j = location_to_idx[destination]
                matrix[i][j] = int(distance)

        # 대칭 매트릭스 처리
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 999999 and matrix[j][i] != 999999:
                    matrix[i][j] = matrix[j][i]
                elif matrix[j][i] == 999999 and matrix[i][j] != 999999:
                    matrix[j][i] = matrix[i][j]

        return matrix, locations

    def create_data_model(self):
        distance_matrix, locations = self.distance_matrix

        destination_orders = defaultdict(list)
        total_volume = 0

        for order in self.data['orders']:
            dest_id = order['destination']
            volume = order['dimension']['width'] * order['dimension']['length'] * order['dimension']['height']
            total_volume += volume

            destination_orders[dest_id].append({
                'order_number': order['order_number'],
                'box_id': order['box_id'],
                'volume': volume,
                'dimension': order['dimension']
            })

        min_vehicles_by_volume = math.ceil(total_volume / self.vehicle_capacity)
        min_vehicles_by_destinations = math.ceil(len(destination_orders) / 25)
        optimal_vehicles = max(min_vehicles_by_volume, min_vehicles_by_destinations, 1)
        optimal_vehicles = min(optimal_vehicles, self.max_vehicles)

        demands = [0]
        order_info = [{}]

        for location in locations[1:]:
            total_volume = sum(order['volume'] for order in destination_orders[location])
            demands.append(total_volume)
            order_info.append(destination_orders[location])

        return {
            'distance_matrix': distance_matrix,
            'demands': demands,
            'vehicle_capacities': [self.vehicle_capacity] * optimal_vehicles,
            'num_vehicles': optimal_vehicles,
            'depot': 0,
            'locations': locations,
            'order_info': order_info
        }

    def solve_vrp(self, data):
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            data['vehicle_capacities'],
            True,
            'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(300)

        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            search_parameters.time_limit.FromSeconds(60)
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            solution = routing.SolveWithParameters(search_parameters)

        return manager, routing, solution

    def extract_routes(self, manager, routing, solution, data):
        routes = []
        total_distance = 0

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            route.append(manager.IndexToNode(index))

            if len(route) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'distance': route_distance,
                    'orders': []
                })
                total_distance += route_distance

        return routes, total_distance

    def find_best_position_extreme(self, box_dim, truck_dims, occupied_spaces, step=2):
        # Extreme Point 방식: 후보 위치 리스트 + 박스 회전 허용
        candidates = []
        if not occupied_spaces:
            candidates = [(0, 0, 0)]
        else:
            for space in occupied_spaces:
                candidates.append((space['x'] + space['width'], space['y'], space['z']))
                candidates.append((space['x'], space['y'] + space['length'], space['z']))
                candidates.append((space['x'], space['y'], space['z'] + space['height']))
        best_position = None
        min_z = float('inf')
        # 박스 회전 허용: (width, length)와 (length, width) 모두 시도
        for dims in [box_dim, {'width': box_dim['length'], 'length': box_dim['width'], 'height': box_dim['height']}]:
            for point in candidates:
                x, y, z = point
                # step 간격으로 x, y를 더 세밀하게 이동
                for dx in range(0, 1, step):
                    for dy in range(0, 1, step):
                        xx, yy = x + dx, y + dy
                        if (xx + dims['width'] > truck_dims['width'] or
                            yy + dims['length'] > truck_dims['length'] or
                            z + dims['height'] > truck_dims['height']):
                            continue
                        overlap = False
                        for space in occupied_spaces:
                            if not (
                                xx + dims['width'] <= space['x'] or
                                space['x'] + space['width'] <= xx or
                                yy + dims['length'] <= space['y'] or
                                space['y'] + space['length'] <= yy or
                                z + dims['height'] <= space['z'] or
                                space['z'] + space['height'] <= z
                            ):
                                overlap = True
                                break
                        if overlap:
                            continue
                        if z < min_z:
                            min_z = z
                            best_position = (xx, yy, z, dims['width'], dims['length'], dims['height'])
        return best_position if best_position is not None else None

    def find_best_position_layer(self, box_dim, truck_dims, occupied_spaces, layer_height=0):
        # 현재 layer_height(층의 바닥) 위에 박스를 놓을 수 있는 위치를 찾음
        step = 5
        best_position = None
        min_x, min_y = None, None
        for x in range(0, truck_dims['width'] - box_dim['width'] + 1, step):
            for y in range(0, truck_dims['length'] - box_dim['length'] + 1, step):
                z = layer_height
                # 겹침 검사
                overlap = False
                for space in occupied_spaces:
                    if not (
                        x + box_dim['width'] <= space['x'] or
                        space['x'] + space['width'] <= x or
                        y + box_dim['length'] <= space['y'] or
                        space['y'] + space['length'] <= y or
                        z + box_dim['height'] <= space['z'] or
                        space['z'] + space['height'] <= z
                    ):
                        overlap = True
                        break
                if not overlap:
                    best_position = (x, y, z)
                    return best_position  # 가장 먼저 찾은 위치에 배치
        return None  # 이 층에 더 이상 못 놓음

    def try_multiple_packing_strategies(self, delivery_sequence, truck_dims):
        # 여러 packing 순서(큰 박스 우선, 작은 박스 우선, 랜덤)를 시도해 최적 결과 선택
        strategies = [
            lambda seq: sorted(seq, key=lambda d: d['order']['dimension']['width'] * d['order']['dimension']['length'] * d['order']['dimension']['height'], reverse=True),  # 큰 박스 우선
            lambda seq: sorted(seq, key=lambda d: d['order']['dimension']['width'] * d['order']['dimension']['length'] * d['order']['dimension']['height']),  # 작은 박스 우선
            lambda seq: random.sample(seq, len(seq)),  # 랜덤
        ]
        best_orders = []
        best_occupied = []
        best_total_shuffling = 0
        for strat in strategies:
            seq = strat(delivery_sequence[:])
            occupied_spaces = []
            orders = []
            total_shuffling = 0
            stacking_order = 1
            for delivery_info in seq:
                order = delivery_info['order']
                box_dim = order['dimension']
                pos = self.find_best_position_extreme(box_dim, {'width': 160, 'length': 280, 'height': 180}, occupied_spaces, step=2)
                if pos is None:
                    continue
                x, y, z, w, l, h = pos
                occupied_spaces.append({
                    'x': x, 'y': y, 'z': z,
                    'width': w,
                    'length': l,
                    'height': h,
                    'delivery_order': delivery_info['delivery_order']
                })
                shuffling_count = self.calculate_shuffling(
                    (x, y, z), {'width': w, 'length': l, 'height': h}, delivery_info['delivery_order'], occupied_spaces[:-1]
                )
                total_shuffling += shuffling_count
                orders.append({
                    'order_number': order['order_number'],
                    'box_id': order['box_id'],
                    'destination': delivery_info['location'],
                    'stacking_order': stacking_order,
                    'position': {'x': x, 'y': y, 'z': z},
                    'box_rotation': {'width': w, 'length': l, 'height': h},
                    'shuffling_count': shuffling_count,
                    'delivery_order': delivery_info['delivery_order']
                })
                stacking_order += 1
            if len(orders) > len(best_orders):
                best_orders = orders
                best_occupied = occupied_spaces
                best_total_shuffling = total_shuffling
        return best_orders, best_occupied, best_total_shuffling

    def calculate_shuffling_for_packed_orders(self, packed_orders):
        shuffling_total = 0
        for i, box in enumerate(packed_orders):
            x, y, z = box['position']['x'], box['position']['y'], box['position']['z']
            w, l, h = box['box_rotation']['width'], box['box_rotation']['length'], box['box_rotation']['height']
            current_box = {'x': x, 'y': y, 'z': z, 'width': w, 'length': l, 'height': h}
            shuffling = 0
            for j, other in enumerate(packed_orders):
                if i == j:
                    continue
                ox, oy, oz = other['position']['x'], other['position']['y'], other['position']['z']
                ow, ol, oh = other['box_rotation']['width'], other['box_rotation']['length'], other['box_rotation']['height']
                # 위에 쌓인 박스가 있는지
                if (oz > z and
                    not (x + w <= ox or ox + ow <= x or y + l <= oy or oy + ol <= y)):
                    shuffling += 1
            box['shuffling_count'] = shuffling
            shuffling_total += shuffling
        return shuffling_total

    def perform_3d_packing(self, routes, data):
        truck_dims = {'width': 160, 'length': 280, 'height': 180}
        all_boxes = []
        order_map = {}
        for i, location in enumerate(data['locations'][1:], 1):
            orders = data['order_info'][i]
            for order in orders:
                all_boxes.append({
                    'order_number': order['order_number'],
                    'box_id': order['box_id'],
                    'destination': location,
                    'dimension': order['dimension'],
                    'delivery_order': i
                })
                order_map[order['order_number']] = {
                    'box_id': order['box_id'],
                    'destination': location,
                    'dimension': order['dimension'],
                    'delivery_order': i
                }
        remaining = all_boxes[:]
        truck_id = 0
        packed_trucks = []
        while remaining and truck_id < self.max_vehicles:
            packer = Packer()
            bin = Bin(f'truck_{truck_id}', truck_dims['width'], truck_dims['length'], truck_dims['height'], 999999)
            packer.add_bin(bin)
            for box in remaining:
                packer.add_item(Item(
                    box['order_number'],
                    box['dimension']['width'],
                    box['dimension']['length'],
                    box['dimension']['height'],
                    1
                ))
            packer.pack(bigger_first=True, distribute_items=False)
            packed_orders = []
            for item in bin.items:
                order_info = order_map.get(item.name, {})
                packed_orders.append({
                    'order_number': item.name,
                    'box_id': order_info.get('box_id', ''),
                    'destination': order_info.get('destination', ''),
                    'stacking_order': 0,
                    'position': {'x': item.position[0], 'y': item.position[1], 'z': item.position[2]},
                    'box_rotation': {'width': item.width, 'length': item.depth, 'height': item.height},
                    'shuffling_count': 0,
                    'delivery_order': order_info.get('delivery_order', 0)
                })
            # 현실적인 셔플링(하차) 횟수 계산
            total_shuffling = self.calculate_shuffling_for_packed_orders(packed_orders)
            packed_trucks.append({'orders': packed_orders, 'total_shuffling': total_shuffling, 'vehicle_id': truck_id, 'route': [], 'distance': 0})
            packed_order_numbers = set(item.name for item in bin.items)
            remaining = [box for box in remaining if box['order_number'] not in packed_order_numbers]
            truck_id += 1
        for i, route_info in enumerate(routes):
            if i < len(packed_trucks):
                route_info['orders'] = packed_trucks[i]['orders']
                route_info['total_shuffling'] = packed_trucks[i]['total_shuffling']
            else:
                route_info['orders'] = []
                route_info['total_shuffling'] = 0
        return routes

    def find_best_position(self, box_dim, truck_dims, occupied_spaces, step=20):
        best_position = None
        min_height = float('inf')

        for x in range(0, truck_dims['width'] - box_dim['width'] + 1, step):
            for y in range(0, truck_dims['length'] - box_dim['length'] + 1, step):
                z = self.find_lowest_z(x, y, box_dim, occupied_spaces)

                if z + box_dim['height'] <= truck_dims['height']:
                    if z < min_height:
                        min_height = z
                        best_position = (x, y, z)

        return best_position

    def find_lowest_z(self, x, y, box_dim, occupied_spaces):
        z = 0

        for space in occupied_spaces:
            if self.boxes_overlap_xy(
                {'x': x, 'y': y, 'width': box_dim['width'], 'length': box_dim['length']},
                space
            ):
                z = max(z, space['z'] + space['height'])

        return z

    def boxes_overlap_xy(self, box1, box2):
        return not (
            box1['x'] + box1['width'] <= box2['x'] or
            box2['x'] + box2['width'] <= box1['x'] or
            box1['y'] + box1['length'] <= box2['y'] or
            box2['y'] + box2['length'] <= box1['y']
        )

    def calculate_shuffling(self, position, box_dim, delivery_order, other_boxes):
        x, y, z = position  # position에서 좌표 추출
        current_box = {
            'x': x, 'y': y, 'z': z,
            'width': box_dim['width'],
            'length': box_dim['length'],
            'height': box_dim['height']
        }
        shuffling = 0
        for other_box in other_boxes:
            if (other_box['delivery_order'] > delivery_order and
                other_box['z'] > z and
                self.boxes_overlap_xy(current_box, other_box)):
                shuffling += 1
        return shuffling

    def calculate_total_cost(self, routes):
        num_vehicles_used = len([r for r in routes if len(r['route']) > 2])

        fixed_cost = num_vehicles_used * self.fixed_cost
        fuel_cost = sum(r['distance'] * self.fuel_cost_per_km / 1000 for r in routes)

        shuffling_cost = sum(r['total_shuffling'] * self.shuffling_cost for r in routes)

        routing_cost = fixed_cost + fuel_cost
        unloading_cost = shuffling_cost
        total_score = routing_cost + unloading_cost

        return {
            'total_score': total_score,
            'routing_cost': routing_cost,
            'fixed_cost': fixed_cost,
            'fuel_cost': fuel_cost,
            'unloading_cost': unloading_cost,
            'shuffling_cost': shuffling_cost,
            'num_vehicles': num_vehicles_used
        }

    def save_results(self, routes, cost_info, data):
        result_data = []

        for route_index, route_info in enumerate(routes):
            vehicle_id = route_index
            route_order = 1

            result_data.append({
                'Vehicle_ID': vehicle_id,
                'Route_Order': route_order,
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': self.data['depot']['location']['longitude'],
                'Latitude': self.data['depot']['location']['latitude'],
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            })

            sorted_orders = sorted(route_info['orders'], key=lambda x: x['delivery_order'])

            current_destination = None
            for order_info in sorted_orders:
                dest_id = order_info['destination']

                if dest_id != current_destination:
                    current_destination = dest_id
                    route_order += 1

                dest_data = next(d for d in self.data['destinations'] if d['destination_id'] == dest_id)
                order_data = next(o for o in self.data['orders'] if o['order_number'] == order_info['order_number'])

                result_data.append({
                    'Vehicle_ID': vehicle_id,
                    'Route_Order': route_order,
                    'Destination': dest_id,
                    'Order_Number': order_info['order_number'],
                    'Box_ID': order_info['box_id'],
                    'Stacking_Order': order_info['stacking_order'],
                    'Lower_Left_X': order_info['position']['x'],
                    'Lower_Left_Y': order_info['position']['y'],
                    'Lower_Left_Z': order_info['position']['z'],
                    'Longitude': dest_data['location']['longitude'],
                    'Latitude': dest_data['location']['latitude'],
                    'Box_Width': order_info['box_rotation']['width'],
                    'Box_Length': order_info['box_rotation']['length'],
                    'Box_Height': order_info['box_rotation']['height']
                })

            route_order += 1
            result_data.append({
                'Vehicle_ID': vehicle_id,
                'Route_Order': route_order,
                'Destination': 'Depot',
                'Order_Number': '',
                'Box_ID': '',
                'Stacking_Order': '',
                'Lower_Left_X': '',
                'Lower_Left_Y': '',
                'Lower_Left_Z': '',
                'Longitude': self.data['depot']['location']['longitude'],
                'Latitude': self.data['depot']['location']['latitude'],
                'Box_Width': '',
                'Box_Length': '',
                'Box_Height': ''
            })

        df = pd.DataFrame(result_data)
        df.to_excel('Result.xlsx', sheet_name='Detailed Route Information', index=False)


def main():
    start_time = time.time()
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        sys.exit(1)

    data_file = sys.argv[1]
    distance_file = sys.argv[2]

    vrp = VehicleRoutingProblem(data_file, distance_file)
    data = vrp.create_data_model()
    manager, routing, solution = vrp.solve_vrp(data)

    if solution:
        routes, total_distance = vrp.extract_routes(manager, routing, solution, data)
        routes = vrp.perform_3d_packing(routes, data)
        cost_info = vrp.calculate_total_cost(routes)
        vrp.save_results(routes, cost_info, data)
        elapsed = time.time() - start_time
        total_orders = len(vrp.data['orders'])
        processed_orders = sum(len(order['orders']) for order in routes)
        throughput = processed_orders / total_orders * 100 if total_orders > 0 else 0
        # 셔플링 횟수 합계 계산
        total_shuffling_count = sum(r.get('total_shuffling', 0) for r in routes)
        print(f"총 주문 수: {total_orders}")
        print(f"처리된 주문 수: {processed_orders}")
        print(f"처리율: {throughput:.2f}%")
        print(f"라우팅 비용: {cost_info['routing_cost']:,}원")
        print(f"하차 비용: {cost_info['unloading_cost']:,}원 (셔플링 횟수: {total_shuffling_count}회)")
        print(f"총 비용: {cost_info['total_score']:,}원")
        print(f"실행 시간: {elapsed:.2f}초")
    else:
        print("해를 찾지 못했습니다.")
        sys.exit(1)

if __name__ == '__main__':
    main()