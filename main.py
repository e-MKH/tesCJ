import json
import sys
import pandas as pd
from collections import defaultdict
import math

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class VehicleRoutingProblem:
    def __init__(self, data_file, distance_file):
        self.data = self.load_data(data_file)
        self.distance_matrix = self.load_distance_matrix(distance_file)
        self.depot_idx = 0

        self.truck_dims = {'width': 160, 'length': 280, 'height': 180}  # cm
        self.vehicle_capacity = self.truck_dims['width'] * self.truck_dims['length'] * self.truck_dims['height']

        self.max_vehicles = 9999
        self.fixed_cost = 150_000
        self.fuel_cost_per_km = 500
        self.shuffling_cost = 500

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_distance_matrix(self, filename):
        df = pd.read_csv(filename, sep='\t')
        used_destinations = set(order['destination'] for order in self.data['orders'])
        locations = ['Depot'] + sorted(list(used_destinations))
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}
        n = len(locations)

        matrix = [[999999] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 0

        for _, row in df.iterrows():
            origin, destination, distance = row['ORIGIN'], row['DESTINATION'], row['DISTANCE_METER']
            if origin in location_to_idx and destination in location_to_idx:
                i, j = location_to_idx[origin], location_to_idx[destination]
                matrix[i][j] = int(distance)

        # 대칭 거리 보정
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 999999 and matrix[j][i] != 999999:
                    matrix[i][j] = matrix[j][i]
                elif matrix[j][i] == 999999 and matrix[i][j] != 999999:
                    matrix[j][i] = matrix[i][j]

        return matrix, locations

    def create_data_model(self, orders):
        """
        orders: 현재 처리할 주문 리스트
        """
        distance_matrix, locations = self.distance_matrix
        destination_orders = defaultdict(list)
        total_volume = 0

        for order in orders:
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
        optimal_vehicles = min(max(1, min_vehicles_by_volume, min_vehicles_by_destinations), self.max_vehicles)

        demands = [0]
        order_info = [{}]

        # locations 중 depot(0)을 제외하고 실제 목적지만 처리
        for location in locations[1:]:
            volume = sum(order['volume'] for order in destination_orders[location])
            demands.append(volume)
            order_info.append(destination_orders[location])

        return {
            'distance_matrix': distance_matrix,
            'demands': demands,
            'vehicle_capacities': [self.vehicle_capacity] * optimal_vehicles,
            'num_vehicles': optimal_vehicles,
            'depot': 0,
            'locations': locations,
            'order_info': order_info,
            'orders': orders  # 현재 주문 리스트 추가 (필요시)
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
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.FromSeconds(300)

        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            # fallback 재시도
            search_parameters.time_limit.FromSeconds(60)
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            solution = routing.SolveWithParameters(search_parameters)

        return manager, routing, solution

    def extract_routes(self, manager, routing, solution, data):
        routes = []
        total_distance = 0

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route, route_distance = [], 0

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

    def perform_3d_packing(self, routes, data):
        failed_orders = set()
        for route_info in routes:
            route = route_info['route']
            truck_dims = self.truck_dims
            delivery_sequence = []

            for i, location_idx in enumerate(route[1:-1], 1):
                location_name = data['locations'][location_idx]
                orders = data['order_info'][location_idx]

                for order in orders:
                    delivery_sequence.append({
                        'delivery_order': i,
                        'order': order,
                        'location': location_name
                    })

            occupied_spaces = []
            stacking_order = 1

            for delivery_info in delivery_sequence:
                order = delivery_info['order']
                box_dim = order['dimension']
                pos = self.find_best_position(box_dim, truck_dims, occupied_spaces)

                if not pos:
                    failed_orders.add(order['order_number'])
                    continue

                x, y, z = pos
                occupied_spaces.append({
                    'x': x, 'y': y, 'z': z,
                    'width': box_dim['width'],
                    'length': box_dim['length'],
                    'height': box_dim['height'],
                    'delivery_order': delivery_info['delivery_order']
                })

                shuffling_count = self.calculate_shuffling(pos, box_dim, delivery_info['delivery_order'], occupied_spaces[:-1])

                route_info['orders'].append({
                    'order_number': order['order_number'],
                    'box_id': order['box_id'],
                    'destination': delivery_info['location'],
                    'stacking_order': stacking_order,
                    'position': {'x': x, 'y': y, 'z': z},
                    'shuffling_count': shuffling_count,
                    'delivery_order': delivery_info['delivery_order']
                })
                stacking_order += 1

            route_info['total_shuffling'] = sum(o['shuffling_count'] for o in route_info['orders'])

        return routes, failed_orders

    def find_best_position(self, box_dim, truck_dims, occupied_spaces):
        best_position, min_height = None, float('inf')
        step = 10  # 더 세밀하게 탐색

        for x in range(0, truck_dims['width'] - box_dim['width'] + 1, step):
            for y in range(0, truck_dims['length'] - box_dim['length'] + 1, step):
                z = self.find_lowest_z(x, y, box_dim, occupied_spaces)
                if z + box_dim['height'] <= truck_dims['height'] and z < min_height:
                    min_height = z
                    best_position = (x, y, z)

        return best_position

    def find_lowest_z(self, x, y, box_dim, occupied_spaces):
        z = 0
        for space in occupied_spaces:
            if self.boxes_overlap_xy({'x': x, 'y': y, 'width': box_dim['width'], 'length': box_dim['length']}, space):
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
        x, y, z = position
        shuffling = 0
        current_box = {'x': x, 'y': y, 'z': z, 'width': box_dim['width'], 'length': box_dim['length'], 'height': box_dim['height']}

        for other in other_boxes:
            if other['delivery_order'] > delivery_order and other['z'] > z and self.boxes_overlap_xy(current_box, other):
                shuffling += 1

        return shuffling

    def calculate_total_cost(self, routes):
        num_vehicles_used = len([r for r in routes if len(r['route']) > 2])
        fixed_cost = num_vehicles_used * self.fixed_cost
        fuel_cost = sum(r['distance'] * self.fuel_cost_per_km / 1000 for r in routes)
        shuffling_cost = sum(r['total_shuffling'] * self.shuffling_cost for r in routes)

        return {
            'total_score': fixed_cost + fuel_cost + shuffling_cost,
            'routing_cost': fixed_cost + fuel_cost,
            'fixed_cost': fixed_cost,
            'fuel_cost': fuel_cost,
            'unloading_cost': shuffling_cost,
            'shuffling_cost': shuffling_cost,
            'num_vehicles': num_vehicles_used
        }

    def save_results(self, routes, cost_info, data):
        result_data = []

        for route_index, route_info in enumerate(routes):
            vehicle_id = route_index
            route_order = 1  # 각 vehicle_id마다 초기화

            # 시작 Depot 추가
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
            route_order += 1  # depot 후 바로 증가

            # 박스 순서대로 기록
            sorted_orders = sorted(route_info['orders'], key=lambda x: x['delivery_order'])

            for order_info in sorted_orders:
                dest_id = order_info['destination']
                dest_data = next(d for d in self.data['destinations'] if d['destination_id'] == dest_id)
                order_data = next(o for o in self.data['orders'] if o['order_number'] == order_info['order_number'])

                result_data.append({
                    'Vehicle_ID': vehicle_id,
                    'Route_Order': route_order,
                    'Destination': dest_id,
                    'Order_Number': order_info['order_number'],
                    'Box_ID': order_info['box_id'],
                    'Stacking_Order': order_info['stacking_order'],
                    'Lower_Left_X': order_info['position']['x'] / 100,  # cm → m
                    'Lower_Left_Y': order_info['position']['y'] / 100,
                    'Lower_Left_Z': order_info['position']['z'] / 100,
                    'Longitude': dest_data['location']['longitude'],
                    'Latitude': dest_data['location']['latitude'],
                    'Box_Width': order_data['dimension']['width'] / 100,
                    'Box_Length': order_data['dimension']['length'] / 100,
                    'Box_Height': order_data['dimension']['height'] / 100
                })
                route_order += 1  # ✅ 매 행마다 증가시킴

            # 마지막 Depot 추가
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

        # 최종 DataFrame으로 저장
        df = pd.DataFrame(result_data)
        df.to_excel('Result.xlsx', sheet_name='Detailed Route Information', index=False)



def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        sys.exit(1)

    data_file = sys.argv[1]
    distance_file = sys.argv[2]

    vrp = VehicleRoutingProblem(data_file, distance_file)

    remaining_orders = vrp.data['orders'][:]  # 초기 전체 주문 복사
    max_attempts = 20
    attempt = 0
    num_vehicles = 1

    all_routes = []  # ✅ 모든 성공한 라우트를 저장할 리스트

    while attempt < max_attempts and remaining_orders:
        print(f"[시도 {attempt + 1}] 처리 주문 수: {len(remaining_orders)}, 차량 수: {num_vehicles}")

        data = vrp.create_data_model(remaining_orders)
        if num_vehicles > data['num_vehicles']:
            data['num_vehicles'] = num_vehicles
            data['vehicle_capacities'] = [vrp.vehicle_capacity] * num_vehicles

        manager, routing, solution = vrp.solve_vrp(data)

        if not solution:
            print(f"[시도 {attempt + 1}] 라우팅 실패 - 차량 수: {num_vehicles}")
            num_vehicles += 1
            attempt += 1
            continue

        routes, _ = vrp.extract_routes(manager, routing, solution, data)
        routes, failed_orders = vrp.perform_3d_packing(routes, data)

        # ✅ 누적 저장
        successful_routes = [r for r in routes if r['vehicle_id'] is not None]
        all_routes.extend(successful_routes)

        if failed_orders:
            print(f"[시도 {attempt + 1}] 적재 누락 주문: {sorted(failed_orders)} → 차량 수 증가")
            remaining_orders = [order for order in remaining_orders if order['order_number'] in failed_orders]
            num_vehicles += 1
            attempt += 1
            continue

        # 모든 주문 처리 성공
        cost_info = vrp.calculate_total_cost(all_routes)  # ✅ 누적된 전체 라우트로 비용 계산
        vrp.save_results(all_routes, cost_info, data)     # ✅ 누적된 전체 라우트 저장
        print(f"✅ 성공적으로 모든 주문 처리 완료 - 시도 횟수: {attempt + 1}")
        break

    else:
        print("❌ 모든 시도 실패: 모든 주문을 처리할 수 없습니다.")
        sys.exit(1)


if __name__ == '__main__':
    main()
