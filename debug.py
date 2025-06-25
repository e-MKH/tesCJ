import json
import sys
import time
import pandas as pd
from collections import defaultdict
import math
import traceback
import os

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class VehicleRoutingProblem:
    def __init__(self, data_file, distance_file):
        print(f"🔧 데이터 파일 로딩: {data_file}, {distance_file}")
        self.data = self.load_data(data_file)
        self.distance_matrix = self.load_distance_matrix(distance_file)
        self.depot_idx = 0

        # 차량 기본 설정 (5톤 트럭 기준)
        self.vehicle_capacity = 25_000_000  # cm³
        self.max_vehicles = 8

        # 평가 기준에 따른 비용 설정
        self.fixed_cost = 150_000  # 차량 고정비 (150,000원)
        self.fuel_cost_per_km = 500  # 유류비 (500원/km)
        self.shuffling_cost = 500  # 셔플링 비용 (500원/셔플링)
        print(f"✅ 초기화 완료 - 주문 수: {len(self.data['orders'])}, 목적지 수: {len(self.data['destinations'])}")

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_distance_matrix(self, filename):
        df = pd.read_csv(filename, sep='\t')

        # 실제 주문이 있는 목적지만 추출하여 메모리 최적화
        used_destinations = set()
        for order in self.data['orders']:
            used_destinations.add(order['destination'])

        print(f"🎯 사용되는 목적지 수: {len(used_destinations)}")

        # 위치 목록 (Depot + 사용되는 목적지만)
        locations = ['Depot'] + sorted(list(used_destinations))
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}

        # 거리 매트릭스 생성
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

        print(f"📊 거리 매트릭스 생성 완료: {n}x{n}")
        return matrix, locations

    def create_data_model(self):
        """OR-Tools용 데이터 모델 생성"""
        print("📋 데이터 모델 생성 중...")
        distance_matrix, locations = self.distance_matrix

        # 주문을 목적지별로 그룹화
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

        # 동적 차량 수 최적화
        min_vehicles_by_volume = math.ceil(total_volume / self.vehicle_capacity)
        min_vehicles_by_destinations = math.ceil(len(destination_orders) / 25)
        optimal_vehicles = max(min_vehicles_by_volume, min_vehicles_by_destinations, 1)
        optimal_vehicles = min(optimal_vehicles, self.max_vehicles)

        print(f"🚛 최적 차량 수: {optimal_vehicles} (볼륨기준: {min_vehicles_by_volume}, 목적지기준: {min_vehicles_by_destinations})")

        # 위치별 수요 계산
        demands = [0]  # Depot 수요는 0
        order_info = [{}]  # Depot 주문 정보

        for location in locations[1:]:  # Depot 제외
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
        """OR-Tools로 VRP 해결"""
        print("🔄 VRP 해결 중...")
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )

        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
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

        # 검색 파라미터 (시간제한)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(300)  # 5분 제한

        print("⏱️ 솔루션 탐색 중 (최대 5분)...")
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            print("⚠️ 첫 번째 시도 실패, 빠른 해법으로 재시도...")
            # 빠른 해법으로 재시도
            search_parameters.time_limit.FromSeconds(60)
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            solution = routing.SolveWithParameters(search_parameters)

        if solution:
            print("✅ 솔루션 탐색 완료!")
        else:
            print("❌ 솔루션 탐색 실패!")

        return manager, routing, solution

    def extract_routes(self, manager, routing, solution, data):
        """해결된 경로 추출"""
        print("🛣️ 경로 추출 중...")
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

            if len(route) > 2:  # 실제 배송이 있는 경우만
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'distance': route_distance,
                    'orders': []
                })
                total_distance += route_distance
                print(f"🚛 차량{vehicle_id}: 경로 {route}, 거리 {route_distance}m")

        print(f"📈 총 {len(routes)}대 차량, 총 거리 {total_distance}m")
        return routes, total_distance

    def perform_3d_packing(self, routes, data):
        """3D 패킹 및 셔플링 계산"""
        print("📦 3D 패킹 및 셔플링 계산 중...")
        for route_info in routes:
            vehicle_id = route_info['vehicle_id']
            route = route_info['route']

            # 5톤 트럭 적재 공간 (cm 단위)
            truck_dims = {'width': 620, 'length': 230, 'height': 230}

            # 배송 순서 수집 (LIFO 방식으로 셔플링 최소화)
            delivery_sequence = []
            for i, location_idx in enumerate(route[1:-1], 1):
                if location_idx == 0:
                    continue
                location_name = data['locations'][location_idx]
                orders = data['order_info'][location_idx]

                for order in orders:
                    delivery_sequence.append({
                        'delivery_order': i,
                        'order': order,
                        'location': location_name
                    })

            print(f"🚛 차량{vehicle_id}: {len(delivery_sequence)}개 주문 처리")

            # 배송 역순으로 적재 (마지막 배송지가 맨 위)
            delivery_sequence.reverse()

            occupied_spaces = []
            total_shuffling = 0
            stacking_order = 1

            for delivery_info in delivery_sequence:
                order = delivery_info['order']
                box_dim = order['dimension']

                # Bottom-Left-Fill 방식으로 최적 위치 찾기
                position = self.find_best_position(box_dim, truck_dims, occupied_spaces)

                if position is None:
                    print(f"⚠️ 박스 {order['box_id']} 적재 불가")
                    continue

                x, y, z = position

                # 점유 공간 추가
                occupied_spaces.append({
                    'x': x, 'y': y, 'z': z,
                    'width': box_dim['width'],
                    'length': box_dim['length'],
                    'height': box_dim['height'],
                    'delivery_order': delivery_info['delivery_order']
                })

                # 셔플링 계산
                shuffling_count = self.calculate_shuffling(
                    position, box_dim, delivery_info['delivery_order'], occupied_spaces[:-1]
                )
                total_shuffling += shuffling_count

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

            route_info['total_shuffling'] = total_shuffling
            print(f"🚛 차량{vehicle_id}: 총 셔플링 {total_shuffling}회")

        return routes

    def find_best_position(self, box_dim, truck_dims, occupied_spaces):
        """Bottom-Left-Fill 알고리즘으로 최적 위치 찾기"""
        best_position = None
        min_height = float('inf')

        step = 20  # 20cm 간격으로 탐색
        for x in range(0, truck_dims['width'] - box_dim['width'] + 1, step):
            for y in range(0, truck_dims['length'] - box_dim['length'] + 1, step):
                z = self.find_lowest_z(x, y, box_dim, occupied_spaces)

                if z + box_dim['height'] <= truck_dims['height']:
                    if z < min_height:
                        min_height = z
                        best_position = (x, y, z)

        return best_position

    def find_lowest_z(self, x, y, box_dim, occupied_spaces):
        """주어진 (x,y) 위치에서 가능한 최하단 z 좌표"""
        z = 0

        for space in occupied_spaces:
            if self.boxes_overlap_xy(
                {'x': x, 'y': y, 'width': box_dim['width'], 'length': box_dim['length']},
                space
            ):
                z = max(z, space['z'] + space['height'])

        return z

    def boxes_overlap_xy(self, box1, box2):
        """두 박스가 xy 평면에서 겹치는지 확인"""
        return not (
            box1['x'] + box1['width'] <= box2['x'] or
            box2['x'] + box2['width'] <= box1['x'] or
            box1['y'] + box1['length'] <= box2['y'] or
            box2['y'] + box2['length'] <= box1['y']
        )

    def calculate_shuffling(self, position, box_dim, delivery_order, other_boxes):
        """셔플링 횟수 계산: 상품을 꺼내기 위해 이동해야 하는 주변 상품의 수"""
        x, y, z = position
        shuffling = 0

        current_box = {
            'x': x, 'y': y, 'z': z,
            'width': box_dim['width'],
            'length': box_dim['length'],
            'height': box_dim['height']
        }

        for other_box in other_boxes:
            # 나중에 배송되고 위에 있으면서 xy평면에서 겹치는 경우
            if (other_box['delivery_order'] > delivery_order and
                other_box['z'] > z and
                self.boxes_overlap_xy(current_box, other_box)):
                shuffling += 1

        return shuffling

    def calculate_total_cost(self, routes):
        """총 비용 계산"""
        num_vehicles_used = len([r for r in routes if len(r['route']) > 2])

        # 1) 라우팅 비용 = 고정비 + 유류비
        fixed_cost = num_vehicles_used * self.fixed_cost
        fuel_cost = sum(r['distance'] * self.fuel_cost_per_km / 1000 for r in routes)

        # 2) 하차 비용 = 셔플링 횟수 x 셔플링 비용
        shuffling_cost = sum(r['total_shuffling'] * self.shuffling_cost for r in routes)

        # Total Score = 라우팅 비용 + 하차 비용
        routing_cost = fixed_cost + fuel_cost
        unloading_cost = shuffling_cost
        total_score = routing_cost + unloading_cost

        print(f"💰 비용 계산 완료:")
        print(f"   - 사용 차량: {num_vehicles_used}대")
        print(f"   - 고정비: {fixed_cost:,}원")
        print(f"   - 유류비: {fuel_cost:,.0f}원")
        print(f"   - 셔플링비: {shuffling_cost:,}원")
        print(f"   - 총 점수: {total_score:,.0f}원")

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
        """Result.xlsx 파일로 결과 저장"""
        print("💾 결과 저장 중...")
        result_data = []

        try:
            for route_index, route_info in enumerate(routes):  # enumerate 추가
            # Vehicle_ID를 0부터 시작하도록 수정
                vehicle_id = route_index  # ✅ 이렇게 변경
                route_order = 1  # ✅ 그대로 유지

                print(f"🚛 차량{vehicle_id} 데이터 처리 중...")

                # Depot 시작
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

                # 배송 순서대로 정렬
                sorted_orders = sorted(route_info['orders'], key=lambda x: x['delivery_order'])
                print(f"   - 처리할 주문 수: {len(sorted_orders)}")

                current_destination = None
                for order_info in sorted_orders:
                    dest_id = order_info['destination']

                    # 새로운 목적지인 경우 Route_Order 증가
                    if dest_id != current_destination:
                        current_destination = dest_id
                        route_order += 1

                    # 목적지 및 주문 정보 찾기
                    try:
                        dest_data = next(d for d in self.data['destinations'] if d['destination_id'] == dest_id)
                    except StopIteration:
                        print(f"❌ 목적지 {dest_id}를 찾을 수 없음!")
                        continue

                    try:
                        order_data = next(o for o in self.data['orders'] if o['order_number'] == order_info['order_number'])
                    except StopIteration:
                        print(f"❌ 주문 {order_info['order_number']}를 찾을 수 없음!")
                        continue

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
                        'Box_Width': order_data['dimension']['width'],
                        'Box_Length': order_data['dimension']['length'],
                        'Box_Height': order_data['dimension']['height']
                    })

                # Depot 복귀
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

            print(f"📊 총 {len(result_data)}행의 데이터 생성")

            # Result.xlsx 파일로 저장
            df = pd.DataFrame(result_data)
            output_path = os.path.abspath('Result.xlsx')
            print(f"💾 저장 경로: {output_path}")

            df.to_excel('Result.xlsx', sheet_name='Detailed Route Information', index=False)

            if os.path.exists('Result.xlsx'):
                print("✅ Result.xlsx 파일 저장 완료!")
            else:
                print("❌ Result.xlsx 파일 저장 실패!")

        except Exception as e:
            print(f"❌ save_results 오류: {str(e)}")
            print(f"❌ 상세 오류: {traceback.format_exc()}")
            raise


def main():
    print("🚀 VRP 시스템 시작")
    print(f"📂 현재 작업 디렉토리: {os.getcwd()}")

    if len(sys.argv) != 3:
        print("❌ 사용법: python main.py data.json distance-data.txt")
        sys.exit(1)

    data_file = sys.argv[1]
    distance_file = sys.argv[2]

    print(f"📁 입력 파일: {data_file}, {distance_file}")

    try:
        # VRP 문제 해결
        vrp = VehicleRoutingProblem(data_file, distance_file)
        data = vrp.create_data_model()

        manager, routing, solution = vrp.solve_vrp(data)

        if solution:
            routes, total_distance = vrp.extract_routes(manager, routing, solution, data)

            if routes:
                routes = vrp.perform_3d_packing(routes, data)
                cost_info = vrp.calculate_total_cost(routes)
                vrp.save_results(routes, cost_info, data)
                print("🎉 모든 처리 완료!")
            else:
                print("❌ 유효한 경로가 생성되지 않음")
                sys.exit(1)
        else:
            print("❌ OR-Tools 솔루션을 찾을 수 없음")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 메인 프로세스 오류: {str(e)}")
        print(f"❌ 상세 오류: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
