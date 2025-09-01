import json
import pandas as pd
import time
import math
from collections import defaultdict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys
import numpy as np

# [1] 상수 정의
MAX_CAPACITY = 160 * 280 * 180
SHUFFLE_COST_PER_BOX = 500
FUEL_COST_PER_KM = 500
FIXED_COST_PER_VEHICLE = 150000

# [2] 데이터 로딩 및 전처리 함수들
def calculate_order_volume(order):
    d = order["dimension"]
    return d["width"] * d["length"] * d["height"]

def load_json(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def load_distance_data(filepath):
    distance_map = {}
    locations = set()
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            origin, dest, _, dist = parts
            distance_map[(origin, dest)] = int(dist)
            locations.update([origin, dest])
    sorted_locations = sorted(list(locations))
    index_map = {loc: i for i, loc in enumerate(sorted_locations)}
    matrix = [[0]*len(sorted_locations) for _ in sorted_locations]
    for (o, d), v in distance_map.items():
        i, j = index_map[o], index_map[d]
        matrix[i][j] = v
        matrix[j][i] = v
    return matrix, sorted_locations, index_map

def get_coords_map(data):
    coords = {d["destination_id"]: d["location"] for d in data["destinations"]}
    coords["Depot"] = data["depot"]["location"]
    return coords

# [3] 고성능 3D 빈 패킹 알고리즘
def is_overlap(b1, b2):
    """두 박스가 겹치는지 확인"""
    return (max(b1["X"], b2["X"]) < min(b1["X"] + b1["Width"], b2["X"] + b2["Width"]) and
            max(b1["Y"], b2["Y"]) < min(b1["Y"] + b1["Length"], b2["Y"] + b2["Length"]) and
            max(b1["Z"], b2["Z"]) < min(b1["Z"] + b1["Height"], b2["Z"] + b2["Height"]))

def is_within_bounds(box):
    """박스가 차량 경계 내에 있는지 확인"""
    return (box["X"] + box["Width"] <= 160 and
            box["Y"] + box["Length"] <= 280 and
            box["Z"] + box["Height"] <= 180)

def find_best_position_greedy(placed_boxes, dimension):
    """욕심쟁이 알고리즘으로 최적 위치 찾기 - 바닥 우선, 벽면 활용"""
    best_pos = None
    best_score = float('inf')

    # 바닥부터 층층이 쌓기
    for z in range(0, 180 - dimension["height"] + 1, 5):
        for y in range(0, 280 - dimension["length"] + 1, 5):
            for x in range(0, 160 - dimension["width"] + 1, 5):
                new_box = {
                    "X": x, "Y": y, "Z": z,
                    "Width": dimension["width"],
                    "Length": dimension["length"],
                    "Height": dimension["height"]
                }

                # 경계 확인
                if not is_within_bounds(new_box):
                    continue

                # 겹침 확인
                if any(is_overlap(new_box, placed) for placed in placed_boxes):
                    continue

                # 점수 계산: 낮은 위치, 뒤쪽, 왼쪽 우선
                # 추가적으로 지지대 점수도 고려
                support_score = 0
                if z == 0:  # 바닥에 닿으면 최고점
                    support_score = 1000
                else:
                    # 아래에 지지대가 있는지 확인
                    for placed in placed_boxes:
                        if (placed["Z"] + placed["Height"] == z and
                            max(placed["X"], x) < min(placed["X"] + placed["Width"], x + dimension["width"]) and
                            max(placed["Y"], y) < min(placed["Y"] + placed["Length"], y + dimension["length"])):
                            support_score += 100

                # 벽면 접촉 보너스
                wall_bonus = 0
                if x == 0: wall_bonus += 10  # 왼쪽 벽
                if y == 0: wall_bonus += 10  # 뒤쪽 벽

                score = z * 100 + y * 10 + x - support_score - wall_bonus

                if score < best_score:
                    best_score = score
                    best_pos = new_box

    return best_pos

def pack_orders_greedy(orders):
    """주문들을 욕심쟁이 알고리즘으로 3D 패킹"""
    # 박스 크기별로 정렬 (큰 것부터, 정육면체 우선)
    def box_priority(order):
        d = order["dimension"]
        volume = d["width"] * d["length"] * d["height"]
        # 정육면체에 가까울수록 우선순위 높음
        dimensions = sorted([d["width"], d["length"], d["height"]])
        ratio = dimensions[2] / dimensions[0] if dimensions[0] > 0 else 1
        return (-volume, ratio)

    sorted_orders = sorted(orders, key=box_priority)

    placed_boxes = []
    packed_orders = []
    total_volume = 0

    for order in sorted_orders:
        order_volume = calculate_order_volume(order)

        # 용량 체크
        if total_volume + order_volume > MAX_CAPACITY:
            continue

        # 위치 찾기
        position = find_best_position_greedy(placed_boxes, order["dimension"])
        if position:
            # 주문에 위치 정보 추가
            order.update(position)
            placed_boxes.append(position)
            packed_orders.append(order)
            total_volume += order_volume

    return packed_orders, total_volume

def pack_all_orders_into_trucks(orders):
    """모든 주문을 트럭들에 최대한 꽉 채워서 적재"""
    trucks = []
    remaining_orders = orders.copy()
    truck_id = 0

    while remaining_orders:
        print(f"트럭 {truck_id} 적재 중... 남은 주문: {len(remaining_orders)}개")

        # 현재 트럭에 최대한 많이 적재
        packed_orders, used_volume = pack_orders_greedy(remaining_orders)

        if not packed_orders:
            # 더 이상 적재할 수 없는 주문들이 남아있음
            print(f"경고: {len(remaining_orders)}개 주문이 적재 불가")
            break

        trucks.append({
            "truck_id": truck_id,
            "orders": packed_orders,
            "volume_used": used_volume,
            "volume_rate": round(used_volume / MAX_CAPACITY * 100, 2)
        })

        print(f"트럭 {truck_id}: {len(packed_orders)}개 주문, 용적률 {trucks[-1]['volume_rate']}%")

        # 적재된 주문들을 제거
        remaining_orders = [o for o in remaining_orders if o not in packed_orders]
        truck_id += 1

    print(f"총 {len(trucks)}대 트럭에 적재 완료")
    return trucks

# [4] 라우팅 최적화 (적재 후 처리)
def optimize_route_for_truck(truck_orders, dist_matrix, locations, idx_map):
    """한 트럭의 배송 경로 최적화"""
    if not truck_orders:
        return ["Depot"]

    # 고유한 목적지 추출
    destinations = list(set(order["destination"] for order in truck_orders))

    if len(destinations) == 1:
        return ["Depot", destinations[0], "Depot"]

    # OR-Tools를 사용한 TSP 해결
    all_locations = ["Depot"] + destinations
    n = len(all_locations)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # Depot이 0번 인덱스
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = all_locations[manager.IndexToNode(from_index)]
        to_node = all_locations[manager.IndexToNode(to_index)]

        if from_node in idx_map and to_node in idx_map:
            return dist_matrix[idx_map[from_node]][idx_map[to_node]]
        return 999999

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 검색 매개변수 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 2

    # 해 구하기
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(all_locations[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        route.append(all_locations[manager.IndexToNode(index)])  # 마지막 Depot
        return route
    else:
        # 최적화 실패시 간단한 휴리스틱
        return ["Depot"] + destinations + ["Depot"]

def calculate_shuffle_cost(truck_orders, route):
    """셔플링 비용 계산 - 배송 순서를 고려한 Z축 기준"""
    if not truck_orders or not route:
        return 0

    total_shuffle_cost = 0
    dest_order = {dest: i for i, dest in enumerate(route) if dest != "Depot"}

    # 목적지별로 박스들을 그룹화
    dest_boxes = defaultdict(list)
    for order in truck_orders:
        dest = order["destination"]
        dest_boxes[dest].append(order)

    # 각 목적지별로 셔플링 계산
    for dest, boxes in dest_boxes.items():
        if dest not in dest_order:
            continue

        # Z축 기준으로 정렬 (높은 것부터 - 나중에 적재된 것)
        boxes.sort(key=lambda x: -x["Z"])

        # 각 박스의 셔플링 순서 = 위에 쌓인 박스 개수
        for i, box in enumerate(boxes):
            shuffle_count = i  # 자신보다 위에 있는 박스 개수
            total_shuffle_cost += shuffle_count * SHUFFLE_COST_PER_BOX

    return total_shuffle_cost

# [5] 결과 생성
def generate_results(trucks_with_routes, coords):
    """최종 결과 데이터프레임 생성"""
    results = []

    for truck_data in trucks_with_routes:
        truck_id = truck_data["truck_id"]
        orders = truck_data["orders"]
        route = truck_data["route"]

        if not orders:
            # 빈 트럭
            results.append({
                "Vehicle_ID": truck_id,
                "Route_Order": 0,
                "Destination": "Depot",
                "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
                "X": 0, "Y": 0, "Z": 0,
                "Longitude": coords["Depot"]["longitude"],
                "Latitude": coords["Depot"]["latitude"],
                "Width": 0, "Length": 0, "Height": 0
            })
            continue

        # 경로 순서 매핑
        route_order = {dest: i for i, dest in enumerate(route)}

        # 시작점 추가
        results.append({
            "Vehicle_ID": truck_id,
            "Route_Order": 0,
            "Destination": "Depot",
            "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
            "X": 0, "Y": 0, "Z": 0,
            "Longitude": coords["Depot"]["longitude"],
            "Latitude": coords["Depot"]["latitude"],
            "Width": 0, "Length": 0, "Height": 0
        })

        # 셔플링 순서 계산
        stacking_orders = {}
        dest_boxes = defaultdict(list)

        for order in orders:
            dest_boxes[order["destination"]].append(order)

        for dest, boxes in dest_boxes.items():
            boxes.sort(key=lambda x: -x["Z"])  # 높은 순서
            for i, box in enumerate(boxes):
                stacking_orders[box["box_id"]] = i

        # 주문 데이터 추가
        for order in orders:
            results.append({
                "Vehicle_ID": truck_id,
                "Route_Order": route_order.get(order["destination"], 1),
                "Destination": order["destination"],
                "Order_Number": order["order_number"],
                "Box_ID": order["box_id"],
                "Stacking_Order": stacking_orders.get(order["box_id"], 0),
                "X": order["X"], "Y": order["Y"], "Z": order["Z"],
                "Longitude": coords[order["destination"]]["longitude"],
                "Latitude": coords[order["destination"]]["latitude"],
                "Width": order["dimension"]["width"],
                "Length": order["dimension"]["length"],
                "Height": order["dimension"]["height"]
            })

        # 종료점 추가
        results.append({
            "Vehicle_ID": truck_id,
            "Route_Order": len([d for d in route if d != "Depot"]) + 1,
            "Destination": "Depot",
            "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
            "X": 0, "Y": 0, "Z": 0,
            "Longitude": coords["Depot"]["longitude"],
            "Latitude": coords["Depot"]["latitude"],
            "Width": 0, "Length": 0, "Height": 0
        })

    return pd.DataFrame(results)

# [6] 메인 최적화 함수
def solve_pack_first(data, dist_matrix, locations, idx_map):
    """적재 우선 + 라우팅 후처리 방식"""
    orders = data["orders"]
    for order in orders:
        order["volume"] = calculate_order_volume(order)

    coords = get_coords_map(data)
    start_time = time.time()

    print("=== 1단계: 모든 주문을 트럭에 최대한 적재 ===")
    trucks = pack_all_orders_into_trucks(orders)

    print(f"\n=== 2단계: 각 트럭별 라우팅 최적화 ===")
    trucks_with_routes = []
    total_shuffle_cost = 0

    for truck in trucks:
        print(f"트럭 {truck['truck_id']} 라우팅 최적화 중...")

        route = optimize_route_for_truck(
            truck["orders"], dist_matrix, locations, idx_map
        )

        shuffle_cost = calculate_shuffle_cost(truck["orders"], route)
        total_shuffle_cost += shuffle_cost

        trucks_with_routes.append({
            "truck_id": truck["truck_id"],
            "orders": truck["orders"],
            "route": route,
            "shuffle_cost": shuffle_cost,
            "volume_rate": truck["volume_rate"]
        })

        print(f"  경로: {' -> '.join(route)}")
        print(f"  셔플링 비용: {shuffle_cost:,}원")

    print(f"\n=== 3단계: 결과 생성 ===")
    df = generate_results(trucks_with_routes, coords)

    # 총 비용 계산
    total_distance = 0
    num_trucks = len(trucks_with_routes)

    for truck_data in trucks_with_routes:
        route = truck_data["route"]
        for i in range(len(route) - 1):
            curr_dest = route[i]
            next_dest = route[i + 1]
            if curr_dest in idx_map and next_dest in idx_map:
                total_distance += dist_matrix[idx_map[curr_dest]][idx_map[next_dest]]

    fuel_cost = int((total_distance / 1000) * FUEL_COST_PER_KM)
    fixed_cost = num_trucks * FIXED_COST_PER_VEHICLE
    total_cost = fixed_cost + fuel_cost + total_shuffle_cost

    # 결과 출력
    processed_orders = len(df[df['Box_ID'] != ''])
    total_orders = len(orders)
    exec_time = round(time.time() - start_time, 2)

    print(f"\n=== 최종 결과 ===")
    print(f"처리율: {processed_orders} / {total_orders} = {round(processed_orders/total_orders*100, 2)}%")
    print(f"차량 수: {num_trucks}")
    print(f"평균 용적률: {round(sum(t['volume_rate'] for t in trucks_with_routes) / len(trucks_with_routes), 2)}%")
    print(f"고정비용: {fixed_cost:,}원")
    print(f"유류비: {fuel_cost:,}원")
    print(f"셔플링비: {total_shuffle_cost:,}원")
    print(f"총 비용: {total_cost:,}원")
    print(f"실행 소요 시간: {exec_time}초")

    return df

# [7] 엑셀 저장
def save_result_excel(df):
    """결과를 엑셀 파일로 저장"""
    column_mapping = {
        "Vehicle_ID": "Vehicle_ID",
        "Route_Order": "Route_Order",
        "Destination": "Destination",
        "Order_Number": "Order_Number",
        "Box_ID": "Box_ID",
        "Stacking_Order": "Stacking_Order",
        "X": "Lower_Left_X",
        "Y": "Lower_Left_Y",
        "Z": "Lower_Left_Z",
        "Longitude": "Longitude",
        "Latitude": "Latitude",
        "Width": "Box_Width",
        "Length": "Box_Length",
        "Height": "Box_Height"
    }

    column_order = [
        "Vehicle_ID", "Route_Order", "Destination", "Order_Number", "Box_ID", "Stacking_Order",
        "Lower_Left_X", "Lower_Left_Y", "Lower_Left_Z",
        "Longitude", "Latitude", "Box_Width", "Box_Length", "Box_Height"
    ]

    df_output = df.rename(columns=column_mapping)
    df_output = df_output[column_order]
    df_output.to_excel("Result.xlsx", index=False)
    print("Result.xlsx 파일이 생성되었습니다.")

# [8] 실행
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
    else:
        data = load_json(sys.argv[1])
        matrix, locs, idmap = load_distance_data(sys.argv[2])

        df = solve_pack_first(data, matrix, locs, idmap)
        save_result_excel(df)
