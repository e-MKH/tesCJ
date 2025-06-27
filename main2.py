import json
import os
import sys
import time
import openpyxl
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -------------------- 상수 --------------------
TRUCK_WIDTH = 160
TRUCK_LENGTH = 280
TRUCK_HEIGHT = 180
FUEL_COST_PER_KM = 500
FIXED_COST_PER_TRUCK = 150000
HANDLING_COST_PER_MOVE = 500
BIG_DISTANCE = 999999

# -------------------- 데이터 로딩 --------------------
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_distance_data(path):
    distance_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            origin, destination, _, meter = parts
            try:
                km = float(meter) / 1000
                distance_map[(origin, destination)] = km
            except:
                continue
    return distance_map

# -------------------- 거리 행렬 생성 --------------------
def create_distance_matrix(data):
    n = len(data['dest_index_map'])
    matrix = [[BIG_DISTANCE]*n for _ in range(n)]
    for (src, dst), dist in data['distance_map'].items():
        if src in data['dest_index_map'] and dst in data['dest_index_map']:
            i = data['dest_index_map'][src]
            j = data['dest_index_map'][dst]
            matrix[i][j] = dist
    return matrix

# -------------------- 데이터 모델 --------------------
def create_data_model(json_data, distance_map):
    order_dest_ids = sorted(set([o['destination'] for o in json_data['orders']]))
    dest_index_map = {d: i for i, d in enumerate(order_dest_ids)}
    index_dest_map = {i: d for d, i in dest_index_map.items()}

    return {
        'depot': json_data['depot'],
        'destinations': {d['destination_id']: d['location'] for d in json_data['destinations'] if d['destination_id'] in dest_index_map},
        'orders': json_data['orders'],
        'distance_map': distance_map,
        'dest_index_map': dest_index_map,
        'index_dest_map': index_dest_map
    }

# -------------------- VRP 최적화 --------------------
def solve_vrp(data):
    matrix = create_distance_matrix(data)
    num_locations = len(matrix)
    num_vehicles = num_locations
    depot_index = 0

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(matrix[i][j] * 1000)  # meters

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)
    routing.AddDimension(transit_cb, 0, 1000000, True, 'Distance')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        print("[오류] 최적화 실패")
        return []

    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # 사용되지 않은 차량
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(data['index_dest_map'][node])
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)
    return routes

# -------------------- 적재 순서 시뮬레이션 --------------------
def assign_stacking_order(data, routes):
    results = []
    for vidx, route in enumerate(routes):
        z = 0
        order_num = 1
        for dest in route:
            for order in data['orders']:
                if order['destination'] == dest:
                    results.append({
                        "Vehicle_ID": f"V_{vidx+1:03}",
                        "Route_Order": route.index(dest),
                        "Destination": dest,
                        "Order_Number": order['order_number'],
                        "Box_ID": order['box_id'],
                        "Stacking_Order": order_num,
                        "Lower_Left_X": 0,
                        "Lower_Left_Y": 0,
                        "Lower_Left_Z": z,
                        "Longitude": data['destinations'][dest]['longitude'],
                        "Latitude": data['destinations'][dest]['latitude'],
                        "Box_Width": order['dimension']['width'],
                        "Box_Length": order['dimension']['length'],
                        "Box_Height": order['dimension']['height']
                    })
                    z += order['dimension']['height']
                    order_num += 1
    return results

# -------------------- 비용 계산 --------------------
def calculate_total_cost(routes, data):
    fuel = 0
    handling = 0
    fixed = len(routes) * FIXED_COST_PER_TRUCK
    for route in routes:
        for i in range(len(route)-1):
            d = data['distance_map'].get((route[i], route[i+1]), 0)
            fuel += d * FUEL_COST_PER_KM
        handling += len(route) * HANDLING_COST_PER_MOVE
    total = fuel + handling + fixed
    return int(fuel), int(handling), int(fixed), int(total)

# -------------------- 엑셀 저장 --------------------
def save_to_excel(results):
    wb = openpyxl.Workbook()
    ws = wb.active
    headers = list(results[0].keys())
    ws.append(headers)
    for row in results:
        ws.append([row[h] for h in headers])
    wb.save("Result.xlsx")

# -------------------- Main --------------------
def main():
    start = time.time()

    if len(sys.argv) < 3:
        print("[Usage] python main.py data.json distance-data.txt")
        return

    json_data = load_json(sys.argv[1])
    distance_map = load_distance_data(sys.argv[2])
    data = create_data_model(json_data, distance_map)
    routes = solve_vrp(data)

    if not routes:
        return

    stacking = assign_stacking_order(data, routes)
    fuel, handling, fixed, total = calculate_total_cost(routes, data)

    print("=== 최적화 결과 ===")
    print(f"차량 수: {len(routes)}대")
    print(f"유류비: {fuel:,.0f}원")
    print(f"하차비: {handling:,.0f}원")
    print(f"고정비: {fixed:,.0f}원")
    print(f"총 비용: {total:,.0f}원")

    save_to_excel(stacking)
    print("[결과 저장] Result.xlsx 완료")
    print(f"[완료] 전체 실행 시간: {time.time() - start:.2f}초")

if __name__ == '__main__':
    main()
