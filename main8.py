# [1] 주요 모듈 및 상수 정의
import json
import pandas as pd
import time
from collections import defaultdict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_CAPACITY = 160 * 280 * 180
SHUFFLE_COST_PER_BOX = 500
FUEL_COST_PER_KM = 500
FIXED_COST_PER_VEHICLE = 150000

# [2] 데이터 로딩 및 전처리
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

# [3] 적재 알고리즘
def is_overlap(b1, b2):
    return (max(b1["X"], b2["X"]) < min(b1["X"] + b1["Width"], b2["X"] + b2["Width"]) and
            max(b1["Y"], b2["Y"]) < min(b1["Y"] + b1["Length"], b2["Y"] + b2["Length"]) and
            max(b1["Z"], b2["Z"]) < min(b1["Z"] + b1["Height"], b2["Z"] + b2["Height"]))

def find_position(placed, dim):
    for z in range(0, 180 - dim["height"] + 1, 5):
        for y in range(0, 280 - dim["length"] + 1, 5):
            for x in range(0, 160 - dim["width"] + 1, 5):
                new_box = {"X": x, "Y": y, "Z": z, "Width": dim["width"], "Length": dim["length"], "Height": dim["height"]}
                if all(not is_overlap(new_box, p) for p in placed):
                    return new_box
    return None

# [4] 라우팅 + 적재 + 비용 계산
def solve(data, dist_matrix, locations, idx_map):
    orders = data["orders"]
    for o in orders:
        o["volume"] = calculate_order_volume(o)

    coords = get_coords_map(data)
    start_time = time.time()
    unplaced = []
    results = []
    vehicle_id = 0
    current_orders = orders.copy()

    while current_orders:
        placed, volume, packed = [], 0, []
        for o in sorted(current_orders, key=lambda x: -x["volume"]):
            if volume + o["volume"] > MAX_CAPACITY:
                continue
            pos = find_position(placed, o["dimension"])
            if pos:
                o.update(pos)
                placed.append(pos)
                packed.append(o)
                volume += o["volume"]
            else:
                continue

        if not packed:
            # 강제로 넣기
            o = current_orders.pop(0)
            pos = find_position(placed, o["dimension"])
            if pos:
                o.update(pos)
                placed.append(pos)
                packed.append(o)
                volume += o["volume"]
            else:
                unplaced.append(o)
                continue

        dests = list(set([p["destination"] for p in packed]))
        route = get_vrp_route(dist_matrix, idx_map, locations, "Depot", dests)
        dest_rank = {d: i for i, d in enumerate(route)}

        for i, o in enumerate(packed):
            results.append({
                "Vehicle_ID": vehicle_id,
                "Route_Order": dest_rank[o["destination"]],
                "Destination": o["destination"],
                "Order_Number": o["order_number"],
                "Box_ID": o["box_id"],
                "Stacking_Order": i,
                "X": o["X"], "Y": o["Y"], "Z": o["Z"],
                "Longitude": coords[o["destination"]]["longitude"],
                "Latitude": coords[o["destination"]]["latitude"],
                "Width": o["dimension"]["width"],
                "Length": o["dimension"]["length"],
                "Height": o["dimension"]["height"]
            })
        vehicle_id += 1
        current_orders = [o for o in current_orders if o not in packed]

    df = pd.DataFrame(results)

    # 비용 계산
    total_distance = 0
    for v in range(vehicle_id):
        df_v = df[df["Vehicle_ID"] == v].sort_values("Route_Order")
        route = ["Depot"] + df_v["Destination"].tolist() + ["Depot"]
        for i in range(len(route) - 1):
            total_distance += dist_matrix[idx_map[route[i]]][idx_map[route[i+1]]]

    fuel = int((total_distance / 1000) * FUEL_COST_PER_KM)
    fixed = vehicle_id * FIXED_COST_PER_VEHICLE
    shuffle = sum(df.groupby("Vehicle_ID").apply(lambda g: g["Z"].rank(method="dense", ascending=False).sum()).astype(int)) * SHUFFLE_COST_PER_BOX

    exec_time = round(time.time() - start_time, 2)
    print(f"처리율: {len(df)} / {len(data['orders'])} = {round(len(df)/len(data['orders'])*100, 2)}%")
    print(f"차량 수: {vehicle_id}")
    print(f"고정비용: {fixed:,}원")
    print(f"유류비: {fuel:,}원")
    print(f"셔플링비: {shuffle:,}원")
    print(f"총 비용: {fixed + fuel + shuffle:,}원")
    print(f"실행 소요 시간: {exec_time}초")

    df.to_excel("Result.xlsx", index=False)

def get_vrp_route(dist_matrix, idx_map, locations, depot_id, destination_ids):
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, idx_map[depot_id])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_callback = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.seconds = 5
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(locations[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        route.append(locations[manager.IndexToNode(index)])
    return route

# [5] 실행
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
    else:
        data = load_json(sys.argv[1])
        matrix, locs, idmap = load_distance_data(sys.argv[2])
        solve(data, matrix, locs, idmap)