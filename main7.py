import json
import sys
import pandas as pd
import time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_WIDTH, MAX_LENGTH, MAX_HEIGHT = 160, 280, 180
MAX_VOLUME = MAX_WIDTH * MAX_LENGTH * MAX_HEIGHT
FUEL_COST_PER_KM = 500
FIXED_COST = 150000
SHUFFLE_COST = 500

def load_json(file):
    with open(file, encoding='utf-8-sig') as f:
        return json.load(f)

def load_distance_matrix(file):
    distance_map = {}
    locations = set()
    with open(file, encoding='utf-8-sig') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            origin, dest, _, dist = parts
            distance_map[(origin, dest)] = int(dist)
            locations |= {origin, dest}
    sorted_locs = sorted(list(locations))
    idx_map = {loc: i for i, loc in enumerate(sorted_locs)}
    n = len(sorted_locs)
    matrix = [[0]*n for _ in range(n)]
    for (o, d), v in distance_map.items():
        i, j = idx_map[o], idx_map[d]
        matrix[i][j] = v
        matrix[j][i] = v
    return matrix, sorted_locs, idx_map

def compute_volume(d): return d["width"] * d["length"] * d["height"]

def split_orders_by_volume(orders):
    vehicles, current, vol = [], [], 0
    for o in sorted(orders, key=lambda x: -x["volume"]):
        if vol + o["volume"] > MAX_VOLUME:
            vehicles.append(current)
            current = [o]
            vol = o["volume"]
        else:
            current.append(o)
            vol += o["volume"]
    if current: vehicles.append(current)
    return vehicles

def get_coords(data):
    coords = {d["destination_id"]: d["location"] for d in data["destinations"]}
    coords["Depot"] = data["depot"]["location"]
    return coords

def get_vrp_route(matrix, idx_map, locations, depot_id, dest_ids):
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, idx_map[depot_id])
    routing = pywrapcp.RoutingModel(manager)
    def dist_cb(i, j): return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = 10
    solution = routing.SolveWithParameters(params)
    route = []
    if solution:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(locations[manager.IndexToNode(idx)])
            idx = solution.Value(routing.NextVar(idx))
        route.append(locations[manager.IndexToNode(idx)])
    return route

def is_overlap(b1, b2):
    def overlap(a1, a2, b1, b2): return max(a1, b1) < min(a2, b2)
    return all([
        overlap(b1["X"], b1["X"] + b1["Width"], b2["X"], b2["X"] + b2["Width"]),
        overlap(b1["Y"], b1["Y"] + b1["Length"], b2["Y"], b2["Y"] + b2["Length"]),
        overlap(b1["Z"], b1["Z"] + b1["Height"], b2["Z"], b2["Z"] + b2["Height"]),
    ])

def find_position(placed, dim, step=10):
    for x in range(0, MAX_WIDTH - dim["width"] + 1, step):
        for y in range(0, MAX_LENGTH - dim["length"] + 1, step):
            for z in range(0, MAX_HEIGHT - dim["height"] + 1, step):
                box = {"X": x, "Y": y, "Z": z, "Width": dim["width"], "Length": dim["length"], "Height": dim["height"]}
                if all(not is_overlap(box, p) for p in placed):
                    return box
    return None

def calculate_shuffle_cost(orders, dest_rank):
    placed_orders = [o for o in orders if "Y" in o]

    sorted_orders = sorted(placed_orders, key=lambda o: dest_rank[o["destination"]])
    
    shuffle_count = 0
    for i, o in enumerate(sorted_orders):
        for j in range(i):
            if sorted_orders[j]["Y"] > o["Y"]:
                shuffle_count += 1
    return shuffle_count * SHUFFLE_COST


def insert_depot_rows(df, depot_coord):
    result = []
    for vid in df["Vehicle_ID"].unique():
        vdf = df[df["Vehicle_ID"] == vid]
        for r in [vdf["Route_Order"].min() - 1, vdf["Route_Order"].max() + 1]:
            result.append({
                "Vehicle_ID": vid, "Route_Order": r, "Destination": "Depot",
                "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
                "Lower_Left_X": "", "Lower_Left_Y": "", "Lower_Left_Z": "",
                "Longitude": depot_coord["longitude"], "Latitude": depot_coord["latitude"],
                "Box_Width": "", "Box_Length": "", "Box_Height": ""
            })
    return pd.concat([df, pd.DataFrame(result)]).sort_values(by=["Vehicle_ID", "Route_Order"]).reset_index(drop=True)

def main():
    start_time = time.time()
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        return
    data = load_json(sys.argv[1])
    dist, locs, idx_map = load_distance_matrix(sys.argv[2])
    coords = get_coords(data)
    orders = data["orders"]
    for o in orders:
        o["volume"] = compute_volume(o["dimension"])
    vehicle_orders = split_orders_by_volume(orders)
    total_fuel, total_shuffle, rows = 0, 0, []
    placed_count = 0
    for vid, v_orders in enumerate(vehicle_orders):
        dests = list({o["destination"] for o in v_orders})
        route = get_vrp_route(dist, idx_map, locs, "Depot", dests)
        dest_rank = {d: i for i, d in enumerate(route)}
        placed = []
        for s_order, o in enumerate(v_orders):
            dim = o["dimension"]
            pos = find_position(placed, dim)
            if not pos: continue
            placed.append(pos)
            o.update(pos)
            row = {
                "Vehicle_ID": vid,
                "Route_Order": dest_rank[o["destination"]],
                "Destination": o["destination"],
                "Order_Number": o["order_number"],
                "Box_ID": o["box_id"],
                "Stacking_Order": s_order,
                "Lower_Left_X": pos["X"],
                "Lower_Left_Y": pos["Y"],
                "Lower_Left_Z": pos["Z"],
                "Longitude": coords[o["destination"]]["longitude"],
                "Latitude": coords[o["destination"]]["latitude"],
                "Box_Width": dim["width"],
                "Box_Length": dim["length"],
                "Box_Height": dim["height"]
            }
            rows.append(row)
            placed_count += 1
        dist_sum = sum(dist[idx_map[route[i]]][idx_map[route[i + 1]]] for i in range(len(route) - 1))
        total_fuel += (dist_sum / 1000) * FUEL_COST_PER_KM
        total_shuffle += calculate_shuffle_cost(v_orders, dest_rank)
    df = pd.DataFrame(rows)
    df = insert_depot_rows(df, coords["Depot"])
    df.to_excel("Result.xlsx", index=False)
    total_cost = int(len(vehicle_orders) * FIXED_COST + total_fuel + total_shuffle)
    print(f"처리율: {placed_count}/{len(orders)} = {placed_count / len(orders) * 100:.2f}%")
    print(f"차량 수: {len(vehicle_orders)}")
    print(f"고정비용: {len(vehicle_orders) * FIXED_COST:,}원")
    print(f"유류비: {int(total_fuel):,}원")
    print(f"셔플링비: {total_shuffle:,}원")
    print(f"총 비용: {total_cost:,}원")
    print(f"실행 소요 시간: {time.time() - start_time:.2f}초")

if __name__ == "__main__":
    main()
