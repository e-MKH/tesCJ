import json
import pandas as pd
import time
from collections import defaultdict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys

MAX_CAPACITY = 160 * 280 * 180
SHUFFLE_COST_PER_BOX = 500
FUEL_COST_PER_KM = 500
FIXED_COST_PER_VEHICLE = 150000

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

def is_overlap(b1, b2):
    return (max(b1["X"], b2["X"]) < min(b1["X"] + b1["Width"], b2["X"] + b2["Width"]) and
            max(b1["Y"], b2["Y"]) < min(b1["Y"] + b1["Length"], b2["Y"] + b2["Length"]) and
            max(b1["Z"], b2["Z"]) < min(b1["Z"] + b1["Height"], b2["Z"] + b2["Height"]))

def is_within_bounds(box):
    return (box["X"] + box["Width"] <= 160 and
            box["Y"] + box["Length"] <= 280 and
            box["Z"] + box["Height"] <= 180)

def find_position_fill_priority(placed, dim):
    candidates = []
    for z in range(0, 180 - dim["height"] + 1, 10):
        for y in range(0, 280 - dim["length"] + 1, 10):
            for x in range(0, 160 - dim["width"] + 1, 10):
                new_box = {"X": x, "Y": y, "Z": z, "Width": dim["width"], "Length": dim["length"], "Height": dim["height"]}
                if is_within_bounds(new_box) and all(not is_overlap(new_box, p) for p in placed):
                    candidates.append(((z, y, x), new_box))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None

def save_result_excel(df):
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
    df = df.rename(columns=column_mapping)
    df = df[column_order]
    df.to_excel("Result.xlsx", index=False)

def get_vrp_route(dist_matrix, idx_map, locations, depot_id, destination_ids):
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, idx_map[depot_id])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_callback = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.seconds = 2
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

def solve(data, dist_matrix, locations, idx_map):
    orders = data["orders"]
    for o in orders:
        o["volume"] = calculate_order_volume(o)

    coords = get_coords_map(data)
    start_time = time.time()
    results = []
    vehicle_id = 0
    current_orders = orders.copy()
    total_shuffle_cost = 0
    packed_ids = set()

    while current_orders:
        placed, volume, packed = [], 0, []

        dests = list({o["destination"] for o in current_orders})
        route = get_vrp_route(dist_matrix, idx_map, locations, "Depot", dests)
        if not route:
            break
        dest_rank = {d: i for i, d in enumerate(route)}

        sorted_orders = sorted(current_orders, key=lambda x: -dest_rank.get(x["destination"], 999))
        sorted_orders = sorted(sorted_orders, key=lambda x: -x["volume"])
        for o in sorted_orders + sorted(current_orders, key=lambda x: x["volume"]):
            if o["box_id"] in packed_ids:
                continue
            pos = find_position_fill_priority(placed, o["dimension"])
            if pos and volume + o["volume"] <= MAX_CAPACITY:
                o.update(pos)
                placed.append(pos)
                packed.append(o)
                packed_ids.add(o["box_id"])
                volume += o["volume"]

        if not packed:
            break

        dests = list({p["destination"] for p in packed})
        route = get_vrp_route(dist_matrix, idx_map, locations, "Depot", dests)
        dest_rank = {d: i for i, d in enumerate(route)}

        results.append({"Vehicle_ID": vehicle_id, "Route_Order": 0, "Destination": "Depot",
                        "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
                        "X": 0, "Y": 0, "Z": 0, "Longitude": coords["Depot"]["longitude"],
                        "Latitude": coords["Depot"]["latitude"], "Width": 0, "Length": 0, "Height": 0})

        stacking_order = {}
        for dest in dests:
            boxes = [o for o in packed if o["destination"] == dest]
            boxes.sort(key=lambda x: -x["Z"])
            for rank, o in enumerate(boxes):
                stacking_order[o["box_id"]] = rank
                total_shuffle_cost += rank * SHUFFLE_COST_PER_BOX

        for i, o in enumerate(packed):
            results.append({
                "Vehicle_ID": vehicle_id,
                "Route_Order": dest_rank[o["destination"]] + 1,
                "Destination": o["destination"],
                "Order_Number": o["order_number"],
                "Box_ID": o["box_id"],
                "Stacking_Order": stacking_order.get(o["box_id"], 0),
                "X": o["X"], "Y": o["Y"], "Z": o["Z"],
                "Longitude": coords[o["destination"]]["longitude"],
                "Latitude": coords[o["destination"]]["latitude"],
                "Width": o["dimension"]["width"],
                "Length": o["dimension"]["length"],
                "Height": o["dimension"]["height"]
            })

        results.append({"Vehicle_ID": vehicle_id, "Route_Order": len(dest_rank)+1, "Destination": "Depot",
                        "Order_Number": "", "Box_ID": "", "Stacking_Order": "",
                        "X": 0, "Y": 0, "Z": 0, "Longitude": coords["Depot"]["longitude"],
                        "Latitude": coords["Depot"]["latitude"], "Width": 0, "Length": 0, "Height": 0})

        vehicle_id += 1
        current_orders = [o for o in current_orders if o["box_id"] not in packed_ids]

    df = pd.DataFrame(results)

    total_distance = 0
    for v in range(vehicle_id):
        df_v = df[df["Vehicle_ID"] == v].sort_values("Route_Order")
        route = df_v["Destination"].tolist()
        for i in range(len(route) - 1):
            total_distance += dist_matrix[idx_map[route[i]]][idx_map[route[i+1]]]

    fuel = int((total_distance / 1000) * FUEL_COST_PER_KM)
    fixed = vehicle_id * FIXED_COST_PER_VEHICLE
    processed_boxes = df[df["Box_ID"] != ""].drop_duplicates("Box_ID")

    save_result_excel(df)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
    else:
        data = load_json(sys.argv[1])
        matrix, locs, idmap = load_distance_data(sys.argv[2])
        solve(data, matrix, locs, idmap)

