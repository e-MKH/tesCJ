import json
import sys
import pandas as pd
from collections import defaultdict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
    n = len(sorted_locations)
    matrix = [[0]*n for _ in range(n)]
    for (o, d), v in distance_map.items():
        i, j = index_map[o], index_map[d]
        matrix[i][j] = v
        matrix[j][i] = v
    return matrix, sorted_locations, index_map

def split_orders_by_volume(orders):
    vehicles = []
    current = []
    volume_sum = 0
    for o in sorted(orders, key=lambda x: -x["volume"]):
        if volume_sum + o["volume"] > MAX_CAPACITY:
            vehicles.append(current)
            current = [o]
            volume_sum = o["volume"]
        else:
            current.append(o)
            volume_sum += o["volume"]
    if current:
        vehicles.append(current)
    return vehicles

def get_order_destinations(orders):
    return list(set([o["destination"] for o in orders]))

def get_coords_map(data):
    coords = {
        d["destination_id"]: {
            "longitude": d["location"]["longitude"],
            "latitude": d["location"]["latitude"]
        } for d in data["destinations"]
    }
    coords["Depot"] = {
        "longitude": data["depot"]["location"]["longitude"],
        "latitude": data["depot"]["location"]["latitude"]
    }
    return coords

def get_vrp_route(dist_matrix, idx_map, locations, depot_id, destination_ids):
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, idx_map[depot_id])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_params)

    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(locations[node])
            index = solution.Value(routing.NextVar(index))
        route.append(locations[manager.IndexToNode(index)])
    return route

def is_overlap(box1, box2):
    def overlap_1d(a1, a2, b1, b2):
        return max(a1, b1) < min(a2, b2)
    return (
        overlap_1d(box1["X"], box1["X"] + box1["Width"], box2["X"], box2["Width"] + box2["X"]) and
        overlap_1d(box1["Y"], box1["Y"] + box1["Length"], box2["Y"], box2["Length"] + box2["Y"]) and
        overlap_1d(box1["Z"], box1["Z"] + box1["Height"], box2["Z"], box2["Height"] + box2["Z"])
    )

def find_valid_position(placed_boxes, box_dim, res=10):
    max_x, max_y, max_z = 160, 280, 180
    for x in range(0, max_x - box_dim["width"] + 1, res):
        for y in range(0, max_y - box_dim["length"] + 1, res):
            for z in range(0, max_z - box_dim["height"] + 1, res):
                new_box = {
                    "X": x, "Y": y, "Z": z,
                    "Width": box_dim["width"],
                    "Length": box_dim["length"],
                    "Height": box_dim["height"]
                }
                if all(not is_overlap(new_box, b) for b in placed_boxes):
                    return new_box
    return None

def calculate_shuffle_cost(orders, dest_rank):
    orders_by_dest = sorted(orders, key=lambda o: dest_rank[o["destination"]])
    shuffle_count = 0
    for i, o in enumerate(orders_by_dest):
        my_y = o.get("Y", 0)
        for j in range(i):
            prev_y = orders_by_dest[j].get("Y", 0)
            if prev_y > my_y:
                shuffle_count += 1
    return shuffle_count * SHUFFLE_COST_PER_BOX

def insert_depot_rows(df: pd.DataFrame, depot_coord: dict) -> pd.DataFrame:
    new_rows = []
    for vehicle_id in df["Vehicle_ID"].unique():
        vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
        min_route = vehicle_df["Route_Order"].min()
        max_route = vehicle_df["Route_Order"].max()

        for order in [min_route - 1, max_route + 1]:
            new_rows.append({
                "Vehicle_ID": vehicle_id,
                "Route_Order": order,
                "Destination": "Depot",
                "Order_Number": "",
                "Box_ID": "",
                "Stacking_Order": "",
                "Lower_Left_X": "",
                "Lower_Left_Y": "",
                "Lower_Left_Z": "",
                "Longitude": depot_coord["longitude"],
                "Latitude": depot_coord["latitude"],
                "Box_Width": "",
                "Box_Length": "",
                "Box_Height": ""
            })
    df_combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df_combined.sort_values(by=["Vehicle_ID", "Route_Order"]).reset_index(drop=True)

def export_result_direct_format(df, output_file="Result.xlsx"):
    df = df.rename(columns={
        "X": "Lower_Left_X",
        "Y": "Lower_Left_Y",
        "Z": "Lower_Left_Z",
        "Width": "Box_Width",
        "Length": "Box_Length",
        "Height": "Box_Height"
    })
    desired_columns = [
        "Vehicle_ID", "Route_Order", "Destination", "Order_Number", "Box_ID",
        "Stacking_Order", "Lower_Left_X", "Lower_Left_Y", "Lower_Left_Z",
        "Longitude", "Latitude", "Box_Width", "Box_Length", "Box_Height"
    ]
    df = df[desired_columns]
    df.to_excel(output_file, index=False)

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        return

    json_file = sys.argv[1]
    dist_file = sys.argv[2]

    data = load_json(json_file)
    orders = data["orders"]
    for o in orders:
        o["volume"] = calculate_order_volume(o)

    dist_matrix, locations, idx_map = load_distance_data(dist_file)
    location_coords = get_coords_map(data)

    vehicle_orders = split_orders_by_volume(orders)

    all_rows = []
    total_fixed, total_fuel, total_shuffle = 0, 0, 0

    for v_id, v_orders in enumerate(vehicle_orders):
        dests = get_order_destinations(v_orders)
        route_ids = get_vrp_route(dist_matrix, idx_map, locations, "Depot", dests)
        dest_rank = {d: i for i, d in enumerate(route_ids)}

        placed = []
        for s_order, o in enumerate(v_orders):
            dim = o["dimension"]
            pos = find_valid_position(placed, dim)
            if not pos:
                continue
            placed.append(pos)
            o.update(pos)
            row = {
                "Vehicle_ID": v_id,
                "Route_Order": dest_rank[o["destination"]],
                "Destination": o["destination"],
                "Order_Number": o["order_number"],
                "Box_ID": o["box_id"],
                "Stacking_Order": s_order,
                "X": pos["X"],
                "Y": pos["Y"],
                "Z": pos["Z"],
                "Longitude": location_coords[o["destination"]]["longitude"],
                "Latitude": location_coords[o["destination"]]["latitude"],
                "Width": dim["width"],
                "Length": dim["length"],
                "Height": dim["height"]
            }
            all_rows.append(row)

        route_dist = 0
        for i in range(len(route_ids) - 1):
            route_dist += dist_matrix[idx_map[route_ids[i]]][idx_map[route_ids[i + 1]]]

        total_fuel += (route_dist / 1000) * FUEL_COST_PER_KM
        total_fixed += FIXED_COST_PER_VEHICLE
        total_shuffle += calculate_shuffle_cost(v_orders, dest_rank)

    df = pd.DataFrame(all_rows)
    df = insert_depot_rows(df, location_coords["Depot"])
    export_result_direct_format(df, "Result.xlsx")

    print(f"차량 수: {len(vehicle_orders)}")
    print(f"고정비용: {total_fixed:,}원")
    print(f"유류비: {int(total_fuel):,}원")
    print(f"셔플링비: {total_shuffle:,}원")
    print(f"총 비용: {int(total_fixed + total_fuel + total_shuffle):,}원")

if __name__ == "__main__":
    main()
