import json
import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import sys

sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class Box:
    order_number: str
    box_id: str
    destination: str
    width: int
    length: int
    height: int

    @property
    def volume(self):
        return self.width * self.length * self.height


@dataclass
class Vehicle:
    width: int = 160
    length: int = 280
    height: int = 180

    @property
    def volume(self):
        return self.width * self.length * self.height


class RoutingOptimizer:
    def __init__(self, data, distance_matrix):
        self.data = data
        self.distance_matrix = distance_matrix

    def solve(self, demands, max_vehicles=50):
        destinations = list(demands.keys())
        manager = pywrapcp.RoutingIndexManager(len(destinations) + 1, max_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            f = manager.IndexToNode(from_index)
            t = manager.IndexToNode(to_index)
            return self.distance_matrix[f][t]

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        def demand_callback(index):
            node = manager.IndexToNode(index)
            return 0 if node == 0 else min(demands.get(destinations[node - 1], 0), 8000000)

        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, [8064000] * max_vehicles, True, 'Capacity')

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 180

        solution = routing.SolveWithParameters(params)
        if not solution:
            return []

        routes = []
        for v in range(max_vehicles):
            idx = routing.Start(v)
            route, dist = [], 0
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    route.append(destinations[node - 1])
                prev_idx = idx
                idx = solution.Value(routing.NextVar(idx))
                dist += routing.GetArcCostForVehicle(prev_idx, idx, v)
            if route:
                routes.append({'vehicle_id': v, 'destinations': route, 'distance': dist / 1000})
        return routes


class LoadingOptimizer:
    def __init__(self, vehicle_dim):
        self.vehicle = Vehicle(**vehicle_dim)

    def can_fit(self, boxes):
        total_volume = sum(b.volume for b in boxes)
        return total_volume <= self.vehicle.volume

    def assign_positions(self, boxes):
        # 단순 stacking_order 기준 (위치는 예시 값으로 고정)
        positions = []
        for i, box in enumerate(reversed(boxes)):
            positions.append({
                'box_id': box.box_id,
                'stacking_order': i + 1,
                'x': (i % 4) * 0.3,
                'y': 0.6 if i % 2 == 0 else 0.0,
                'z': (i // 4) * 0.4
            })
        return positions

    def calculate_shuffling_cost(self, boxes, destinations):
        box_map = defaultdict(list)
        for box in boxes:
            box_map[box.destination].append(box)

        stacking_order = list(reversed(boxes))
        cost = 0
        visited = set()
        for dest in destinations:
            for box in box_map[dest]:
                shuffles = sum(1 for b in stacking_order if b.box_id not in visited and b != box)
                cost += shuffles * 500
                visited.add(box.box_id)
        return cost


def load_distance_matrix(file_path, num_nodes):
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'TIME_MIN' in line or not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                f_idx = 0 if parts[0] == 'Depot' else int(parts[0].split('_')[1])
                t_idx = 0 if parts[1] == 'Depot' else int(parts[1].split('_')[1])
                dist = int(float(parts[2]) * 1000)
                matrix[f_idx][t_idx] = dist
            except:
                continue
    return matrix


def main():
    start = time.time()
    with open('data.json', 'r') as f:
        data = json.load(f)

    boxes = [Box(o['order_number'], o['box_id'], o['destination'],
                 o['dimension']['width'], o['dimension']['length'], o['dimension']['height'])
             for o in data['orders']]

    dest_boxes = defaultdict(list)
    dest_demands = defaultdict(int)
    for b in boxes:
        dest_boxes[b.destination].append(b)
        dest_demands[b.destination] += b.volume

    location_map = {d['destination_id']: d['location'] for d in data['destinations']}

    matrix = load_distance_matrix('distance-data.txt', 301)
    routes = RoutingOptimizer(data, matrix).solve(dest_demands)

    results = []
    routing_cost, unloading_cost = 0, 0
    processed = set()
    loader = LoadingOptimizer(data['vehicles'][0]['dimension'])

    for route in routes:
        boxes_this_route = [b for d in route['destinations'] for b in dest_boxes[d]]
        if not loader.can_fit(boxes_this_route):
            boxes_this_route = boxes_this_route[:len(boxes_this_route) // 2]

        fuel = route['distance'] * 500
        cost = 150000 + fuel
        routing_cost += cost

        positions = loader.assign_positions(boxes_this_route)
        pos_map = {p['box_id']: p for p in positions}

        shuffle = loader.calculate_shuffling_cost(boxes_this_route, route['destinations'])
        unloading_cost += shuffle

        for i, dest in enumerate(route['destinations']):
            for box in dest_boxes[dest]:
                if box.box_id not in pos_map:
                    continue
                pos = pos_map[box.box_id]
                processed.add(box.box_id)
                loc = location_map.get(dest, {'longitude': 0.0, 'latitude': 0.0})
                results.append({
                    'Vehicle_ID': route['vehicle_id'],
                    'Route_Order': i + 1,
                    'Destination': dest,
                    'Order_Number': box.order_number,
                    'Box_ID': box.box_id,
                    'Stacking_Order': pos['stacking_order'],
                    'Lower_Left_X': pos['x'],
                    'Lower_Left_Y': pos['y'],
                    'Lower_Left_Z': pos['z'],
                    'Longitude': loc['longitude'],
                    'Latitude': loc['latitude'],
                    'Box_Width': box.width,
                    'Box_Length': box.length,
                    'Box_Height': box.height
                })

    df = pd.DataFrame(results)
    df.to_excel('Result.xlsx', index=False)
    print(f"총 라우팅 비용: {routing_cost:,.0f}원")
    print(f"총 하차 비용: {unloading_cost:,.0f}원")
    print(f"총 비용: {routing_cost + unloading_cost:,.0f}원")
    print(f"처리 박스 수: {len(processed)}/{len(boxes)}개")
    print(f"실행 시간: {time.time() - start:.2f}초")


if __name__ == '__main__':
    main()