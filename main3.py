import json
import math
import pandas as pd
from typing import Tuple, Dict, List
from itertools import permutations
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 데이터 파일 로드 (경로 수정 필요 시 여기에)
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 기본 정보
orders = data["orders"]
depot = data["depot"]
destinations_info = {d["destination_id"]: d["location"] for d in data["destinations"]}
TRUCK_DIMENSIONS = (160, 280, 180)
TRUCK_VOLUME = TRUCK_DIMENSIONS[0] * TRUCK_DIMENSIONS[1] * TRUCK_DIMENSIONS[2]

# 박스 부피 계산 함수
def calculate_volume(dim: Dict[str, float]) -> float:
    return dim["width"] * dim["length"] * dim["height"]

# 박스 리스트 생성 및 정렬
boxes = []
for order in orders:
    volume = calculate_volume(order["dimension"])
    boxes.append({
        "box_id": order["box_id"],
        "destination": order["destination"],
        "dimension": order["dimension"],
        "volume": volume
    })
boxes.sort(key=lambda x: x["volume"], reverse=True)

# 그리디 적재
trucks = []
current_truck = {"volume": 0, "boxes": []}
for box in boxes:
    if current_truck["volume"] + box["volume"] <= TRUCK_VOLUME:
        current_truck["boxes"].append(box)
        current_truck["volume"] += box["volume"]
    else:
        trucks.append(current_truck)
        current_truck = {"volume": box["volume"], "boxes": [box]}
if current_truck["boxes"]:
    trucks.append(current_truck)

# 거리 계산 (위경도 유클리디안 근사)
def geo_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = (a[0] - b[0]) * 88.74
    dy = (a[1] - b[1]) * 111.32
    return math.sqrt(dx**2 + dy**2)

# Nearest Neighbor 알고리즘
def nearest_neighbor_route(locations: List[Tuple[str, float, float]]) -> List[str]:
    route = []
    current = ("Depot", depot["location"]["longitude"], depot["location"]["latitude"])
    unvisited = locations[:]
    while unvisited:
        next_stop = min(unvisited, key=lambda x: geo_distance((current[1], current[2]), (x[1], x[2])))
        route.append(next_stop[0])
        current = next_stop
        unvisited.remove(next_stop)
    return route

# 결과 저장용 리스트
summary_rows = []
for i, truck in enumerate(trucks, 1):
    truck_id = f"Truck_{i}"
    dest_ids = list(set(box["destination"] for box in truck["boxes"]))
    dest_coords = [(d, destinations_info[d]["longitude"], destinations_info[d]["latitude"]) for d in dest_ids]
    route = nearest_neighbor_route(dest_coords)

    # 전체 거리 계산
    total_distance = 0
    current_loc = (depot["location"]["longitude"], depot["location"]["latitude"])
    for dest_id in route:
        next_loc = (destinations_info[dest_id]["longitude"], destinations_info[dest_id]["latitude"])
        total_distance += geo_distance(current_loc, next_loc)
        current_loc = next_loc

    # 비용 계산
    vehicle_cost = 150000
    fuel_cost = int(total_distance * 500)
    unloading_count = {d: 0 for d in dest_ids}
    for box in truck["boxes"]:
        unloading_count[box["destination"]] += 1
    unloading_cost = sum(v * 500 for v in unloading_count.values())
    total_cost = vehicle_cost + fuel_cost + unloading_cost

    for seq, dest_id in enumerate(route, 1):
        summary_rows.append({
            "Truck_ID": truck_id,
            "Stop_Sequence": seq,
            "Destination": dest_id,
            "Boxes_Delivered": unloading_count[dest_id],
            "Distance(km)": round(total_distance, 2),
            "Vehicle_Cost": vehicle_cost,
            "Fuel_Cost": fuel_cost,
            "Unloading_Cost": unloading_cost,
            "Total_Cost": total_cost
        })

# 결과 데이터프레임 생성
df_summary = pd.DataFrame(summary_rows)

# 트럭별 첫 행만 추출해 중복 비용 제거
truck_costs = df_summary.groupby("Truck_ID", as_index=False).first()[
    ["Vehicle_Cost", "Fuel_Cost", "Unloading_Cost", "Total_Cost"]
]

# 총합 행 추가
df_summary.loc["Total"] = {
    "Truck_ID": "총합",
    "Stop_Sequence": "",
    "Destination": "",
    "Boxes_Delivered": df_summary["Boxes_Delivered"].sum(),
    "Distance(km)": "",
    "Vehicle_Cost": truck_costs["Vehicle_Cost"].sum(),
    "Fuel_Cost": truck_costs["Fuel_Cost"].sum(),
    "Unloading_Cost": truck_costs["Unloading_Cost"].sum(),
    "Total_Cost": truck_costs["Total_Cost"].sum()
}

# 엑셀 저장
df_summary.to_excel("Result.xlsx", index=False)
print(" 배송 경로 및 비용 계산이 완료되었습니다. Result.xlsx 파일을 확인하세요.")