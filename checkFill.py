import pandas as pd

# 경로에 맞게 수정 (main.py와 같은 폴더일 경우 그냥 "Result.xlsx")
df = pd.read_excel("Result.xlsx")

def validate_truck_fill(df):
    from collections import defaultdict

    TRUCK_VOLUME = 160 * 280 * 180
    vehicle_stats = defaultdict(lambda: {"used_volume": 0, "over_bound": 0, "box_count": 0})

    for _, row in df[df["Box_ID"] != ""].iterrows():
        v_id = row["Vehicle_ID"]
        x, y, z = row["Lower_Left_X"], row["Lower_Left_Y"], row["Lower_Left_Z"]
        w, l, h = row["Box_Width"], row["Box_Length"], row["Box_Height"]

        vehicle_stats[v_id]["box_count"] += 1
        volume = w * l * h
        vehicle_stats[v_id]["used_volume"] += volume

        # 트럭 경계 체크
        if x + w > 160 or y + l > 280 or z + h > 180:
            vehicle_stats[v_id]["over_bound"] += 1

    print(f"{'Vehicle':>8} | {'Boxes':>5} | {'Used(m³)':>10} | {'Rate(%)':>8} | {'OutOfBound':>11}")
    print("-"*55)
    for v, stat in sorted(vehicle_stats.items()):
        rate = stat["used_volume"] / TRUCK_VOLUME * 100
        print(f"{v:>8} | {stat['box_count']:>5} | {stat['used_volume']:>10,} | {rate:>7.2f}% | {stat['over_bound']:>11}")

validate_truck_fill(df)
