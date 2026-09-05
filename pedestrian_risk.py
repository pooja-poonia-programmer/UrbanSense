import math


def calculate_pedestrian_risk(people, vehicles):

    risk_detected = False
    closest_distance = None

    for person in people:

        px1, py1, px2, py2 = person

        person_center_x = (px1 + px2) // 2
        person_center_y = (py1 + py2) // 2

        for vehicle in vehicles:

            vx1, vy1, vx2, vy2 = vehicle

            vehicle_center_x = (vx1 + vx2) // 2
            vehicle_center_y = (vy1 + vy2) // 2

            distance = math.sqrt(
                (person_center_x - vehicle_center_x) ** 2
                +
                (person_center_y - vehicle_center_y) ** 2
            )

            if closest_distance is None:
                closest_distance = distance
            else:
                closest_distance = min(
                    closest_distance,
                    distance
                )

            if distance < 150:
                risk_detected = True

    if risk_detected:
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    return {
        "risk": risk_detected,
        "level": risk_level,
        "people": len(people),
        "vehicles": len(vehicles),
        "closest_distance": closest_distance
    }