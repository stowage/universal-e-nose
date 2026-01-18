"""
Calibration example for the universal e-nose system
"""

from src import UniversalENoseSystem


def main():
    e_nose = UniversalENoseSystem()

    print("Performing baseline calibration...")
    e_nose.collect_baseline_calibration()

    if e_nose.training_data.get("baseline"):
        baseline = e_nose.training_data["baseline"]
        print(f"Baseline mean values: {baseline['mean']}")
        print(f"Baseline std values: {baseline['std']}")

    raw_reading = e_nose.simulate_sensor_reading()
    calibrated_reading = e_nose.apply_calibration(raw_reading)

    print(f"Raw reading: {raw_reading[:3]}...")
    print(f"Calibrated reading: {calibrated_reading[:3]}...")


if __name__ == "__main__":
    main()