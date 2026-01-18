"""
Basic usage example for the universal e-nose system
"""
from pandas import np

from src import UniversalENoseSystem
from src.utils import get_timestamp


def main():
    e_nose = UniversalENoseSystem()

    print("Adding training samples...")

    items_to_train = [
        ("Chanel No 5", "perfume"),
        ("Granny Smith Apple", "fruit"),
        ("Coffee", "beverage"),
        ("Vanilla", "spice")
    ]

    for item_name, category in items_to_train:
        samples = []
        for _ in range(30):
            duration = 80 if category == "perfume" else 30
            time_series = []
            for _ in range(duration):
                time_series.append(e_nose.simulate_sensor_reading(item_name))

            features = e_nose.signal_processor.extract_universal_features(
                np.array(time_series),
                mode=category if category != "perfume" else "perfume"
            )
            samples.append(features.tolist())

        for sample in samples:
            e_nose.training_data["samples"].append({
                "features": sample,
                "odor": item_name,
                "category": category,
                "timestamp": get_timestamp()
            })

        e_nose.training_data["odors"][item_name] = {
            "sample_count": len(samples),
            "category": category,
            "created": get_timestamp(),
            "last_updated": get_timestamp()
        }

    print("Training universal model...")
    e_nose.train_hierarchical_models()

    print("Performing universal recognition...")
    results = e_nose.recognize_universal(duration=8.0)
    e_nose.display_universal_results(results)


if __name__ == "__main__":
    main()