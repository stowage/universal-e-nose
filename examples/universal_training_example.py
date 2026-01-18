"""
Universal training example showing perfume and fruit recognition
"""
from pandas import np

from src import UniversalENoseSystem
from src.utils import get_timestamp


def main():
    e_nose = UniversalENoseSystem()

    print("=== UNIVERSAL TRAINING DEMONSTRATION ===")

    training_items = [
        {"name": "Chanel No 5", "category": "perfume", "samples": 50},
        {"name": "Dior J'adore", "category": "perfume", "samples": 50},
        {"name": "Granny Smith Apple", "category": "fruit", "samples": 30},
        {"name": "Orange", "category": "fruit", "samples": 30},
        {"name": "Coffee", "category": "beverage", "samples": 25},
        {"name": "Vanilla", "category": "spice", "samples": 25}
    ]

    for item_info in training_items:
        print(f"\nAdding {item_info['name']} ({item_info['category']})...")

        samples = []
        duration = 8.0 if item_info['category'] == 'perfume' else 3.0

        for _ in range(item_info['samples']):
            time_series = []
            time_points = int(duration * 10)
            for _ in range(time_points):
                time_series.append(e_nose.simulate_sensor_reading(item_info['name']))

            features = e_nose.signal_processor.extract_universal_features(
                np.array(time_series),
                mode=item_info['category'] if item_info['category'] != 'perfume' else 'perfume'
            )
            samples.append(features.tolist())

        for sample in samples:
            e_nose.training_data["samples"].append({
                "features": sample,
                "odor": item_info['name'],
                "category": item_info['category'],
                "timestamp": get_timestamp()
            })

        e_nose.training_data["odors"][item_info['name']] = {
            "sample_count": len(samples),
            "category": item_info['category'],
            "created": get_timestamp(),
            "last_updated": get_timestamp()
        }

    total_samples = sum(item['samples'] for item in training_items)
    print(f"\nAdded {len(training_items)} items with {total_samples} total samples")

    print("\nTraining universal hierarchical model...")
    success = e_nose.train_hierarchical_models()

    if success:
        print("\nModel trained successfully!")

        test_items = ["Chanel No 5", "Granny Smith Apple", "Coffee"]

        for test_item in test_items:
            print(f"\n--- Testing {test_item} ---")
            results = e_nose.recognize_universal(duration=8.0)
            e_nose.display_universal_results(results)


if __name__ == "__main__":
    main()