"""
Advanced E-Nose System Base Class
Base system class for odor training and recognition
"""

import numpy as np
import json
import os
import time
import warnings
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union
import joblib

from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .arduino_interface import ArduinoENoseInterface
from .signal_processing import SignalProcessor
from .utils import load_config, ensure_directory_exists, get_timestamp, safe_save_json

warnings.filterwarnings('ignore')


class AdvancedENoseSystem:
    """Advanced E-Nose System base class"""

    def __init__(self, config_file: str = "config/default_config.json"):
        self.config = load_config(config_file)
        self.data_dir = self.config["data"]["data_directory"]
        self.model_dir = self.config["data"]["model_directory"]

        ensure_directory_exists(self.data_dir)
        ensure_directory_exists(self.model_dir)

        self.data_file = os.path.join(self.data_dir, "e_nose_data_universal.json")
        self.model_file = os.path.join(self.model_dir, "universal_model.pkl")
        self.baseline_file = os.path.join(self.data_dir, "baseline_calibration.json")

        self.training_data = {"odors": {}, "samples": [], "baseline": None}
        self.model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.is_trained = False
        self.environmental_data = {"temperature": 25.0, "humidity": 50.0}

        self.signal_processor = SignalProcessor()
        self.load_all_data()

    def load_all_data(self):
        """Load training data, models, and calibration data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.training_data = json.load(f)
                print(f"✓ Loaded {len(self.training_data['samples'])} training samples")
            except Exception as e:
                print(f"⚠ Warning: Could not load training data: {e}")
                self.training_data = {"odors": {}, "samples": [], "baseline": None}

        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    self.training_data["baseline"] = baseline_data
                print("✓ Loaded baseline calibration data")
            except Exception as e:
                print(f"⚠ Warning: Could not load baseline data: {e}")

        if os.path.exists(self.model_file):
            try:
                model_data = joblib.load(self.model_file)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_selector = model_data.get('feature_selector')
                self.selected_features = model_data.get('selected_features')
                self.is_trained = model_data['is_trained']
                print("✓ Loaded trained model")
            except Exception as e:
                print(f"⚠ Warning: Could not load model: {e}")
                self.reset_model()
        else:
            self.reset_model()

    def reset_model(self):
        """Reset model components"""
        self.model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.is_trained = False

    def save_all_data(self):
        """Save all data and models"""
        try:
            safe_save_json(self.training_data, self.data_file)

            if self.training_data.get("baseline"):
                safe_save_json(self.training_data["baseline"], self.baseline_file)

            if self.is_trained and self.model is not None:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_selector': self.feature_selector,
                    'selected_features': self.selected_features,
                    'is_trained': self.is_trained
                }
                joblib.dump(model_data, self.model_file)

            print("✓ All data saved successfully")
        except Exception as e:
            print(f"✗ Error saving data: {e}")

    def preprocess_signal(self, raw_signal: np.ndarray, sensor_id: int = None) -> np.ndarray:
        """Advanced signal preprocessing with adaptive filtering"""
        return self.signal_processor.preprocess_signal(raw_signal, sensor_id)

    def calibrate_sensors(self, baseline_readings: List[List[float]]):
        """Perform comprehensive sensor calibration"""
        baseline_array = np.array(baseline_readings)

        baseline_stats = {
            "mean": np.mean(baseline_array, axis=0).tolist(),
            "std": np.std(baseline_array, axis=0).tolist(),
            "median": np.median(baseline_array, axis=0).tolist(),
            "timestamp": get_timestamp(),
            "sample_count": len(baseline_readings)
        }

        self.training_data["baseline"] = baseline_stats
        print(f"✓ Calibrated sensors with {len(baseline_readings)} baseline samples")

    def apply_calibration(self, raw_readings: List[float]) -> List[float]:
        """Apply calibration to raw sensor readings"""
        if not self.training_data.get("baseline"):
            print("⚠ Warning: No baseline calibration available")
            return raw_readings

        baseline = self.training_data["baseline"]
        calibrated = []

        for i, reading in enumerate(raw_readings):
            median_baseline = baseline["median"][i]
            std_baseline = baseline["std"][i]

            if std_baseline > 0.01:
                relative_response = (reading - median_baseline) / std_baseline
            else:
                relative_response = reading - median_baseline

            calibrated.append(relative_response)

        return calibrated

    def extract_comprehensive_features(self, sensor_data: np.ndarray) -> np.ndarray:
        """Extract comprehensive features from sensor array"""
        return self.signal_processor.extract_comprehensive_features(sensor_data)

    def simulate_sensor_reading(self, item_name: str = None, noise_level: float = 0.05) -> List[float]:
        """Simulate realistic sensor readings for testing"""
        item_profiles = {
            "Chanel No 5": [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.35, 0.45, 0.55, 0.65],
            "Dior J'adore": [0.75, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.35, 0.45, 0.55],
            "YSL Black Opium": [0.55, 0.65, 0.75, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.35],
            "Granny Smith Apple": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35],
            "Orange": [0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            "Coffee": [0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55],
            "Vanilla": [0.7, 0.5, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.25, 0.35, 0.45],
            "None": [0.1] * 11
        }

        if item_name and item_name in item_profiles:
            base_profile = np.array(item_profiles[item_name])
        else:
            base_profile = np.random.rand(11) * 0.3

        noise = np.random.normal(0, noise_level, 11)
        drift = np.random.normal(0, 0.02, 11)

        reading = base_profile + noise + drift
        reading = np.clip(reading, 0, 1.2)

        return reading.tolist()

    def collect_extended_time_series(self, duration: float) -> np.ndarray:
        """Collect extended time series for analysis (simulation mode)"""
        time_series = []
        start_time = time.time()

        while time.time() - start_time < duration:
            raw_reading = self.simulate_sensor_reading()
            time_series.append(raw_reading)
            time.sleep(0.1)

        return np.array(time_series)

    def get_sensor_reading(self):
        """Get sensor reading (to be overridden by hardware class)"""
        return self.simulate_sensor_reading()

    def collect_baseline_calibration(self):
        """Collect baseline (clean air) calibration data"""
        print("\n" + "=" * 50)
        print("BASELINE CALIBRATION")
        print("=" * 50)
        print("Ensure clean air environment (no odors present)")
        print("System will collect baseline readings for 60 seconds...")

        input("Press Enter when ready to start baseline calibration...")

        baseline_samples = []
        duration = 60
        start_time = time.time()

        print("Collecting baseline data...")
        while time.time() - start_time < duration:
            raw_data = self.simulate_sensor_reading(noise_level=0.01)
            baseline_samples.append(raw_data)
            time.sleep(0.5)
            print(f"Collected {len(baseline_samples)} baseline samples...", end='\r')

        self.calibrate_sensors(baseline_samples)
        self.save_all_data()
        print(f"\n✓ Baseline calibration completed with {len(baseline_samples)} samples")

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and optimize training data"""
        min_samples = self.config["data"]["min_samples_per_item"]
        if len(self.training_data["samples"]) < min_samples:
            raise ValueError(f"Insufficient training data. Need at least {min_samples} samples.")

        X = []
        y = []

        for sample in self.training_data["samples"]:
            X.append(sample["features"])
            y.append(sample["odor"])

        X = np.array(X)
        y = np.array(y)

        X_balanced, y_balanced = self.balance_dataset(X, y)

        max_features = 100
        X_selected, self.selected_features = self.select_optimal_features(X_balanced, y_balanced, max_features)

        return X_selected, y_balanced

    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE and undersampling"""
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())

        if max_count / min_count < 2:
            return X, y

        if min_count >= 6:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        class_counts_resampled = Counter(y_resampled)
        target_size = int(np.mean(list(class_counts_resampled.values())))

        if max(class_counts_resampled.values()) > target_size * 2:
            undersampler = RandomUnderSampler(
                sampling_strategy={cls: min(count, target_size * 2)
                                   for cls, count in class_counts_resampled.items()},
                random_state=42
            )
            X_final, y_final = undersampler.fit_resample(X_resampled, y_resampled)
        else:
            X_final, y_final = X_resampled, y_resampled

        print(f"Dataset balanced: {Counter(y)} -> {Counter(y_final)}")
        return X_final, y_final

    def select_optimal_features(self, X: np.ndarray, y: np.ndarray, max_features: int = 100) -> Tuple[
        np.ndarray, np.array]:
        """Select most discriminative features"""
        n_features_to_select = min(max_features, X.shape[1])

        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
        X_selected = selector.fit_transform(X, y)

        selected_indices = selector.get_support(indices=True)
        print(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")

        return X_selected, selected_indices

    def create_optimized_ensemble(self) -> VotingClassifier:
        """Create optimized ensemble model"""
        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=22,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=125,
            learning_rate=0.08,
            max_depth=12,
            random_state=42
        )

        svm = SVC(
            probability=True,
            kernel='rbf',
            C=12,
            gamma='scale',
            random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft',
            weights=[0.4, 0.4, 0.2]
        )

        return ensemble

    def train_optimized_model(self):
        """Train the optimized model with comprehensive validation"""
        try:
            print("\n" + "=" * 60)
            print("UNIVERSAL MODEL TRAINING")
            print("=" * 60)

            X, y = self.prepare_training_data()
            print(f"Training data shape: {X.shape}")
            print(f"Classes: {np.unique(y)}")

            X_scaled = self.scaler.fit_transform(X)

            self.model = self.create_optimized_ensemble()

            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')

            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            self.model.fit(X_scaled, y)

            y_pred = self.model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            print(f"Training accuracy: {accuracy:.4f}")

            self.is_trained = True
            self.save_all_data()

            print("✓ Universal model training completed successfully!")
            return True

        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return False

    def main_menu(self):
        """Main menu - to be overridden by universal system"""
        print("Base system main menu")

    def calibration_mode(self):
        """Calibration management"""
        print("\n" + "-" * 30)
        print("CALIBRATION MODE")
        print("-" * 30)
        print("1. Perform baseline calibration")
        print("2. View calibration status")
        print("3. Back to main menu")

        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            self.collect_baseline_calibration()
        elif choice == '2':
            if self.training_data.get("baseline"):
                baseline = self.training_data["baseline"]
                print(f"Baseline calibration:")
                print(f"  Samples: {baseline['sample_count']}")
                print(f"  Timestamp: {baseline['timestamp']}")
                print(f"  Mean response: {[f'{x:.3f}' for x in baseline['mean'][:3]]}...")
            else:
                print("No baseline calibration found")
        elif choice == '3':
            return
        else:
            print("Invalid choice!")

    def data_management_mode(self):
        """Advanced data management"""
        print("\n" + "-" * 30)
        print("DATA MANAGEMENT")
        print("-" * 30)
        print("1. View training statistics")
        print("2. Export training data")
        print("3. Clear all data")
        print("4. Back to main menu")

        choice = input("Enter choice (1-4): ").strip()

        if choice == '1':
            self.view_training_statistics()
        elif choice == '2':
            self.export_training_data()
        elif choice == '3':
            self.clear_all_data()
        elif choice == '4':
            return
        else:
            print("Invalid choice!")

    def view_training_statistics(self):
        """Display comprehensive training statistics"""
        print("\n" + "=" * 50)
        print("TRAINING STATISTICS")
        print("=" * 50)

        if not self.training_data["odors"]:
            print("No training data available.")
            return

        print(f"Total samples: {len(self.training_data['samples'])}")
        print(f"Item types: {len(self.training_data['odors'])}")
        print(f"Features per sample: {len(self.training_data['samples'][0]['features']) if self.training_data['samples'] else 'N/A'}")
        print()

        print("Per-item statistics:")
        for item, info in self.training_data["odors"].items():
            count = info['sample_count']
            category = info.get('category', 'Unknown')
            print(f"  {item}: {count} samples | Category: {category}")

        if self.is_trained:
            print(f"\nModel status: Trained")
            print(f"Selected features: {len(self.selected_features) if self.selected_features is not None else 'All'}")

    def export_training_data(self):
        """Export training data to CSV"""
        if not self.training_data["samples"]:
            print("No data to export!")
            return

        try:
            import pandas as pd

            data_rows = []
            for sample in self.training_data["samples"]:
                row = sample["features"] + [sample["odor"]]
                data_rows.append(row)

            feature_names = [f"Feature_{i}" for i in range(len(data_rows[0]) - 1)] + ["Item"]
            df = pd.DataFrame(data_rows, columns=feature_names)

            filename = f"data/universal_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Training data exported to: {filename}")

        except ImportError:
            print("✗ pandas not installed. Install with: pip install pandas")
        except Exception as e:
            print(f"✗ Export failed: {e}")

    def clear_all_data(self):
        """Clear all data with confirmation"""
        confirm = input("⚠ This will delete ALL data! Type 'DELETE' to confirm: ").strip()
        if confirm == 'DELETE':
            self.training_data = {"odors": {}, "samples": [], "baseline": None}
            self.reset_model()

            for file_path in [self.data_file, self.model_file, self.baseline_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            print("✓ All data cleared!")
        else:
            print("Operation cancelled.")

    def configuration_mode(self):
        """System configuration management"""
        print("\n" + "-" * 30)
        print("CONFIGURATION")
        print("-" * 30)
        print("Current settings:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")

        print("\n1. Edit configuration")
        print("2. Reset to defaults")
        print("3. Back to main menu")

        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            self.edit_configuration()
        elif choice == '2':
            self.reset_configuration()
        elif choice == '3':
            return
        else:
            print("Invalid choice!")

    def edit_configuration(self):
        """Edit configuration parameters"""
        print("\nEdit configuration parameters:")
        for key, value in self.config.items():
            if not isinstance(value, dict):
                new_value = input(f"{key} [{value}]: ").strip()
                if new_value:
                    try:
                        if isinstance(value, int):
                            self.config[key] = int(new_value)
                        elif isinstance(value, float):
                            self.config[key] = float(new_value)
                        elif isinstance(value, bool):
                            self.config[key] = new_value.lower() in ['true', '1', 'yes']
                        else:
                            self.config[key] = new_value
                    except ValueError:
                        self.config[key] = new_value

        safe_save_json(self.config, "config/default_config.json")
        print("✓ Configuration updated!")

    def reset_configuration(self):
        """Reset configuration to defaults"""
        confirm = input("Reset to default configuration? (y/n): ").strip().lower()
        if confirm == 'y':
            default_config = load_config("config/default_config.json")
            self.config = default_config
            print("✓ Configuration reset to defaults!")

    def save_uncertain_sample(self, results: Dict):
        """Save uncertain recognition results for training"""
        print("\nSaving uncertain sample for training...")
        print("Available items:")
        item_names = list(self.training_data["odors"].keys())

        for i, item in enumerate(item_names, 1):
            print(f"{i}. {item}")
        print(f"{len(item_names) + 1}. New item")

        try:
            choice = int(input("Select correct item (or new): ")) - 1

            if choice == len(item_names):
                new_item = input("Enter new item name: ").strip()
                if new_item:
                    correct_item = new_item
                    if new_item not in self.training_data["odors"]:
                        self.training_data["odors"][new_item] = {
                            "sample_count": 0,
                            "created": get_timestamp(),
                            "last_updated": get_timestamp()
                        }
                else:
                    print("Invalid item name!")
                    return
            elif 0 <= choice < len(item_names):
                correct_item = item_names[choice]
            else:
                print("Invalid selection!")
                return

            self.training_data["samples"].append({
                "features": results.get("raw_features", results["predictions"][0].get("features", [])),
                "odor": correct_item,
                "timestamp": get_timestamp()
            })

            self.training_data["odors"][correct_item]["sample_count"] += 1
            self.training_data["odors"][correct_item]["last_updated"] = get_timestamp()

            self.save_all_data()
            print(f"✓ Added sample to '{correct_item}'")

        except ValueError:
            print("Invalid input!")