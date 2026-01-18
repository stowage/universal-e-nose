"""
Universal Odor Recognition System
Handles perfumes, fruits, beverages, and everything in between
"""
import time

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from .e_nose_system import AdvancedENoseSystem
from .signal_processing import UniversalSignalProcessor
from .utils import get_timestamp


class UniversalENoseSystem(AdvancedENoseSystem):
    """
    Universal odor recognition system that handles all types of odors
    """

    def __init__(self, config_file: str = "config/default_config.json"):
        super().__init__(config_file)
        self.signal_processor = UniversalSignalProcessor()
        self.category_model = None
        self.perfume_model = None
        self.general_model = None
        self.is_hierarchical = self.config["model"]["use_hierarchical_classification"]

    def prepare_training_data_universal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for hierarchical classification
        Returns: (X_features, y_items, y_categories)
        """
        min_samples = self.config["data"]["min_samples_per_item"]
        if len(self.training_data["samples"]) < min_samples:
            raise ValueError(f"Insufficient training data.")

        X = []
        y_items = []
        y_categories = []

        for sample in self.training_data["samples"]:
            X.append(sample["features"])
            y_items.append(sample["odor"])
            y_categories.append(sample.get("category", "unknown"))

        X = np.array(X)
        y_items = np.array(y_items)
        y_categories = np.array(y_categories)

        return X, y_items, y_categories

    def train_hierarchical_models(self):
        """Train hierarchical classification models"""
        try:
            print("\n" + "=" * 60)
            print("UNIVERSAL HIERARCHICAL MODEL TRAINING")
            print("=" * 60)

            X, y_items, y_categories = self.prepare_training_data_universal()
            X_scaled = self.scaler.fit_transform(X)

            self.category_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42
            )
            self.category_model.fit(X_scaled, y_categories)

            unique_categories = np.unique(y_categories)

            for category in unique_categories:
                category_mask = y_categories == category
                X_cat = X_scaled[category_mask]
                y_cat = y_items[category_mask]

                if len(np.unique(y_cat)) > 1:
                    if category == "perfume":
                        self.perfume_model = self.create_perfume_ensemble()
                        self.perfume_model.fit(X_cat, y_cat)
                    elif category == "general":
                        self.general_model = self.create_general_ensemble()
                        self.general_model.fit(X_cat, y_cat)
                    else:
                        universal_model = self.create_universal_ensemble()
                        universal_model.fit(X_cat, y_cat)
                        setattr(self, f"{category}_model", universal_model)

            self.is_trained = True
            self.save_all_data()

            print("✓ Universal hierarchical models trained successfully!")
            return True

        except Exception as e:
            print(f"✗ Hierarchical model training failed: {e}")
            return False

    def create_perfume_ensemble(self) -> VotingClassifier:
        """Create ensemble optimized for perfumes"""
        rf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=15, random_state=42)
        svm = SVC(probability=True, kernel='rbf', C=15, gamma='scale', random_state=42)

        weights = self.config["model"]["ensemble_weights"]["perfume"]
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft',
            weights=weights
        )

    def create_general_ensemble(self) -> VotingClassifier:
        """Create ensemble optimized for general odors"""
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
        svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42)

        weights = self.config["model"]["ensemble_weights"]["general"]
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft',
            weights=weights
        )

    def create_universal_ensemble(self) -> VotingClassifier:
        """Create universal ensemble for unknown categories"""
        rf = RandomForestClassifier(n_estimators=250, max_depth=22, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=125, learning_rate=0.08, max_depth=12, random_state=42)
        svm = SVC(probability=True, kernel='rbf', C=12, gamma='scale', random_state=42)

        weights = self.config["model"]["ensemble_weights"]["universal"]
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft',
            weights=weights
        )

    def recognize_universal(self, duration: Optional[float] = None) -> Dict:
        """Universal recognition that handles all odor types"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")

        if duration is None:
            if self.config["recognition"]["adaptive_duration"]:
                duration = self.config["recognition"]["max_duration_seconds"]
            else:
                duration = self.config["recognition"]["min_duration_seconds"]

        print(f"\nAnalyzing odor for {duration} seconds...")

        time_series = self.collect_extended_time_series(duration)
        features = self.signal_processor.extract_universal_features(time_series, mode="adaptive")

        if self.selected_features is not None:
            features = features[self.selected_features]
        features_scaled = self.scaler.transform([features])

        category_probs = self.category_model.predict_proba(features_scaled)[0]
        categories = self.category_model.classes_
        category_results = list(zip(categories, category_probs))
        category_results.sort(key=lambda x: x[1], reverse=True)

        best_category = category_results[0][0]
        category_confidence = category_results[0][1]

        if category_confidence >= self.config["model"]["category_threshold"]:
            if best_category == "perfume" and self.perfume_model:
                item_probs = self.perfume_model.predict_proba(features_scaled)[0]
                item_classes = self.perfume_model.classes_
            elif best_category == "general" and self.general_model:
                item_probs = self.general_model.predict_proba(features_scaled)[0]
                item_classes = self.general_model.classes_
            else:
                item_probs = self.model.predict_proba(features_scaled)[0]
                item_classes = self.model.classes_
        else:
            item_probs = self.model.predict_proba(features_scaled)[0]
            item_classes = self.model.classes_
            best_category = "mixed"

        item_results = []
        for item_class, prob in zip(item_classes, item_probs):
            item_results.append({
                "item": item_class,
                "probability": float(prob),
                "percentage": float(prob * 100),
                "category": best_category
            })

        item_results.sort(key=lambda x: x["probability"], reverse=True)

        best_prob = item_results[0]["probability"]
        confidence_level = self.assess_universal_confidence(best_prob, category_confidence, best_category)

        return {
            "predictions": item_results,
            "best_match": item_results[0]["item"],
            "category": best_category,
            "category_confidence": category_confidence,
            "confidence": best_prob,
            "confidence_level": confidence_level,
            "analysis": self.generate_universal_analysis(item_results, best_category, category_confidence)
        }

    def assess_universal_confidence(self, item_confidence: float, category_confidence: float, category: str) -> str:
        """Assess confidence for universal recognition"""
        if category == "perfume":
            if item_confidence >= 0.85 and category_confidence >= 0.8:
                return "High"
            elif item_confidence >= 0.70 and category_confidence >= 0.6:
                return "Medium"
            else:
                return "Low"
        elif category == "general":
            if item_confidence >= 0.80 and category_confidence >= 0.7:
                return "High"
            elif item_confidence >= 0.65 and category_confidence >= 0.5:
                return "Medium"
            else:
                return "Low"
        else:
            if item_confidence >= 0.75 and category_confidence >= 0.6:
                return "High"
            elif item_confidence >= 0.60 and category_confidence >= 0.4:
                return "Medium"
            else:
                return "Low"

    def generate_universal_analysis(self, results: List[Dict], category: str, category_confidence: float) -> Dict:
        """Generate comprehensive analysis for universal recognition"""
        top_result = results[0]

        analysis = {
            "primary_match": top_result["item"],
            "detected_category": category,
            "category_confidence": category_confidence,
            "confidence_assessment": "High" if top_result["probability"] >= 0.85 else "Medium" if top_result[
                                                                                                      "probability"] >= 0.70 else "Low",
            "similar_items": [r["item"] for r in results[1:4] if r["probability"] > 0.1],
            "recommendation": self.get_universal_recommendation(top_result, category, category_confidence)
        }

        return analysis

    def get_universal_recommendation(self, result: Dict, category: str, category_confidence: float) -> str:
        """Provide recommendation based on universal recognition results"""
        item_confidence = result["probability"]

        if category_confidence >= 0.8:
            if category == "perfume":
                if item_confidence >= 0.85:
                    return "High confidence perfume identification - reliable match"
                elif item_confidence >= 0.70:
                    return "Medium confidence perfume match - consider verification"
                else:
                    return "Low confidence perfume identification - may need more training"
            else:
                if item_confidence >= 0.80:
                    return "High confidence general odor identification - reliable match"
                elif item_confidence >= 0.65:
                    return "Medium confidence general odor match - reasonably reliable"
                else:
                    return "Low confidence general odor identification - consider additional sampling"
        else:
            return f"Mixed category detection (category: {category}) - result may be less reliable"

    def display_universal_results(self, results: Dict):
        """Display universal recognition results"""
        print("\n" + "=" * 70)
        print("UNIVERSAL ODOR RECOGNITION RESULTS")
        print("=" * 70)
        print(f"Best Match: {results['best_match']}")
        print(f"Category: {results['category'].title()}")
        print(f"Category Confidence: {results['category_confidence']:.1%}")
        print(f"Item Confidence: {results['confidence']:.1%} ({results['confidence_level']} confidence)")
        print("-" * 70)
        print("Detailed Probabilities:")
        print("-" * 70)

        for pred in results["predictions"][:5]:
            percentage = pred["percentage"]
            bar_length = min(40, int(percentage / 2.5))
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{pred['item']:<25} | {percentage:5.1f}% | {bar}")

        print("-" * 70)

        analysis = results["analysis"]
        print(f"\nANALYSIS:")
        print(f"Primary Match: {analysis['primary_match']}")
        print(f"Detected Category: {analysis['detected_category'].title()}")
        print(f"Category Confidence: {analysis['category_confidence']:.1%}")
        print(f"Confidence Assessment: {analysis['confidence_assessment']}")
        if analysis['similar_items']:
            print(f"Similar Items: {', '.join(analysis['similar_items'])}")
        print(f"Recommendation: {analysis['recommendation']}")

    def add_universal_item(self, item_name: str, category: str = "auto"):
        """Add any type of item (perfume, fruit, etc.)"""
        print(f"\nAdding '{item_name}' as {category if category != 'auto' else 'auto-detected category'}...")

        if category == "auto":
            duration = self.config["recognition"]["max_duration_seconds"]
            num_samples = 30
        elif category == "perfume":
            duration = 8.0
            num_samples = 50
        else:
            duration = 3.0
            num_samples = 20

        samples = self.collect_universal_samples(item_name, num_samples, duration)

        if category == "auto":
            category = "unknown"

        for sample_features in samples:
            self.training_data["samples"].append({
                "features": sample_features,
                "odor": item_name,
                "category": category,
                "timestamp": get_timestamp()
            })

        if item_name not in self.training_data["odors"]:
            self.training_data["odors"][item_name] = {
                "sample_count": 0,
                "category": category,
                "created": get_timestamp(),
                "last_updated": get_timestamp()
            }

        self.training_data["odors"][item_name]["sample_count"] += len(samples)
        self.training_data["odors"][item_name]["last_updated"] = get_timestamp()

        self.save_all_data()
        print(f"✓ Added {len(samples)} samples for '{item_name}' as {category} category")

    def collect_universal_samples(self, item_name: str, num_samples: int, duration: float) -> List[List[float]]:
        """Collect samples with universal approach"""
        print(f"Collecting {num_samples} samples for '{item_name}' (duration: {duration}s)...")
        input("Press Enter when ready to start sampling...")

        samples = []
        for sample_idx in range(num_samples):
            print(f"Sample {sample_idx + 1}/{num_samples}...")

            time_series = []
            start_time = time.time()
            while time.time() - start_time < duration:
                reading = self.get_sensor_reading()
                time_series.append(reading)
                time.sleep(0.1)

            if time_series:
                features = self.signal_processor.extract_universal_features(
                    np.array(time_series),
                    mode="adaptive"
                )
                samples.append(features.tolist())

            time.sleep(0.5)

        return samples

    def main_menu(self):
        """Universal main menu"""
        while True:
            print("\n" + "=" * 60)
            print("UNIVERSAL ODOR RECOGNITION SYSTEM")
            print("=" * 60)
            print(f"Status: {'✓ Trained' if self.is_trained else '✗ Not trained'}")
            print(f"Training samples: {len(self.training_data['samples'])}")
            print(f"Known items: {list(self.training_data['odors'].keys()) if self.training_data['odors'] else 'None'}")
            if self.training_data.get("baseline"):
                print(f"✓ Baseline calibrated ({self.training_data['baseline']['sample_count']} samples)")
            else:
                print("⚠ No baseline calibration")
            print("-" * 60)
            print("1. Universal Training Mode")
            print("2. Universal Recognition Mode")
            print("3. Calibration")
            print("4. Data Management")
            print("5. System Configuration")
            print("6. Exit")
            print("-" * 60)

            choice = input("Enter your choice (1-6): ").strip()

            if choice == '1':
                self.universal_training_mode()
            elif choice == '2':
                self.universal_recognition_mode()
            elif choice == '3':
                self.calibration_mode()
            elif choice == '4':
                self.data_management_mode()
            elif choice == '5':
                self.configuration_mode()
            elif choice == '6':
                self.save_all_data()
                print("Goodbye!")
                break
            else:
                print("✗ Invalid choice. Please enter 1-6.")

    def universal_training_mode(self):
        """Universal training mode"""
        print("\n" + "=" * 50)
        print("UNIVERSAL TRAINING MODE")
        print("=" * 50)
        print("This system automatically adapts to any odor type:")
        print("• Perfumes: Complex, evolving scents")
        print("• Fruits: Fresh, steady aromas")
        print("• Beverages, spices, flowers, chemicals")
        print("• Unknown: Anything in between")
        print("-" * 50)

        while True:
            print("\nTraining Options:")
            print("1. Add new item (auto-detect category)")
            print("2. Add new item (specify category)")
            print("3. Add samples to existing item")
            print("4. Train universal model")
            print("5. Back to main menu")

            choice = input("Enter choice (1-5): ").strip()

            if choice == '1':
                self.add_item_auto_category()
            elif choice == '2':
                self.add_item_specify_category()
            elif choice == '3':
                self.add_samples_to_existing_item()
            elif choice == '4':
                if self.is_hierarchical:
                    self.train_hierarchical_models()
                else:
                    self.train_optimized_model()
            elif choice == '5':
                break
            else:
                print("Invalid choice!")

    def add_item_auto_category(self):
        """Add item with automatic category detection"""
        item_name = input("Enter item name: ").strip()
        if not item_name:
            print("Item name cannot be empty!")
            return

        self.add_universal_item(item_name, category="auto")

    def add_item_specify_category(self):
        """Add item with specified category"""
        item_name = input("Enter item name: ").strip()
        if not item_name:
            print("Item name cannot be empty!")
            return

        print("Available categories:")
        print("1. Perfume")
        print("2. Fruit")
        print("3. Beverage")
        print("4. Spice")
        print("5. Flower")
        print("6. Chemical")
        print("7. Other")

        cat_choice = input("Select category (1-7): ").strip()
        category_map = {
            "1": "perfume", "2": "fruit", "3": "beverage",
            "4": "spice", "5": "flower", "6": "chemical", "7": "other"
        }
        category = category_map.get(cat_choice, "other")

        self.add_universal_item(item_name, category=category)

    def add_samples_to_existing_item(self):
        """Add more samples to an existing item"""
        if not self.training_data["odors"]:
            print("✗ No existing items found. Please add a new item first.")
            return

        print("\nExisting items:")
        for i, item in enumerate(self.training_data["odors"].keys(), 1):
            count = self.training_data["odors"][item]["sample_count"]
            category = self.training_data["odors"][item].get("category", "Unknown")
            print(f"{i}. {item} ({count} samples, {category})")

        try:
            choice = int(input("Select item number to add samples to: ")) - 1
            item_names = list(self.training_data["odors"].keys())
            if 0 <= choice < len(item_names):
                selected_item = item_names[choice]
                self.collect_additional_item_samples(selected_item)
            else:
                print("✗ Invalid selection!")
        except ValueError:
            print("✗ Invalid input!")

    def collect_additional_item_samples(self, item_name: str):
        """Collect additional samples for existing item"""
        print(f"\nAdding samples to '{item_name}'")
        print("Place the item sample near the sensors and press Enter when ready...")
        input("Press Enter to continue...")

        try:
            num_samples = int(input("How many additional samples? (default: 10): ") or "10")
            if num_samples <= 0:
                num_samples = 10
        except ValueError:
            num_samples = 10

        category = self.training_data["odors"][item_name].get("category", "general")
        if category == "perfume":
            duration = 8.0
        else:
            duration = 3.0

        new_samples = self.collect_universal_samples(item_name, num_samples, duration)

        for sample in new_samples:
            self.training_data["samples"].append({
                "features": sample,
                "odor": item_name,
                "category": category,
                "timestamp": get_timestamp()
            })

        self.training_data["odors"][item_name]["sample_count"] += len(new_samples)
        self.training_data["odors"][item_name]["last_updated"] = get_timestamp()

        print(f"✓ Added {len(new_samples)} samples to '{item_name}'")
        self.save_all_data()

    def universal_recognition_mode(self):
        """Universal recognition mode"""
        if not self.is_trained:
            print("✗ Model is not trained! Please train the model first.")
            return

        print("\n" + "=" * 50)
        print("UNIVERSAL RECOGNITION MODE")
        print("=" * 50)
        print("Place the unknown sample near the sensors.")
        print("The system will automatically determine the best approach.")
        print("Examples: Chanel No. 5, Granny Smith Apple, Coffee, Vanilla, etc.")

        input("Press Enter when ready to start recognition...")

        try:
            results = self.recognize_universal()
            self.display_universal_results(results)

            if results["confidence_level"] == "Low":
                save_choice = input("\nSave this sample for training? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_uncertain_sample_universal(results)

        except Exception as e:
            print(f"✗ Recognition failed: {e}")

    def save_uncertain_sample_universal(self, results: Dict):
        """Save uncertain samples with category information"""
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
                    category = results.get("category", "unknown")
                    if new_item not in self.training_data["odors"]:
                        self.training_data["odors"][new_item] = {
                            "sample_count": 0,
                            "category": category,
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
                "category": self.training_data["odors"][correct_item].get("category", "unknown"),
                "timestamp": get_timestamp()
            })

            self.training_data["odors"][correct_item]["sample_count"] += 1
            self.training_data["odors"][correct_item]["last_updated"] = get_timestamp()

            self.save_all_data()
            print(f"✓ Added sample to '{correct_item}'")

        except ValueError:
            print("Invalid input!")


class UniversalHardwareENoseSystem(UniversalENoseSystem):
    """Hardware-enabled universal system"""

    def __init__(self, arduino_port: str = 'COM3', config_file: str = "config/default_config.json"):
        super().__init__(config_file)
        self.arduino = ArduinoENoseInterface(port=arduino_port)
        self.hardware_connected = False

    def connect_hardware(self) -> bool:
        """Connect to Arduino hardware"""
        if self.arduino.connect():
            self.hardware_connected = True
            return True
        return False

    def disconnect_hardware(self):
        """Disconnect from Arduino hardware"""
        self.arduino.disconnect()
        self.hardware_connected = False

    def get_sensor_reading(self):
        """Get sensor reading from real hardware"""
        if not self.hardware_connected:
            if not self.connect_hardware():
                print("⚠ Hardware not available, using simulation")
                return self.simulate_sensor_reading()

        sensor_data = self.arduino.read_single_sample()
        if sensor_data is None or len(sensor_data) != 11:
            print("⚠ Sensor reading failed, using simulation")
            return self.simulate_sensor_reading()

        return sensor_data

    def collect_extended_time_series(self, duration: float) -> np.ndarray:
        """Collect extended time series from real hardware"""
        if not self.hardware_connected:
            if not self.connect_hardware():
                print("⚠ Hardware not available, using simulation")
                return super().collect_extended_time_series(duration)

        try:
            time_series = self.arduino.collect_extended_time_series(duration)
            if time_series:
                return np.array(time_series)
            else:
                print("⚠ Hardware time series collection failed, using simulation")
                return super().collect_extended_time_series(duration)
        except Exception as e:
            print(f"⚠ Hardware collection error: {e}, using simulation")
            return super().collect_extended_time_series(duration)

    def collect_baseline_calibration(self):
        """Perform calibration using Arduino hardware"""
        if not self.hardware_connected:
            if not self.connect_hardware():
                print("✗ Cannot perform calibration without hardware connection")
                return

        print("\n" + "=" * 50)
        print("ARDUINO BASELINE CALIBRATION")
        print("=" * 50)
        print("Arduino will perform baseline calibration automatically")
        print("Ensure clean air environment (no odors present)")

        input("Press Enter when ready to start Arduino calibration...")

        self.arduino.perform_calibration()

        self.training_data["baseline"] = {
            "hardware_calibrated": True,
            "timestamp": get_timestamp(),
            "arduino_port": self.arduino.port
        }

        self.save_all_data()
        print("✓ Arduino baseline calibration completed!")

    def collect_universal_samples(self, item_name: str, num_samples: int, duration: float) -> List[List[float]]:
        """Collect samples using real Arduino sensors"""
        if not self.hardware_connected:
            if not self.connect_hardware():
                print("✗ Cannot collect samples without hardware connection")
                return []

        print(f"\nCollecting {num_samples} samples for '{item_name}' using Arduino...")
        print("Place item sample near sensors and press Enter when ready")
        input("Press Enter to continue...")

        samples = []
        for sample_idx in range(num_samples):
            print(f"Sample {sample_idx + 1}/{num_samples} - Reading from Arduino...")

            time_series = []
            start_time = time.time()
            while time.time() - start_time < duration:
                raw_reading = self.get_sensor_reading()
                if raw_reading:
                    time_series.append(raw_reading)
                time.sleep(0.1)

            if time_series:
                time_series_array = np.array(time_series)
                processed_series = []

                for sensor_idx in range(11):
                    sensor_signal = time_series_array[:, sensor_idx]
                    processed_signal = self.preprocess_signal(sensor_signal, sensor_idx)
                    processed_series.append(processed_signal)

                processed_series = np.array(processed_series).T
                features = self.signal_processor.extract_universal_features(processed_series, mode="adaptive")
                samples.append(features.tolist())

            time.sleep(0.5)

        return samples