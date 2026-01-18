"""
Unit tests for the universal e-nose system
"""

import unittest
import numpy as np
from src.universal_e_nose_system import UniversalENoseSystem


class TestUniversalENoseSystem(unittest.TestCase):

    def setUp(self):
        self.e_nose = UniversalENoseSystem()

    def test_universal_feature_extraction(self):
        """Test universal feature extraction functionality"""
        sensor_data = np.random.rand(80, 11)

        features = self.e_nose.signal_processor.extract_universal_features(sensor_data, mode="adaptive")

        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_signal_type_detection(self):
        """Test automatic signal type detection"""
        from src.signal_processing import UniversalSignalProcessor

        perfume_signal = np.zeros((50, 11))
        for i in range(11):
            perfume_signal[:, i] = np.linspace(0.1, 0.8, 50) + np.random.normal(0, 0.05, 50)

        signal_type = UniversalSignalProcessor.detect_signal_type(perfume_signal)
        self.assertIn(signal_type, ["perfume", "unknown"])

    def test_simulation(self):
        """Test sensor simulation functionality"""
        reading = self.e_nose.simulate_sensor_reading("Chanel No 5")

        self.assertEqual(len(reading), 11)
        self.assertTrue(all(0 <= x <= 1.2 for x in reading))

    def test_category_detection(self):
        """Test category detection functionality"""
        self.e_nose.training_data["samples"] = [
            {"features": [1.0] * 50, "odor": "Chanel No 5", "category": "perfume"},
            {"features": [2.0] * 50, "odor": "Apple", "category": "fruit"},
            {"features": [3.0] * 50, "odor": "Coffee", "category": "beverage"}
        ]

        self.e_nose.training_data["odors"] = {
            "Chanel No 5": {"sample_count": 1, "category": "perfume"},
            "Apple": {"sample_count": 1, "category": "fruit"},
            "Coffee": {"sample_count": 1, "category": "beverage"}
        }

        try:
            X, y_items, y_categories = self.e_nose.prepare_training_data_universal()
            self.assertEqual(len(y_categories), 3)
            self.assertIn("perfume", y_categories)
        except Exception as e:
            self.fail(f"Category detection failed: {e}")


if __name__ == '__main__':
    unittest.main()