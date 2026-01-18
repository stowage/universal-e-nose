"""
Signal Processing Module
Advanced signal processing functions for universal odor recognition
"""

import numpy as np
from scipy import signal
from scipy.sparse import diags
from scipy.stats import entropy
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Advanced signal processing for e-nose data"""

    @staticmethod
    def als_baseline_correction(y: np.ndarray, lam: float = 1e6, p: float = 0.01) -> np.ndarray:
        """Asymmetric Least Squares baseline correction"""
        L = len(y)
        if L < 3:
            return y

        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(10):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            try:
                z = np.linalg.solve(Z.toarray(), w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            except np.linalg.LinAlgError:
                logger.warning("Baseline correction failed, returning original signal")
                return y
        return z

    @staticmethod
    def preprocess_signal(raw_signal: np.ndarray, sensor_id: int = None) -> np.ndarray:
        """Advanced signal preprocessing with adaptive filtering"""
        if len(raw_signal) < 3:
            return raw_signal

        try:
            baseline = SignalProcessor.als_baseline_correction(raw_signal)
            corrected = raw_signal - baseline

            if len(corrected) >= 7:
                window_length = min(7, len(corrected) // 2 * 2 + 1)
                if window_length >= 5:
                    smoothed = signal.savgol_filter(corrected, window_length=window_length, polyorder=2)
                else:
                    smoothed = corrected
            else:
                smoothed = corrected

            mad = np.median(np.abs(smoothed - np.median(smoothed)))
            if mad > 0:
                threshold = np.median(smoothed) + 3 * mad
                smoothed = np.clip(smoothed, None, threshold)

            return smoothed

        except Exception as e:
            logger.warning(f"Signal preprocessing failed for sensor {sensor_id}: {e}")
            return raw_signal


class UniversalSignalProcessor(SignalProcessor):
    """Universal signal processor for all odor types"""

    @staticmethod
    def extract_universal_features(time_series: np.ndarray, mode: str = "adaptive") -> np.ndarray:
        """
        Extract features suitable for any odor type
        Automatically adapts based on signal characteristics
        """
        n_timepoints, n_sensors = time_series.shape

        universal_features = []

        for sensor_idx in range(n_sensors):
            sensor_signal = time_series[:, sensor_idx]
            universal_features.extend([
                np.mean(sensor_signal),
                np.std(sensor_signal),
                np.max(sensor_signal),
                np.min(sensor_signal),
                np.percentile(sensor_signal, 75) - np.percentile(sensor_signal, 25),
                entropy(np.histogram(sensor_signal, bins=10)[0] + 1e-10),
                np.mean(np.abs(np.diff(sensor_signal))) if len(sensor_signal) > 1 else 0,
                np.max(np.abs(np.diff(sensor_signal))) if len(sensor_signal) > 1 else 0,
            ])

        if mode == "adaptive":
            signal_type = UniversalSignalProcessor.detect_signal_type(time_series)

            if signal_type == "perfume":
                perfume_features = UniversalSignalProcessor.extract_perfume_features(time_series)
                return np.concatenate([universal_features, perfume_features])
            elif signal_type == "general":
                steady_features = UniversalSignalProcessor.extract_steady_state_features(time_series)
                return np.concatenate([universal_features, steady_features])
            else:
                perfume_features = UniversalSignalProcessor.extract_perfume_features(time_series)
                steady_features = UniversalSignalProcessor.extract_steady_state_features(time_series)
                return np.concatenate([universal_features, perfume_features, steady_features])

        elif mode == "perfume":
            perfume_features = UniversalSignalProcessor.extract_perfume_features(time_series)
            return np.concatenate([universal_features, perfume_features])
        else:
            steady_features = UniversalSignalProcessor.extract_steady_state_features(time_series)
            return np.concatenate([universal_features, steady_features])

    @staticmethod
    def detect_signal_type(time_series: np.ndarray) -> str:
        """Automatically detect if signal is from perfume, general odor, or unknown"""
        n_timepoints, n_sensors = time_series.shape

        if n_timepoints < 10:
            return "general"

        stability_scores = []
        temporal_dynamics = []

        for sensor_idx in range(n_sensors):
            signal = time_series[:, sensor_idx]

            if len(signal) >= 20:
                final_stability = np.std(signal[-10:])
                initial_stability = np.std(signal[:10])
                stability_ratio = final_stability / (initial_stability + 1e-6)
                stability_scores.append(stability_ratio)

            if len(signal) > 1:
                derivatives = np.abs(np.diff(signal))
                max_derivative = np.max(derivatives)
                mean_derivative = np.mean(derivatives)
                temporal_dynamics.append(max_derivative / (mean_derivative + 1e-6))

        avg_stability = np.mean(stability_scores) if stability_scores else 1.0
        avg_dynamics = np.mean(temporal_dynamics) if temporal_dynamics else 1.0

        if avg_stability < 0.5 and avg_dynamics > 2.0:
            return "perfume"
        elif avg_stability > 0.8 and avg_dynamics < 1.5:
            return "general"
        else:
            return "unknown"

    @staticmethod
    def extract_perfume_features(time_series: np.ndarray) -> np.ndarray:
        """Specialized feature extraction for perfumes"""
        n_timepoints, n_sensors = time_series.shape

        if n_timepoints < 20:
            return np.array([])

        features = []
        top_end = min(n_timepoints // 3, 20)
        middle_end = min(2 * n_timepoints // 3, top_end + 30)

        top_phase = time_series[:top_end, :]
        middle_phase = time_series[top_end:middle_end, :]
        base_phase = time_series[middle_end:, :]

        for phase_name, phase_data in [("top", top_phase), ("middle", middle_phase), ("base", base_phase)]:
            if phase_data.size > 0:
                phase_mean = np.mean(phase_data, axis=0)
                phase_std = np.std(phase_data, axis=0)
                phase_max = np.max(phase_data, axis=0)
                phase_min = np.min(phase_data, axis=0)

                features.extend([
                    *phase_mean, *phase_std, *phase_max, *phase_min,
                ])

                if phase_name == "top" and middle_phase.size > 0:
                    top_to_middle_diff = np.mean(middle_phase[:5], axis=0) - np.mean(top_phase[-5:], axis=0)
                    features.extend(top_to_middle_diff)

                if phase_name == "middle" and base_phase.size > 0:
                    middle_to_base_diff = np.mean(base_phase[:5], axis=0) - np.mean(middle_phase[-5:], axis=0)
                    features.extend(middle_to_base_diff)

        for sensor_idx in range(n_sensors):
            sensor_signal = time_series[:, sensor_idx]
            derivative = np.diff(sensor_signal)
            features.extend([
                np.mean(derivative[:10]),
                np.mean(derivative[-10:]),
                np.max(derivative),
                np.argmax(derivative),
            ])

            if len(sensor_signal) >= 30:
                final_stability = np.std(sensor_signal[-20:])
                features.append(final_stability)

        if top_phase.size > 0 and base_phase.size > 0:
            for i in range(n_sensors):
                if len(top_phase) > 0 and len(base_phase) > 0:
                    top_response = np.mean(top_phase[:, i])
                    base_response = np.mean(base_phase[:, i])
                    if abs(base_response) > 1e-6:
                        ratio = top_response / base_response
                    else:
                        ratio = 0.0
                    features.append(ratio)

        return np.array(features)

    @staticmethod
    def extract_steady_state_features(time_series: np.ndarray) -> np.ndarray:
        """Extract features optimized for steady-state general odors"""
        n_timepoints, n_sensors = time_series.shape
        features = []

        steady_start = max(n_timepoints // 2, n_timepoints - 20)
        steady_state = time_series[steady_start:, :]

        if steady_state.size > 0:
            steady_mean = np.mean(steady_state, axis=0)
            steady_std = np.std(steady_state, axis=0)

            features.extend([
                *steady_mean, *steady_std, *np.max(steady_state, axis=0), *np.min(steady_state, axis=0)
            ])

            for i in range(n_sensors):
                for j in range(i + 1, n_sensors):
                    if abs(steady_mean[j]) > 1e-6:
                        ratio = steady_mean[i] / steady_mean[j]
                    else:
                        ratio = 0.0
                    features.append(ratio)

        return np.array(features)