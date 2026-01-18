"""
Universal Odor Recognition E-Nose System
A universal electronic nose system for recognizing any trained odor type.
"""

__version__ = "2.1.0"
__author__ = "Universal Odor Recognition E-Nose Project"
__email__ = "universal-e-nose-project@example.com"

from .e_nose_system import AdvancedENoseSystem
from .universal_e_nose_system import UniversalENoseSystem, UniversalHardwareENoseSystem
from .arduino_interface import ArduinoENoseInterface
from .signal_processing import SignalProcessor, UniversalSignalProcessor
