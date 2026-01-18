"""
Main entry point for the Universal Odor Recognition E-Nose System
"""

import sys
import argparse
from src import UniversalENoseSystem, UniversalHardwareENoseSystem


def main():
    """Main function to run the universal odor recognition system"""
    parser = argparse.ArgumentParser(description='Universal Odor Recognition E-Nose System')
    parser.add_argument('--hardware', action='store_true', help='Use Arduino hardware interface')
    parser.add_argument('--port', type=str, default='COM3', help='Arduino port (default: COM3)')
    parser.add_argument('--config', type=str, default='config/default_config.json', help='Configuration file')

    args = parser.parse_args()

    print("Starting Universal Odor Recognition E-Nose System...")
    print("This system can recognize perfumes, fruits, beverages, spices, and more!")
    print("Configured for your specific 11-sensor array:")
    print("• SEN0515 (ENS160) - TVOC/eCO2/AQI")
    print("• SEN0536 (SCD41) - CO₂")
    print("• SEN0569 (MEMS Ethanol) - Alcohol")
    print("• SEN0567 (MEMS NH₃) - Ammonia")
    print("• SEN0574 (MEMS NO₂) - Nitrogen dioxide")
    print("• SEN0466 (Electrochemical CO) - Carbon monoxide")
    print("• SEN0572 (MEMS H₂) - Hydrogen")
    print("• SEN0565 (MEMS CH₄) - Methane")
    print("• SEN0570 (MEMS Smoke) - Smoke detection")
    print("• MQ-135 - Air quality")
    print("• MQ-3 - Alcohol/benzene")

    if args.hardware:
        try:
            import serial
            e_nose = UniversalHardwareENoseSystem(arduino_port=args.port, config_file=args.config)

            if e_nose.connect_hardware():
                print("✓ Hardware connection established")
            else:
                print("⚠ Hardware connection failed - system will use simulation mode")
                e_nose = UniversalENoseSystem(config_file=args.config)

        except ImportError:
            print("✗ pyserial library not found!")
            print("Install it with: pip install pyserial")
            print("\nRunning simulation mode instead...")
            e_nose = UniversalENoseSystem(config_file=args.config)
    else:
        e_nose = UniversalENoseSystem(config_file=args.config)

    try:
        e_nose.main_menu()
    finally:
        if hasattr(e_nose, 'disconnect_hardware'):
            e_nose.disconnect_hardware()


if __name__ == "__main__":
    main()