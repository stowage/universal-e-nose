"""
Arduino Interface Module
Handles communication with Arduino e-nose hardware with I2C support
"""

import serial
import json
import time
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ArduinoENoseInterface:
    """
    Interface to communicate with Arduino e-nose hardware
    """

    def __init__(self, port: str = 'COM3', baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False

    def connect(self) -> bool:
        """Establish connection with Arduino"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)
            self.serial_connection.flushInput()

            response = self.send_command("status")
            if response and "Calibration Status" in response:
                logger.info(f"Connected to Arduino on {self.port}")
                self.is_connected = True
                return True
            else:
                logger.error("No valid response from Arduino")
                return False

        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            logger.info("Arduino connection closed")

    def send_command(self, command: str, timeout: float = None) -> str:
        """Send command to Arduino and return response"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Arduino")

        try:
            if timeout:
                original_timeout = self.serial_connection.timeout
                self.serial_connection.timeout = timeout

            self.serial_connection.write(f"{command}\n".encode())
            response = self.serial_connection.readline().decode('utf-8').strip()

            if timeout:
                self.serial_connection.timeout = original_timeout

            return response
        except Exception as e:
            logger.error(f"Command error: {e}")
            return ""

    def read_single_sample(self) -> Optional[List[float]]:
        """Read a single sensor sample from Arduino"""
        if not self.is_connected:
            return None

        try:
            response = self.send_command("read", timeout=1.0)
            if response:
                try:
                    data = json.loads(response)
                    if "sensors" in data and len(data["sensors"]) == 11:
                        return data["sensors"]
                    else:
                        logger.warning(f"Invalid sensor data format: {response}")
                        return None
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response: {response}")
                    return None
            return None

        except Exception as e:
            logger.error(f"Reading error: {e}")
            return None

    def collect_extended_time_series(self, duration: float = 10.0) -> List[List[float]]:
        """Collect extended time series for universal analysis"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Arduino")

        logger.info(f"Starting extended time series collection for {duration} seconds...")
        self.send_command("sample")

        time_series = []
        start_time = time.time()

        while time.time() - start_time < duration:
            try:
                response = self.serial_connection.readline().decode('utf-8').strip()
                if response:
                    try:
                        data = json.loads(response)
                        if "sensors" in data:
                            time_series.append(data["sensors"])
                    except json.JSONDecodeError:
                        continue
            except Exception:
                break
            time.sleep(0.05)

        logger.info(f"Collected {len(time_series)} samples for universal analysis")
        return time_series

    def perform_calibration(self):
        """Perform baseline calibration on Arduino"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Arduino")

        logger.info("Starting Arduino calibration...")
        self.send_command("calibrate")
        time.sleep(10)

        status = self.send_command("status")
        logger.info(f"Calibration status: {status}")

    def get_calibration_status(self) -> str:
        """Get current calibration status"""
        return self.send_command("status")