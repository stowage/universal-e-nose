"""
Unit tests for Arduino interface (mocked)
"""

import unittest
from unittest.mock import Mock, patch
from src.arduino_interface import ArduinoENoseInterface


class TestArduinoENoseInterface(unittest.TestCase):

    @patch('src.arduino_interface.serial.Serial')
    def test_connection(self, mock_serial):
        """Test Arduino connection functionality"""
        mock_serial.return_value.is_open = True
        mock_serial.return_value.readline.return_value = b'Calibration Status: NOT CALIBRATED\n'

        arduino = ArduinoENoseInterface(port='COM3')
        result = arduino.connect()

        self.assertTrue(result)
        self.assertTrue(arduino.is_connected)

    @patch('src.arduino_interface.serial.Serial')
    def test_send_command(self, mock_serial):
        """Test command sending functionality"""
        mock_instance = Mock()
        mock_instance.is_open = True
        mock_instance.readline.return_value = b'{"sensors":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.5]}\n'
        mock_serial.return_value = mock_instance

        arduino = ArduinoENoseInterface(port='COM3')
        arduino.is_connected = True

        response = arduino.send_command("read")

        self.assertIn("sensors", response)


if __name__ == '__main__':
    unittest.main()