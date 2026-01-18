# Universal Odor Recognition E-Nose System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Arduino](https://img.shields.io/badge/Arduino-Compatible-green)
![License](https://img.shields.io/badge/License-MIT-orange)

A universal electronic nose system that can recognize **perfumes**, **fruits**, **beverages**, **spices**, and any other odor type using your specific sensor array:
- **SEN0515** (ENS160) - TVOC/eCO2/AQI
- **SEN0536** (SCD41) - CO₂  
- **SEN0569** (MEMS Ethanol) - Alcohol
- **SEN0567** (MEMS NH₃) - Ammonia
- **SEN0574** (MEMS NO₂) - Nitrogen dioxide
- **SEN0466** (Electrochemical CO) - Carbon monoxide
- **SEN0572** (MEMS H₂) - Hydrogen
- **SEN0565** (MEMS CH₄) - Methane
- **SEN0570** (MEMS Smoke) - Smoke detection
- **MQ-135** - Air quality
- **MQ-3** - Alcohol/benzene

## 🌟 Universal Recognition Capabilities

This system can recognize **any trained item** including:
- **Perfumes**: Chanel No. 5, Dior J'adore, YSL Black Opium
- **Fruits**: Apple, Orange, Banana, Strawberry
- **Beverages**: Coffee, Tea, Wine
- **Spices**: Vanilla, Cinnamon, Clove
- **Flowers**: Rose, Lavender, Jasmine
- **Chemicals**: Alcohol, Acetone, Ammonia

## 🚀 Key Features

- **Hardware-Optimized**: Specifically configured for your 11-sensor array
- **I2C + Analog Support**: Handles both digital (I2C) and analog sensors
- **Universal Recognition**: Automatically detects odor categories
- **Specific Identification**: Recognizes trained items by name with confidence
- **Professional-Grade Sensors**: Uses your high-quality MEMS and electrochemical sensors
- **Real-time Processing**: 10 Hz sampling with advanced signal processing
- **Interactive Console**: Easy training and recognition workflow

## 🛠️ Hardware Requirements

### **Required Sensors:**
- SEN0515 (ENS160) - TVOC/eCO2/AQI
- SEN0536 (SCD41) - CO₂
- SEN0569 (MEMS Ethanol) - Alcohol
- SEN0567 (MEMS NH₃) - Ammonia  
- SEN0574 (MEMS NO₂) - Nitrogen dioxide
- SEN0466 (Electrochemical CO) - Carbon monoxide
- SEN0572 (MEMS H₂) - Hydrogen
- SEN0565 (MEMS CH₄) - Methane
- SEN0570 (MEMS Smoke) - Smoke detection
- MQ-135 - Air quality sensor
- MQ-3 - Alcohol/benzene sensor

### **Additional Hardware:**
- Arduino Uno/Mega (or compatible)
- Breadboard and jumper wires
- I2C pull-up resistors (if not built into sensors)
- USB cable for Arduino connection

## 📦 Software Requirements

### **Arduino Libraries (install via Library Manager):**
- Adafruit SCD4x Library
- SparkFun ENS160 Arduino Library

### **Python Dependencies:**
```bash
pip install -r requirements.txt