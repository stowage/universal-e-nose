# Training Data Directory

This directory stores all universal training data and calibration information.

## File Structure

- `e_nose_data_universal.json` - Main training data file with item metadata
- `baseline_calibration.json` - Baseline calibration data
- `training_logs/` - Training session logs (optional)

## Important Notes

- **Item metadata**: Each sample includes category (perfume, fruit, beverage, etc.)
- **Do not manually edit** the JSON files as they contain structured data
- **Backup regularly** as this contains all your trained item profiles
- **Version control**: Consider adding these files to version control for reproducible results