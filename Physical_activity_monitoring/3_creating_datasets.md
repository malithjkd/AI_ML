# Creation of Datasets and Dataset History

This document outlines the process of creating the cleaned datasets used in this project from the raw PAMAP2 data.

## 1. Data Pre-processing

The raw data from the PAMAP2 dataset is provided in individual `.dat` files for each subject. These files are space-delimited and do not contain headers. The following steps were performed to clean and structure the data into a usable format.

1.  **Loading and Labeling:**
    *   Each subject's `.dat` file was loaded into a pandas DataFrame.
    *   A predefined list of 54 column names, including timestamps, activity ID, heart rate, and various IMU sensor readings (temperature, acceleration, gyroscope, magnetometer, orientation) for the hand, chest, and ankle, were assigned to the DataFrame.

2.  **Handling Missing Values (`NaN`):**
    A two-part strategy was used to handle missing data:

    *   **Heart Rate Imputation:** The `heart_rate` column had a significant number of missing values, likely due to its lower sampling frequency compared to the IMU sensors. To preserve the data continuity, these missing values were imputed using a forward-fill (`ffill`) followed by a backward-fill (`bfill`). This propagates the last known valid observation forward and then fills any remaining `NaN`s (usually at the beginning of the file) with the next valid observation.

    *   **Row-wise Deletion for Sensor Data:** For all other sensor columns, any row containing one or more `NaN` values was completely removed. This approach ensures that every saved timestamp has a complete and valid set of readings from all IMU sensors, which is critical for feature engineering and model training.

3.  **Data Conversion and Storage:**
    *   After cleaning, the pandas DataFrame was converted into a NumPy array for efficient numerical computation and storage.
    *   The processed data for each subject was saved from its original `.dat` format to a new `.csv` file.
    *   To ensure consistent formatting and prevent data from being saved in scientific notation, the `np.savetxt` function was used with a format specifier (`fmt='%.6f'`), saving all floating-point numbers with 6 decimal places.
    *   These new, cleaned `.csv` files are stored in the `protocol_data_v1/` directory.