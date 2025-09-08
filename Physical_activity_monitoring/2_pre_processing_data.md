# 1. Introduction about dataset

The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities, performed by 9 subjects. Each subject wore 3 inertial measurement units (IMUs) and a heart rate monitor. This dataset is suitable for activity recognition and intensity estimation, providing a basis for developing and applying algorithms for data processing, segmentation, feature extraction, and classification.

## Sensors

-   **Three `Colibri wireless inertial measurement units` (IMU):**
    -   **Sampling Frequency:** 100Hz
    -   **Sensor Placement:**
        -   1 IMU on the wrist of the dominant arm
        -   1 IMU on the chest
        -   1 IMU on the ankle of the dominant side
-   **HR-Monitor:**
    -   **Sampling Frequency:** ~9Hz

## Data Collection Protocol

Each subject followed a protocol involving 12 different activities, with recordings stored in the `Protocol` folder. Some subjects also performed optional activities, which are available in the `Optional` folder. The activities were performed inside a lab environment.

## PAMAP2 Dataset Folder Structure

```
PAMAP2_Dataset/
├── DataCollectionProtocol.pdf
├── DescriptionOfActivities.pdf
├── folder_structure.md
├── PerformedActivitiesSummary.pdf
├── readme.pdf
├── Reiss2012b.pdf
├── subjectInformation.pdf
├── Optional/
│   ├── subject101.dat
│   ├── subject105.dat
│   ├── subject106.dat
│   ├── subject108.dat
│   └── subject109.dat
├── Protocol/
│   ├── subject101.dat
│   ├── subject102.dat
│   ├── subject103.dat
│   ├── subject104.dat
│   ├── subject105.dat
│   ├── subject106.dat
│   ├── subject107.dat
│   ├── subject108.dat
│   └── subject109.dat
```

This structure lists all files and folders included in the PAMAP2_Dataset directory.

## Data Files

Raw sensory data can be found in space-separated text-files (.dat), with one data file per subject per session (protocol or optional). Missing values are indicated with `NaN`. Each line in the data files corresponds to a timestamped and labeled instance of sensory data, containing 54 columns.

## Data Format

Synchronized and labeled raw data from all sensors are merged into one `.dat` text file per subject per session.

### File Columns

Each data file contains 54 columns, structured as follows:

| Columns | Description              |
| :------ | :----------------------- |
| 1       | Timestamp (s)            |
| 2       | Activity ID              |
| 3       | Heart Rate (bpm)         |
| 4-20    | IMU Hand Data            |
| 21-37   | IMU Chest Data           |
| 38-54   | IMU Ankle Data           |

### IMU Data Columns

Each IMU data block consists of 17 columns:

| Column | Description                                         |
| :----- | :-------------------------------------------------- |
| 1      | Temperature (°C)                                    |
| 2-4    | 3D Acceleration (ms⁻²), scale: ±16g, 13-bit         |
| 5-7    | 3D Acceleration (ms⁻²), scale: ±6g, 13-bit          |
| 8-10   | 3D Gyroscope (rad/s)                                |
| 11-13  | 3D Magnetometer (μT)                                |
| 14-17  | Orientation (invalid in this collection)            |

**Note:** Missing sensor readings are marked as `NaN`.

## Activity ID Mapping

The `activityID` column maps to the following activities:

| ID | Activity            |
| :- | :------------------ |
| 1  | Lying               |
| 2  | Sitting             |
| 3  | Standing            |
| 4  | Walking             |
| 5  | Running             |
| 6  | Cycling             |
| 7  | Nordic walking      |
| 9  | Watching TV         |
| 10 | Computer work       |
| 11 | Car driving         |
| 12 | Ascending stairs    |
| 13 | Descending stairs   |
| 16 | Vacuum cleaning     |
| 17 | Ironing             |
| 18 | Folding laundry     |
| 19 | House cleaning      |
| 20 | Playing soccer      |
| 24 | Rope jumping        |
| 0  | Other (transient)   |

**Important:** Data with `activityID=0` should be discarded, as it represents transient activities between main exercises.

## Data Summary & Quality

This is a realistic dataset with some missing data due to two main reasons:
1.  **Wireless Signal Drop:** Occurred rarely. The effective sampling frequencies were 99.63Hz (hand), 99.89Hz (chest), and 99.65Hz (ankle).
2.  **Hardware Issues:** Connection loss or system crashes caused some activities to be partially or completely missing for certain subjects.

Over 10 hours of data were collected, with nearly 8 hours labeled as one of the 18 activities. For a detailed breakdown, see `PerformedActivitiesSummary.pdf`.

## Data Preprocessing and Benchmarking

This document serves as the primary source for understanding the data preprocessing requirements.

Standard methods are used for creating the benchmark:
1.  **Synchronization:** Raw sensor data is synchronized.
2.  **Segmentation:** Data is segmented using a sliding window of 5.12 seconds with a 50% overlap.


---

---


# 2. Creation of Datasets and Dataset History

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

3.  **Removing Static Values:**
    After loading and cleaning the data, we identified columns where all values were zero. These static columns do not provide any useful information for activity recognition and were removed from the dataset. This step reduces the dimensionality and ensures that only relevant sensor data is retained for further processing.

4. **Chunking**:
    The cleaned data files are large and contain multiple activities performed by each subject. To facilitate model training and evaluation, we segmented the data into fixed-size chunks. Specifically, we created 1-second data chunks for selected activities (e.g., standing, running, cycling, nordic walking). Each chunk contains consecutive rows with consistent time intervals, ensuring temporal continuity. Chunks with missing or non-consecutive timestamps were skipped to maintain data quality. The resulting chunks were saved as separate CSV files in a dedicated folder for easy access during feature engineering and model training.

5.  **Data Conversion and Storage:**
    *   After cleaning, the pandas DataFrame was converted into a NumPy array for efficient numerical computation and storage.
    *   The processed data for each subject was saved from its original `.dat` format to a new `.csv` file.
    *   To ensure consistent formatting and prevent data from being saved in scientific notation, the `np.savetxt` function was used with a format specifier (`fmt='%.6f'`), saving all floating-point numbers with 6 decimal places.
    *   These new, cleaned `.csv` files are stored in the `protocol_data_v1/` directory.

