# Introduction about dataset

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



