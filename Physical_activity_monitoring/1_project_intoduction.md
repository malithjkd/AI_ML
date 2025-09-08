# Physical Activity Monitoring ML Project Plan

## 1. Project Goal

The primary objective of this project is to develop and evaluate multiple machine learning models to accurately identify a person's physical activity based on data from wearable sensors. The project will follow a structured approach, from data preparation and feature engineering to model training and performance analysis, using the PAMAP2 dataset.

## 2. Project Stages

The project is divided into the following key stages:

### Stage 1: Data Acquisition and Exploration
- **Objective:** Load the dataset and perform an initial exploratory data analysis (EDA) to understand its structure and characteristics.
- **Tasks:**
    - Load the `.dat` files for all subjects.
    - Verify the data format against the column descriptions in the dataset documentation.
    - Handle missing values (`NaNs`) using an appropriate strategy (e.g., interpolation for time-series data or removal).
    - Perform EDA:
        - Plot sensor data (e.g., acceleration, heart rate) over time for different activities. `- Done`
        - Analyze the distribution of data points across the 18 activities. `- Done`
        - Check for and handle any data inconsistencies. `- Done`

### Stage 2: Data Preprocessing
- **Objective:** Prepare the raw sensor data for feature extraction by cleaning and segmenting it.
- **Tasks:**
    - Filter out all data points labeled with `activityID = 0` (transient activities).
    - Synchronize sensor data streams based on timestamps.
    - Segment the continuous data into fixed-size windows. - **current stage** - Chunking in to one second of data chunks 
    - Based on the reference paper, we will use a **sliding window of 5.12 seconds with a 1-second shift**. - next stage

### Stage 3: Feature Engineering
- **Objective:** Extract meaningful features from the segmented data windows to serve as input for the machine learning models.
- **Tasks:**
    - **IMU Sensor Data (Acceleration, Gyroscope, Magnetometer):**
        - **Time-Domain Features:** Calculate `mean`, `variance`, `standard deviation`, `min`, `max`, and `energy` for each window.
        - **Frequency-Domain Features:** Apply a Fast Fourier Transform (FFT) and calculate features like `spectral energy` and dominant frequencies.
    - **Heart Rate Data:**
        - Calculate `mean` and `gradient` (rate of change) over each window.
    - Combine all extracted features into a single feature vector for each window.

### Stage 4: Model Training and Evaluation
- **Objective:** Train various classification models and evaluate their performance on different activity recognition tasks.
- **Tasks:**
    - **Define Classification Problems:**
        1.  **Intensity Estimation:** 3 classes (light, moderate, vigorous).
        2.  **Basic Activity Recognition:** 5 classes (lie, sit/stand, walk, run, cycle).
        3.  **All Protocol Activities:** 12 classes.
    - **Train Baseline Models:**
        - Decision Tree (C4.5)
        - Naïve Bayes
        - k-Nearest Neighbors (kNN)
    - **Train Advanced Models:**
        - Ensemble Methods (Random Forest, Bagging, Boosting)
        - Deep Learning (e.g., LSTM for time-series classification).
    - **Evaluation Strategy:**
        - **Subject-Dependent:** Use a standard 9-fold cross-validation where data from all subjects is mixed.
        - **Subject-Independent:** Use leave-one-subject-out cross-validation to assess model generalization to new, unseen users.
    - **Performance Metrics:** Evaluate models using `accuracy`, `F1-score`, and `confusion matrices`.

### Stage 5: Results Analysis and Reporting
- **Objective:** Analyze and compare the performance of the different models and document the findings.
- **Tasks:**
    - Compare model performance across the different classification problems.
    - Analyze which features were most important for classification.
    - Document the complete workflow, results, and conclusions.
    - Provide recommendations for future improvements or deployment.


