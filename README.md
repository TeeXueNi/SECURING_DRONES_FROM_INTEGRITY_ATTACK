================================================================================
Securing Drones from Integrity Attack
--------------------------------------------------------------------------------
 Final Year Project 2 - Source Code
 Author : Tee Xue Ni  (Student ID: 243UT246WB)
 Faculty of Information Science and Technology, Multimedia University
================================================================================


1. OVERVIEW
--------------------------------------------------------------------------------
This project implements a hybrid GPS-spoofing detection framework for UAVs that
combines:
  - an LSTM motion predictor (trained on normal flight only) that produces
    prediction residuals,
  - an XGBoost classifier that detects attacks from those residual features, and
  - a weighted decision fusion layer.

The pipeline has three stages, each in its own script:
  (1) colab_preprocessing.py  - builds the dataset (merges raw GPS+IMU logs,
                                injects spoofing attacks, produces final_dataset.csv)
  (2) fyp2_fixed_v3.py        - trains and evaluates the hybrid model and the
                                ablation model across multiple seeds
  (3) pathB_analysis.py       - post-hoc early-detection / detection-latency
                                analysis using the saved predictions


2. FILE LIST
--------------------------------------------------------------------------------
  README.txt              - this file
  colab_preprocessing.py  - Stage 1: dataset construction and attack injection
  fyp2_fixed_v3.py        - Stage 2: model training and evaluation (LATEST)
  pathB_analysis.py       - Stage 3: detection-latency / early-window analysis

  (Earlier versions fyp2_fixed.py and fyp2_fixed_v2.py are included for history;
   fyp2_fixed_v3.py is the final version and the one that should be run.)


3. ENVIRONMENT AND TOOLS
--------------------------------------------------------------------------------
The project was developed and run on Google Colaboratory (free tier), which
provides Python and all major libraries pre-installed. It can also be run
locally.

  Tool / Runtime         Version           Download / Reference
  ---------------------  ----------------  -----------------------------------
  Python                 3.10 - 3.12       https://www.python.org/downloads/
  Google Colaboratory    (cloud)           https://colab.research.google.com/
  PX4 SITL (data gen)    v1.14+            https://px4.io/
  Gazebo (data gen)      Garden/Classic    https://gazebosim.org/

  NOTE: PX4 and Gazebo are only needed if you want to REGENERATE the raw flight
  logs from scratch. To reproduce the machine-learning results, you only need
  Python and the libraries below, plus the dataset (see Section 6).


4. PYTHON LIBRARIES
--------------------------------------------------------------------------------
The following libraries are required. On Google Colab, all of these are already
installed except (optionally) a matching XGBoost version.

  Library         Tested Version   Install command
  --------------  ---------------  ------------------------------
  numpy           1.26.x           pip install numpy
  pandas          2.2.x            pip install pandas
  scikit-learn    1.5.x            pip install scikit-learn
  tensorflow      2.16.x           pip install tensorflow
  xgboost         2.0.x            pip install xgboost
  matplotlib      3.9.x            pip install matplotlib
  joblib          1.4.x            pip install joblib
  scipy           1.13.x           pip install scipy   (Stage 3 only)

Quick install (all at once):

  pip install numpy pandas scikit-learn tensorflow xgboost matplotlib joblib scipy


5. HOW TO RUN
--------------------------------------------------------------------------------
STEP 1 - Get the dataset
  Download final_dataset.csv (and/or the raw CSV logs) from the Kaggle link in
  Section 6. Place final_dataset.csv in your working directory.

STEP 2 - (Optional) Regenerate the dataset from raw logs
  Only needed if you want to rebuild final_dataset.csv from the raw GPS/IMU logs.
    1. Open colab_preprocessing.py.
    2. Set RAW_DIR to the folder containing the raw CSV files
       (normal_hover.csv, normal_hover_imu.csv, etc.).
    3. Set OUT_DIR to your desired output folder.
    4. Run the script. It produces:
         - normal_master.csv
         - final_dataset.csv
         - sanity_check.png

STEP 3 - Train and evaluate the models
    1. Open fyp2_fixed_v3.py.
    2. Set CONFIG["csv_path"] to the full path of final_dataset.csv.
    3. Run the script:
         python fyp2_fixed_v3.py
       (In Colab: %run fyp2_fixed_v3.py)
    4. Outputs produced:
         - results_summary.csv        (aggregated metrics across seeds)
         - results_full.json          (per-seed metrics)
         - predictions_seed{N}.json   (saved predictions per seed)
         - roc_curves.png
         - hybrid_predictor_seed{N}.keras, hybrid_xgb_seed{N}.pkl, etc.

STEP 4 - Detection-latency / early-window analysis
    1. Open pathB_analysis.py.
    2. Set PRED_DIR to the folder containing predictions_seed*.json from Step 2.
    3. Run the script:
         python pathB_analysis.py
    4. Outputs produced:
         - detection_latency_summary.csv
         - early_window_summary.csv
         - B1_detection_latency.png
         - B3_early_window_f1.png


6. DATASET DOWNLOAD (SELF-COLLECTED)
--------------------------------------------------------------------------------
The self-collected dataset (generated in PX4 SITL + Gazebo) is hosted on Kaggle:

    Kaggle dataset link:  [PASTE YOUR KAGGLE DATASET URL HERE]

The Kaggle dataset contains:
    - final_dataset.csv        (the processed, ready-to-train dataset)
    - raw/ folder              (the original per-trajectory GPS + IMU CSV logs)

Feature columns in final_dataset.csv:
    lat, lon, alt, vel_n, vel_e, vel_d      (GPS-derived)
    dvel_x, dvel_y, dvel_z, dang_x, dang_y, dang_z   (IMU-derived)
    label        (0 = normal, 1 = spoofed)
    flight_id, trajectory   (organisational columns, not model features)


================================================================================
