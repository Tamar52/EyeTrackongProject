# EyeTrackingProject
# This project is based on MIT "eye tracking for everyone" using keras.
# Before you start, make sure you have downloaded GazeCapture dataset.

# First step to use this project will be runing "prepare_dataset.py", this is needed in order to Prepare the GazeCapture dataset for use with the keras code. 
# This code also Crops images and compiles JSONs into metadata.mat
# This script requiers the following:
# --dataset_path is the path to the downloaded GazeCapture dataset.
# --output_path is the path to th folder you want to save the prepered data


# After prepering the data, go to main and give the the desired flag as an input, the options are as follow:
#   -train to train a model
#   -eval to evaluate an existing model
#   -predict to get prediction based on existing model
# In addition you can choose max_epoch and batc size

# Last step will be analysing the results, this step can be done runing eye_tracking_analysis.py, the desiered analyse must be in main function below.
