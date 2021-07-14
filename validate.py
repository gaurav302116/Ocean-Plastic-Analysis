from os import path
import numpy as np
import csv


def check_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '" + predicted_test_Y_file_path +"' file")


def check_format(test_X_file_path, predicted_test_Y_file_path):
    pred_Y = []
    with open(predicted_test_Y_file_path, 'r') as file:
        reader = csv.reader(file)
        pred_Y = list(reader)
    pred_Y = np.array(pred_Y)

    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")


def check_weighted_f1_score(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=',', dtype=np.int)
    actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=',', dtype=np.int)
    from sklearn.metrics import f1_score
    weighted_f1_score = f1_score(actual_Y, pred_Y, average = 'weighted')
    print("Weighted F1 score", weighted_f1_score)
    return weighted_f1_score


def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_knn.csv"
    
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    check_weighted_f1_score(actual_test_Y_file_path, predicted_test_Y_file_path)