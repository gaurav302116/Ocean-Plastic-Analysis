import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
X=np.genfromtxt('train_X_knn.csv',delimiter=',',dtype=np.float64,skip_header=1)
Y=np.genfromtxt('train_Y_knn.csv',delimiter=',',dtype=np.float64,skip_header=1)
Y=np.insert(Y,0,1)

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def compute_ln_norm_distance(vector1, vector2, n):
    a=0
    for i in range(len(vector1)):
        a+=abs((vector2[i]-vector1[i])**n)
    return a**(1/n)

def find_k_nearest_neighbour(train_X,test_example,n,k):
    dist_indices_pair=[]
    index=0
    for i in train_X:
        dist=compute_ln_norm_distance(i,test_example,n)
        dist_indices_pair.append([index,dist])
        index+=1
    dist_indices_pair.sort(key=lambda x:x[1])
    top_k_pairs=dist_indices_pair[:k]
    top_k_index=[i[0] for i in top_k_pairs]
    return top_k_index


def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    test_Y=[]
    for i in test_X:
        top_k_index=find_k_nearest_neighbour(X,i,1,1)
        l1=[]
        for i in top_k_index:
            l1.append(Y[i])
        Y_values = list(set(l1))
        max_count =0
        most_frequent_label = -1
        for y in Y_values:
            count = l1.count(y)
            if(count > max_count):
                  max_count = count
                  most_frequent_label = y
        test_Y.append(int(most_frequent_label))
    return test_Y
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    pred_Y=np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 