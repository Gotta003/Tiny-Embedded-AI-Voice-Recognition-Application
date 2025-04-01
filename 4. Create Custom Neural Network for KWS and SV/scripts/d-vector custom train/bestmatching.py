import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import sys

import random
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import shutil
tfk = tf.keras
tfkl = tf.keras.layers

print(tf.__version__)

seed = 22

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

dv_model=None

def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)

def compute_similarity(input_vector, d_vectors):
  similarities = []

  for dv in d_vectors:
    similarity = cosine_similarity(input_vector, dv)
    similarities.append(similarity)

  return max(similarities)

def predictDVector(d_vectors,authlabel,input_data, input_labels, threshold, verbose=True):
  input_vectors = dv_model.predict(input_data)
  total = len(input_vectors)
  total_auth = 0
  total_denied = 0

  for i in range(len(input_labels)):
    if(input_labels[i]!=auth_class):
      total_denied = total_denied+1
    else:
      total_auth = total_auth + 1

  correct_auth=0
  correct_denied=0

  for i in range(len(input_vectors)):
    similarity=compute_similarity(input_vectors[i], d_vectors)
    result = " -- ERROR!"
    if(similarity>threshold and input_labels[i] == authlabel):
      correct_auth = correct_auth + 1
      result = ""
    if(similarity<=threshold and input_labels[i] != authlabel):
      correct_denied = correct_denied + 1
      result = ""
    if(verbose):
      print("similarity: " + str(similarity) + " --- Class: " + str(input_labels[i]) + " " + result)
  correct = correct_auth + correct_denied

  print('-----------------------')
  print(" --- Testing Results ---")
  true_positive = correct_auth
  false_positive = total_denied - correct_denied
  false_negative = total_auth - correct_auth
  prec = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)

  print("True Positive Rate: " + str(correct_auth) + "/" + str(total_auth) + " (" + str(correct_auth*100/total_auth) + "%)")
  print("False Positive Rate: " + str(false_positive) + "/" + str(total_denied) + " (" + str((false_positive)*100/total_denied) + "%)")
  print("Precision: " + str(prec))
  print("Recall: " + str(recall))
  print('******************')
  print("Total correct " + str(correct) + "/" + str(total))
  acc = correct/total
  f1score = 2*prec*recall/(prec+recall)
  print("Accuracy on this dataset: " + str(acc))
  print("F1-Score on this dataset: " + str(f1score))

  return acc, f1score

def provide_predictions(d_vectors, input_data):
  y_predictions_prob = np.zeros((len(input_data), 1))
  input_vectors = dv_model.predict(input_data)
  for i in range(len(input_vectors)):
    similarity=compute_similarity(input_vectors[i], d_vectors)
    y_predictions_prob[i] = similarity
  return y_predictions_prob

def evaluate_model(auth_class, train_size):
  print("Testing with speaker id: " + str(auth_class) + " and train size: " + str(train_size))

  train_dir = f"dataset/user_0_organized/npz_features/train_{train_size}_{auth_class}_features.npz"
  training_npz = np.load(train_dir)
  x_train = training_npz['features']

  val_dir = "dataset/user_0_organized/npz_features/validation_features.npz"
  validation_npz = np.load(val_dir)
  x_val, y_val = validation_npz['features'], validation_npz['labels']

  print("Validation class distribution:", np.unique(y_val, return_counts=True))

  testing_dir = "dataset/user_0_organized/npz_features/testing_features.npz"
  testing_npz = np.load(testing_dir)
  x_test, y_test = testing_npz['features'], testing_npz['labels']

  print("=== Dataset Summary ===")
  print(f"Training: {len(x_train)} samples (should include both classes)")
  print(f"Validation: {len(x_val)} samples - Classes: {np.unique(y_val, return_counts=True)}")
  print(f"Testing: {len(x_test)} samples - Classes: {np.unique(y_test, return_counts=True)}")

  d_vectors = dv_model.predict(x_train.reshape(train_size,40,40,1))
  print(d_vectors.shape)

  save_path = f"d_vectors_{auth_class}_{train_size}.npz"
  np.savez(save_path, labels=y_val, d_vectors=d_vectors)
  print(f"Saved d_vectors and labels to {save_path}")

  y_pred_prob = provide_predictions(d_vectors, x_val)

  y_val_bin = np.where(y_val == auth_class, 1, 0)

  for i,classvalue in enumerate(y_val):
    if(classvalue!=auth_class):
      y_val_bin[i] = 0

  if len(np.unique(y_val_bin)) > 1:
    fpr, tpr, thresholds = roc_curve(y_val_bin, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print("-----")

    print("Plotting the Receiving Operating Characteristic curve:")
    '''
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''
    fnr = 1 - tpr
    try:
      eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
      print("-----")
      print("EER Threshold: ", eer_threshold)
      abs_diffs = np.abs(fpr - fnr)
      min_index = np.argmin(abs_diffs)
      EER = np.mean((fpr[min_index], fnr[min_index]))
    except:
      eer_threshold=0.5
      EER=1.0
      print(print("Warning: EER calculation failed - using default threshold"))

  else:
    print("Warning: Only one class present in validation data")
    eer_threshold = 0.5
    EER = 1.0
    roc_auc = 0.5
    print("EER = " + str(EER))
    print("AUC = " + str(roc_auc))

  acc, f1score = predictDVector(d_vectors, auth_class, x_test, y_test, threshold=eer_threshold, verbose=False)

  with open("test-results-td-bestmatch.txt", "a") as f:
      f.write(f"Speaker {auth_class} | Train Size: {train_size}\n")
      f.write(f"Accuracy: {acc:.4f} | F1: {f1score:.4f} | EER: {EER:.4f} | AUC: {roc_auc:.4f}\n")
      f.close

def main():
    auth_class = 0
    train_sizes = [1, 8, 16, 64]
    
    global dv_model

    d_vector_model_name = "d-vector-extractor-256.h5"
    dv_model = tfk.models.load_model(d_vector_model_name)
    dv_model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                    optimizer=tfk.optimizers.Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
    dv_model.summary()
    with open("test-results-td-bestmatch.txt", "w") as f:
        f.write("Speaker Verification Results\n")
        f.write("==========================\n\n")
    
    for size in train_sizes:
        evaluate_model(auth_class, size)

if __name__ == "__main__":
    !rm -r /content/d_vectors/
    main()
    !mkdir /content/d_vectors/
    !mv d_vectors* /content/d_vectors/
