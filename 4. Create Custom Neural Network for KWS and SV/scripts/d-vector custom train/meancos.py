import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 22
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

print(tf.__version__)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / (norm_product + 1e-8)

def compute_mean_d_vector(d_vectors):
    """Compute the mean d-vector from enrollment samples"""
    return np.mean(d_vectors, axis=0)

def predictDVector(mean_d_vector, authlabel, input_data, input_labels, threshold, verbose=True):
    """Evaluate performance using mean cosine similarity"""
    input_vectors = dv_model.predict(input_data)
    total = len(input_vectors)
    
    is_target = (input_labels == authlabel)
    total_auth = np.sum(is_target)
    total_denied = total - total_auth
    
    similarities = np.array([cosine_similarity(vec, mean_d_vector) for vec in input_vectors])
    predictions = similarities > threshold
    
    true_pos = np.sum(predictions & is_target)
    true_neg = np.sum(~predictions & ~is_target)
    false_pos = np.sum(predictions & ~is_target)
    false_neg = np.sum(~predictions & is_target)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / total_auth if total_auth > 0 else 0
        f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        acc = (true_pos + true_neg) / total
        fpr = false_pos / total_denied if total_denied > 0 else 0
    
    print('-----------------------')
    print(" --- Testing Results ---")
    print(f"Target samples: {total_auth}/{total}")
    print(f"Non-target samples: {total_denied}/{total}")
    print(f"True Positive Rate: {true_pos}/{total_auth} ({true_pos/total_auth*100:.1f}%)")
    print(f"False Positive Rate: {false_pos}/{total_denied} (N/A)" if total_denied == 0 else
          f"False Positive Rate: {false_pos}/{total_denied} ({false_pos/total_denied*100:.1f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    return acc, f1

def provide_predictions(mean_d_vector, input_data):
    """Generate similarity predictions using mean d-vector"""
    input_vectors = dv_model.predict(input_data)
    return np.array([cosine_similarity(vec, mean_d_vector) for vec in input_vectors])

def evaluate_model(auth_class, train_size):
    """Complete evaluation pipeline for mean cosine approach"""
    print(f"\nEvaluating speaker {auth_class} with {train_size} enrollment samples (Mean Cosine)")
    
    try:
        train_dir = f"dataset/user_0_organized/npz_features/train_{train_size}_{auth_class}_features.npz"
        training_npz = np.load(train_dir)
        x_train = training_npz['features']
        
        val_dir = "dataset/user_0_organized/npz_features/validation_features.npz"
        validation_npz = np.load(val_dir)
        x_val, y_val = validation_npz['features'], validation_npz['labels']
        
        testing_dir = "dataset/user_0_organized/npz_features/testing_features.npz"
        testing_npz = np.load(testing_dir)
        x_test, y_test = testing_npz['features'], testing_npz['labels']

        print("=== Dataset Summary ===")
        print(f"Training: {len(x_train)} samples")
        print(f"Validation: {len(x_val)} samples - Classes: {np.unique(y_val, return_counts=True)}")
        print(f"Testing: {len(x_test)} samples - Classes: {np.unique(y_test, return_counts=True)}")

        d_vectors = dv_model.predict(x_train.reshape(-1, 40, 40, 1))
        mean_d_vector = compute_mean_d_vector(d_vectors)
        print(f"Mean D-Vector computed using {len(d_vectors)} samples")

        y_pred_prob = provide_predictions(mean_d_vector, x_val.reshape(-1, 40, 40, 1))
        y_val_bin = (y_val == auth_class).astype(int)
        
        if len(np.unique(y_val_bin)) > 1:
            fpr, tpr, thresholds = roc_curve(y_val_bin, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            fnr = 1 - tpr
            try:
                eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
                abs_diffs = np.abs(fpr - fnr)
                min_index = np.argmin(abs_diffs)
                EER = np.mean((fpr[min_index], fnr[min_index]))
            except:
                eer_threshold = 0.5
                EER = 1.0
                print("Warning: EER calculation failed - using default threshold")
        else:
            print("Warning: Only one class present in validation data")
            eer_threshold = 0.5
            EER = 1.0
            roc_auc = 0.5
        
        acc, f1score = predictDVector(mean_d_vector, auth_class,
                                    x_test.reshape(-1, 40, 40, 1), y_test,
                                    threshold=eer_threshold, verbose=False)
        
        with open("test-results-td-meancos.txt", "a") as f:
            f.write(f"Speaker {auth_class} | Train Size: {train_size}\n")
            f.write(f"Accuracy: {acc:.4f} | F1: {f1score:.4f} ")
            f.write(f"| EER: {EER:.4f} | AUC: {roc_auc:.4f}\n\n")
        
        return acc, f1score, EER, roc_auc
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0, 0, 1, 0

def main():
    auth_class = 0
    train_sizes = [1, 8, 16, 64]
    
    global dv_model
    d_vector_model_name = "d-vector-extractor-256.h5"
    dv_model = tf.keras.models.load_model(d_vector_model_name)
    dv_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
    
    with open("test-results-td-meancos.txt", "w") as f:
        f.write("Speaker Verification Results (Mean Cosine)\n")
        f.write("========================================\n\n")
    
    for size in train_sizes:
        evaluate_model(auth_class, size)

if __name__ == "__main__":
    main()
