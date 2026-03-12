import numpy as np
import time
import sys

BATCH_SIZE = 100  #batch size for k-NN prediction
PCA = True  #whether to apply PCA dimensionality reduction
PCA_DIM = 25  #number of PCA dimensions to reduce to
FLAG = True  #whether to normalize (True) or standardize (False) the data

def load_data():
    try:
        from tensorflow.keras.datasets import cifar10
    except Exception as e:
        raise RuntimeError("Error importing TensorFlow Keras. Please ensure TensorFlow is installed.") from e
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(f"Loaded CIFAR-10 dataset: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples.", flush=True)
    #print({'x_train': x_train.shape, 'y_train': y_train.shape, 'x_test': x_test.shape, 'y_test': y_test.shape}, flush=True)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return x_train, y_train, x_test, y_test

# Preprocess the data: normalize or standardize and reshape
def preprocess_data(x_train, x_test, flag):
    N = x_train.shape[0]
    #N is the number of samples (batch size). 
    #For CIFAR-10 training set, N would be 50000.
    D = np.prod(x_train.shape[1:])
    #D is the product of the remaining dimensions (height × width × channels).
    #For CIFAR-10 images shaped (N, 32, 32, 3), D = 32 × 32 × 3 = 3072.
    x_train = x_train.reshape(N, D)
    x_test = x_test.reshape(x_test.shape[0], D)
    #Reshapes each image to a 1-D vector of length D, producing shape (N, D). 
    #That converts images from (H,W,C) to a flat feature vector per sample.
    if flag:
        print("Normalized data ", flush=True)
        # Converts x to dtype float32 (if not already) then divides by 255.0.
        # normalize pixel values from integer 0–255 to floating-point 0.0–1.0.
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    else:
        print("Standardized data ", flush=True)
        # Converts x to dtype float32 (if not already), then standardizes each feature
        # to have zero mean and unit variance across the dataset.
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        mean = np.mean(x_train, axis=0, keepdims=True) #Compute mean across the dataset for each feature (pixel position)
        std = np.std(x_train, axis=0, keepdims=True) + 1e-7 #Compute std deviation across the dataset for each feature
        x_train = (x_train - mean) / std 
        x_test = (x_test - mean) / std

    return x_train, x_test

def pca_fit_transform(X, k):
    if k <= 0 or k > X.shape[1]:
        raise ValueError("k must be a positive integer not exceeding the feature dimension of the data.")
    X_meaned = X.mean(axis=0)
    X_centered = X - X_meaned
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    Vk = Vt[:k, :].T
    Xk = X_centered.dot(Vk)
    return Xk, Vk, X_meaned

def pca_transform(X, Vk, X_meaned):
    X_centered = X - X_meaned
    Xk = X_centered.dot(Vk)
    return Xk

def compute_centroids(X_train, y_train, num_classes=None):
    if num_classes is None:
        num_classes = int(np.max(y_train)) + 1

    D = X_train.shape[1] #Reads the feature dimension D from X_train's second axis. Assumes X_train is shaped (N, D). 
                         #If X_train is 1-D or has incompatible shape, this will raise an IndexError. D is used to size the centroid vectors.
    centroids = np.zeros((num_classes, D), dtype=X_train.dtype) #an array to hold centroid vectors, one per class.
    counts = np.zeros(num_classes, dtype=np.int32) #Integer array to store how many training samples belong to each class.
    for c in range(num_classes):
        mask = (y_train == c) #Builds a boolean mask of shape (N,) where mask[i] is True iff y_train[i] equals the current class c.
        # This mask is used to select samples of class c from X_train.
        if np.any(mask):
            centroids[c] = X_train[mask].mean(axis=0) #If there are any samples of class c (i.e., mask has any True values),
                                                      #compute the mean of those samples along axis 0 (feature dimension) to get the centroid vector for class c.
            counts[c] = np.sum(mask) #Counts how many samples belong to class c by summing the boolean mask (True counts as 1, False as 0).
        else:
            centroids[c] = 0
            counts[c] = 0
    return centroids, counts

def predict_nearest_centroid(X_test, centroids):
    a2 = np.sum(X_test * X_test, axis=1, keepdims=True)
    b2 = np.sum(centroids * centroids, axis=1)
    ab = X_test.dot(centroids.T)
    dists = a2 + b2 - 2 * ab
    y_pred = np.argmin(dists, axis=1)
    return y_pred

def knn_predict(X_train, y_train, X_test, k, batch_size):
    if k <= 0 or k > X_train.shape[0]:
        raise ValueError("k must be a positive integer not exceeding the number of training samples.")
    
    num_test = X_test.shape[0] #number of test samples
    y_pred = np.zeros(num_test, dtype=y_train.dtype)
    b2 = np.sum(X_train * X_train, axis=1) #squared norms of training samples
    for i in range(0, num_test, batch_size):
        j = min(i + batch_size, num_test) #end index of the batch
        Xt = X_test[i:j] #current batch of test samples ( shape (B,D) where B = j-i is batch size)
        a2 = np.sum(Xt * Xt, axis=1, keepdims=True) #squared norms of test samples of the batch
        ab = Xt.dot(X_train.T)
        dists = a2 + b2 - 2 * ab


        #For each test sample in the batch, find the index of the nearest training sample
        # (minimum distance) and assign its label to y_pred.
        if k == 1:
            y_pred[i:j] = y_train[np.argmin(dists, axis=1)] 
        else:
            #for k > 1, find the k nearest neighbors and perform majority voting 
            knn_indices = np.argpartition(dists, kth=k-1, axis=1)[:, :k] #Get indices of the k smallest distances for each test sample
            knn_distances = np.take_along_axis(dists, knn_indices, axis=1) #Retrieve the distances of the k nearest neighbors
            knn_labels = y_train[knn_indices] #Get the labels of the k nearest neighbors


            #for each test sample in the batch, perform majority voting among the k neighbors
            #Tie-breaking: when two labels tie for count, np.unique returns labels sorted, and if the number of
            # candidates is greater than 1, we choose the closest in terms of distance among the tied labels.
            for n in range(j - i):
                labels, counts = np.unique(knn_labels[n], return_counts=True)
                max_count = np.max(counts)
                candidates = labels[counts == max_count]
                # Tie-breaking: choose the closest in terms of distance
                if candidates.size > 1:
                    closest = np.argmin(knn_distances[n][counts == max_count])
                    y_pred[i + n] = candidates[closest]
                else:
                    y_pred[i + n] = candidates[0]

    return y_pred

# Compute the accuracy of the predictions
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) #Compares true labels with predicted labels element-wise, counts the number of correct predictions,
                                    #and divides by the total number of samples to get the accuracy as a fraction.

def main():
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = preprocess_data(x_train, x_test, FLAG)

    print(f"Run Params:PCA={PCA}, PCA_DIM={PCA_DIM}, BATCH_SIZE={BATCH_SIZE}", flush=True)


    if PCA:
        print(f"\n--- PCA to {PCA_DIM} dimensions ---", flush=True)
        t0 = time.time()
        x_train, Vk, X_meaned = pca_fit_transform(x_train, PCA_DIM)
        x_test = pca_transform(x_test, Vk, X_meaned)
        t1 = time.time()
        print(f"PCA fit and transform time: {t1 - t0:.3f}s", flush=True)

    print("\n--- Nearest Centroid ---", flush=True)
    t0 = time.time()
    centroids, counts = compute_centroids(x_train, y_train)
    t1 = time.time()
    y_pred_centroid = predict_nearest_centroid(x_test, centroids)
    t2 = time.time()
    acc_centroid = compute_accuracy(y_test, y_pred_centroid)
    print(f"Centroid compute time: {t1 - t0:.3f}s, predict time: {t2 - t1:.3f}s, accuracy: {acc_centroid:.4f}")

    k_knn = 1
    print("\n--- 1-NN ---", flush=True)
    t0 = time.time()
    y_pred_knn1 = knn_predict(x_train, y_train, x_test, k_knn, BATCH_SIZE)
    t1 = time.time()
    acc_knn1 = compute_accuracy(y_test, y_pred_knn1)
    print(f"1-NN predict time: {t1 - t0:.3f}s, accuracy: {acc_knn1:.4f}")

    k_knn = 3
    print("\n--- 3-NN ---", flush=True)
    t0 = time.time()
    y_pred_knn3 = knn_predict(x_train, y_train, x_test, k_knn, BATCH_SIZE)
    t1 = time.time()
    acc_knn3 = compute_accuracy(y_test, y_pred_knn3)
    print(f"3-NN predict time: {t1 - t0:.3f}s, accuracy: {acc_knn3:.4f}")

if __name__ == "__main__":
    main()