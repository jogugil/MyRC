from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from tslearn.metrics import dtw
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, v_measure_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import numpy as np
from scipy.spatial.distance import pdist, squareform

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import pywt

import mne
from mne import io
from mne.datasets import refmeg_noise
from mne.viz import plot_alignment
from mne.preprocessing import ICA
from mne.preprocessing import regress_artifact

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from nilearn import plotting
from scipy.signal import resample
from scipy.signal import welch
from sklearn.metrics import homogeneity_score
def evaluate_tsne (representations, labels):
    '''
        * Silhouette Score: Evalúa cómo de similar es un punto a su propio cluster en comparación con otros clusters. Un valor cercano a 1 indica que los puntos están bien agrupados.
        * Homogeneity Score: Mide si todos los clusters contienen solo puntos que pertenecen a una sola clase.
        * Completeness Score: Mide si todos los puntos de una clase están asignados al mismo cluster.
        * V-Measure Score: Es la media armónica de la homogeneidad y la completitud, proporcionando un único valor para evaluar ambos aspectos.
    '''
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(representations)
    
    # Visualizar t-SNE
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colores[i] for i in labels], s=4)
    plt.title("t-SNE Visualization")
    plt.show()
    
    # Calcular y mostrar métricas de calidad de clustering
    silhouette_avg = silhouette_score(representations, labels)
    homogeneity = homogeneity_score(labels, labels)
    completeness = completeness_score(labels, labels)
    v_measure = v_measure_score(labels, labels)
    
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Homogeneity Score: {homogeneity:.3f}")
    print(f"Completeness Score: {completeness:.3f}")
    print(f"V-Measure Score: {v_measure:.3f}")

def visualize_tsne(representations, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(representations)
    
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colores[i] for i in labels], s=4)
    plt.title("t-SNE Visualization")
    plt.show()

def evaluate_kmeans(representations, true_labels, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(representations)
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(representations, predicted_labels)
    
    print("Adjusted Rand Index (ARI):", ari)
    print("Silhouette Score:", silhouette)
    
    return ari, silhouette

def evaluate_pca(representations, true_labels):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(representations)
    
    explained_variance = pca.explained_variance_ratio_
    explained_variance_percentage = explained_variance * 100
    title = f"PCA Visualization - Varianza Explicada: {explained_variance_percentage[0]:.2f}% y {explained_variance_percentage[1]:.2f}%"
    
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=[colores[i] for i in true_labels], s=4)
    plt.title(title)
    plt.show()

def silhouette_score_custom(data, labels):
    return silhouette_score(data, labels)

def davies_bouldin_score_custom(data, labels):
    return davies_bouldin_score(data, labels)

def dunn_index(data, labels):
    clusters = np.unique(labels)
    cluster_distances = [data[labels == cluster] for cluster in clusters]
    
    max_intracluster_distance = max([np.max(pdist(cluster)) for cluster in cluster_distances])
    min_intercluster_distance = min(squareform(pdist(data)))
    
    dunn = min_intercluster_distance / max_intracluster_distance
    return dunn

def calinski_harabasz_score_custom (data, labels):
    return calinski_harabasz_score(data, labels)