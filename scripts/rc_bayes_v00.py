# Cargamos las luibrerias y creamos las clases
# General imports

import os
import gc
import pickle


import random
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.io
import scipy.spatial.distance as ssd

import pywt

import mne
from mne import io
from mne.datasets import refmeg_noise
from mne.viz import plot_alignment
from mne.preprocessing import ICA
from mne.preprocessing import regress_artifact

from io import StringIO
from pathlib import Path
from pympler import muppy, summary
from contextlib import redirect_stdout
from matplotlib.pyplot import cm

from nilearn import plotting



from deap import tools, algorithms
from deap import base, creator, tools, algorithms

from scipy.signal import welch
from scipy.signal import resample
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.metrics import v_measure_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA
from sklearn.metrics import homogeneity_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

######
from base.eeg import EEG_Data, visualize_eeg_data_mne, read_eeg_data, create_dataset
from base.eeg import create_3d_matrix, flatten_representations, extract_features
from base.eeg import all_channels, eeg_channels, eog_channels, exg_channels
from base.eeg import load_data, assing_channels_dt, filtering_dt, ica_artifact, create_3d_matrix
from base.utils import evaluate_tsne, evaluate_kmeans, evaluate_pca
from base.utils import  calinski_harabasz_score_custom, silhouette_score_custom, davies_bouldin_score_custom

from base.ExractFeatures import ExractFeatures

from base.MyRC_ESN import MyESN
from base.MyRC_ESN import MyRC
from base.MyRC_ESN import print_prediction_results, calculate_dtw_distance, calculate_mean_absolute_error
from base.MyRC_ESN import calculate_mean_squared_error, calculate_pearson_correlation, normalize_time_series

from functools import partial
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.spatial.distance import squareform
from matplotlib.pyplot import get_cmap

#### main print
def script_procesing_EEG (dataset_Younger, dataset_Older, ICA_flag = True):

  ### 2. Asignamos nombre canales
  assing_channels_dt (dataset_Younger,all_channels, eeg_channels, eog_channels, exg_channels)
  assing_channels_dt (dataset_Older, all_channels, eeg_channels, eog_channels, exg_channels)
  ### 3. Aplicamos filtros a los datos
  filtering_dt (dataset_Younger, cut_low = 35, n_decim = 4)
  filtering_dt (dataset_Older, cut_low = 35, n_decim = 4)
  ### 4. Aplicar ICA
  if ICA_flag:
    ica_artifact (dataset_Younger)
    ica_artifact (dataset_Older)

  ### 5. CREAMOS DATASET PARA TRABAJAR CON EL RC

  # Dataset estudio RC from Younger data
  dataset_RC_y = create_3d_matrix (dataset_Younger, dataset_Younger.shape [0])
  dataset_RC_y = np.transpose(dataset_RC_y, (0, 2, 1))

  # Dataset estudio RC from Older data
  dataset_RC_o = create_3d_matrix (dataset_Older, dataset_Older.shape [0])
  dataset_RC_o = np.transpose(dataset_RC_o, (0, 2, 1))

  # Recortar los datasets para que tengan el mismo tamaño
  min_length = min(dataset_RC_y.shape [1], dataset_RC_o.shape [1])
  dataset_RC_y = dataset_RC_y[:,:min_length,:]
  dataset_RC_o = dataset_RC_o[:,:min_length,:]

  # Concatenar los datasets
  dt_classifier = np.concatenate((dataset_RC_y, dataset_RC_o), axis=0)

  # Me invento los targets para young y Old
  # Definir las clases

  # Los datos almacenados en dataset_RC_y y dataset_RC_o son arrays de NumPy tridimensionales (n_subjects, n_muestras, n_channels)
  # Crear etiquetas para cada instancia
  labels_y = np.zeros((dataset_RC_y.shape[0], 2))  # Etiquetas para dataset_RC_y
  labels_y[:, 0] = 1  # Columna 0 con valor 1 para dataset_RC_y

  labels_o = np.zeros((dataset_RC_o.shape[0], 2))  # Etiquetas para dataset_RC_o
  labels_o[:, 1] = 1  # Columna 1 con valor 1 para dataset_RC_o

  # Concatenar las etiquetas
  dt_labels = np.concatenate((labels_y, labels_o), axis=0)

  # Verificar las formas resultantes
  print("Forma de los datos concatenados:", dt_classifier.shape)
  print("Forma de las etiquetas concatenadas:", dt_labels.shape)

  # Normalizamos la matriz de datos eeg dimensiones (muestras, canales)
  normalized_time_series_data = normalize_time_series (dt_classifier) # dataset_RC_y
  print("normalized_time_series_data shape:", normalized_time_series_data.shape)

  X = normalized_time_series_data
  Y = dt_labels

  Y_labels = np.where (np.all(Y == [1, 0], axis=1), 'young', 'old')
  Y_labels

  return X, Y, Y_labels

### bayes

 
# Ejemplo de configuración inicial
config_clus  = {
    'seed': 1,
    'init_type':'orthogonal',
    'init_std':0.1,
    'init_mean':0,
    'input_size': num_input,
    'n_internal_units': num_input*10,
    'spectral_radius': 1,
    'leak': 0.3,
    'input_scaling':0.1,
    'nonlinearity':'relu', # 'relu','tanh'
    'connectivity': 0.1,
    'noise_level': 0.1,
    'n_drop': 50,
    'washout':'init',
    'use_input_bias':True,
    'use_input_layer':True,
    'use_output_bias':True,
    'use_bias':True,
    'readout_type': None,
    'threshold':0.3,
    'svm_kernel': 'linear',
    'svm_gamma': 0.005,
    'svm_C': 5.0,
    'w_ridge': 5.0,
    'w_ridge_embedding':18.80,
    'learning_rate': 0.001,
    'learning_rate_type': 'constant',
    'mts_rep':'reservoir',
    'bidir': True,
    'circle': False,
    'dimred_method': 'tenpca',
    'n_dim': 36,
    'mlp_layout': (10, 10),
    'mlp_w_l2': 0.001,
    'mlp_num_epochs': 2000,
    'mlp_batch_size': 32,
    'mlp_learning_rate': 0.01,
    'mlp_learning_rate_type': 'constant',
    'plasticity_synaptic': None, # 'hebb'.'oja', 'covariance'
    'theta_m':0.01,
    'plasticity_intrinsic':None, # 'excitability', 'activation_function'
    'new_activation_function':'tanh',
    'excitability_factor':0.01,
    'device': 'cpu'
}

def find_best_threshold(Z, desired_clusters=2, margin=1e-5):
    optim_t = 0
    for i in range(Z.shape[0]):
        optim_t = Z[i, 2]
        clust = fcluster(Z, t=optim_t, criterion='distance')
        n_clusts = len(np.unique(clust))
        if n_clusts == desired_clusters:
            break
    # Ajuste ligero para asegurar consistencia en el borde
    clust = fcluster(Z, t=optim_t + margin, criterion='distance')
    if len(np.unique(clust)) == desired_clusters:
        optim_t += margin
    return optim_t

def count_clusters(dn, threshold):
    icoord = np.array(dn['icoord'])
    dcoord = np.array(dn['dcoord'])
    color_list = np.array(dn['color_list'])

    clusters = Counter()
    for d, c in zip(dcoord, color_list):
        if d[1] <= threshold:
            clusters[c] += 1

    return len(clusters)

def calculate_performance_metric(input_repr, Y_label):
    Y_cod = list(map(lambda y_: 2 if y_ == 'young' else 1, Y_label))

    cmap = get_cmap('tab20')
    if cmap == None:
      cmap = plt.cm.tab20(np.linspace(0, 1, 12))
      plt.register_cmap(name='tab20', cmap=plt.cm.tab20)
    similarity_matrix = cosine_similarity(input_repr)
    similarity_matrix = (similarity_matrix + 1.0) / 2.0

    fig = plt.figure(figsize=(5, 5))
    h = plt.imshow(similarity_matrix)
    plt.title("RC similarity matrix")
    plt.colorbar(h)
    plt.show()

    colores = ['blue', 'red', 'green', 'orange', 'purple']
    kpca = KernelPCA(n_components=2, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(similarity_matrix)
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=[colores[i] for i in Y_cod], s=4)
    plt.title("PCA embeddings")
    plt.show()

    Dist = 1.0 - similarity_matrix
    np.fill_diagonal(Dist, 0)
    distArray = squareform(Dist)
    Z = linkage(distArray, 'ward')
    Z = Z[np.argsort(Z[:, 2]), :]

    best_threshold = find_best_threshold(Z)

    print(f"optim_t: {best_threshold}")

    clust = fcluster(Z, t=best_threshold, criterion='distance')

    print(f"clust: {clust}")
    print("Found N-clust: (%d)" % len(np.unique(clust)))

    nmi = v_measure_score(Y_cod, clust)
    print("Normalized Mutual Information (v-score): %.3f" % nmi)

    fig = plt.figure(figsize=(20, 10))
    dn = dendrogram(Z, color_threshold=best_threshold, above_threshold_color='k')
    plt.axhline(y=best_threshold, c='r', linestyle='--')
    plt.show()

    num_clusters_dendrogram = count_clusters(dn, best_threshold)
    print(f"N. clusters in dendrogram: {num_clusters_dendrogram}")
    print("N. clusters: ", np.unique(dn['color_list']).shape[0] - 1)

    accuracy = accuracy_score(Y_cod, clust)
    print("Accuracy of cluster labels:", accuracy)

    return accuracy

@use_named_args([
    Real(0.1, 0.5, name='spectral_radius'),
    Real(0.1, 0.9, name='leak'),
    Real(0.1, 1.0, name='input_scaling'),
    Integer(100, 1000, name='n_internal_units'),
    Integer(2, 10, name='n_dim'),
    Categorical(['relu', 'tanh'], name='nonlinearity'),  # Hiperparámetro de lista
    Categorical([True, False], name='bidir')             # Hiperparámetro binario
])

def evaluate_model(spectral_radius, leak, input_scaling, n_internal_units, n_dim, nonlinearity, bidir):
    global X, Y_label

    config_clus['spectral_radius'] = spectral_radius
    config_clus['leak'] = leak
    config_clus['input_scaling'] = input_scaling
    config_clus['n_internal_units'] = n_internal_units
    config_clus['n_dim'] = n_dim
    config_clus['nonlinearity'] = nonlinearity
    config_clus['bidir'] = bidir

    print(f'config_clus[n_internal_units]: {n_internal_units}')
    print(f'config_clus[spectral_radius]: {spectral_radius}')
    print(f'config_clus[leak]: {leak}')
    print(f'config_clus[input_scaling]: {input_scaling}')
    print(f'config_clus[n_dim]: {n_dim}')
    print(f'config_clus[nonlinearity]: {nonlinearity}')
    print(f'config_clus[bidir]: {bidir}')

    model_clus = MyESN(config_clus)
    my_rc_clus = MyRC(model_clus, config_clus)
    
    print(config_clus)
    print(f'my_rc_clus.fit (X)')
    result_rc = my_rc_clus.fit (X)
    print(f'result_rc')
    output_rc_layer, mtx_output_rc_layer, reservoir_state, mtx_rc_state, input_repr, mtx_input_repr, red_states, mtx_red_states = result_rc
    print(f'output_rc_layer:{output_rc_layer.shape}')
    print(f'mtx_output_rc_layer:{mtx_output_rc_layer.shape}')
    print(f'reservoir_state:{reservoir_state.shape}')
    print(f'mtx_rc_state:{mtx_rc_state.shape}')
    print(f'input_repr:{input_repr.shape}')
    print(f'mtx_input_repr:{mtx_input_repr.shape}')
    print(f'red_states:{red_states.shape}')
    print(f'mtx_red_states:{mtx_red_states.shape}')
    
    accuracy2 = calculate_performance_metric(mtx_output_rc_layer, Y_label)
    print(f'accuracy: {accuracy2}')
    accuracy1 = calculate_performance_metric(output_rc_layer, Y_label)
    print(f'accuracy: {accuracy1}')
    return -accuracy1

# Importa tu función de optimización y las funciones auxiliares aquí
def main(ICAflag, n_calls, direct_Younger, direct_Older, n_y_subject, n_o_subject, verbose):
    global X, Y_label
    print("ICA flag:", ICAflag)
    print("n_calls size:", n_calls)
    print("Direct Younger:", direct_Younger)
    print("Direct Older:", direct_Older)
    print("Number of younger subjects:", n_y_subject)
    print("Number of older subjects:", n_o_subject)
    print("verbose:", verbose)
    
    # Load data and preprocess
    dataset_Younger, dataset_Older = load_data(direct_Younger=direct_Younger, direct_Older=direct_Older, n_y_subject=n_y_subject, n_o_subject=n_o_subject, verbose = verbose)
    X, Y, Y_label = script_procesing_EEG(dataset_Younger, dataset_Older, ICA_flag=ICAflag)

    # Define the hyperparameter search space
    space = [
        Real(0.1, 0.5, name='spectral_radius'),
        Real(0.1, 0.9, name='leak'),
        Real(0.1, 1.0, name='input_scaling'),
        Integer(100, 1000, name='n_internal_units'),
        Integer(2, 10, name='n_dim'),
        Categorical(['relu', 'tanh'], name='nonlinearity'),
        Categorical([True, False], name='bidir')
    ]

    # Run Bayesian optimization
    result = gp_minimize(evaluate_model, space, n_calls=n_calls, random_state=42)

    # Print and save the best hyperparameters found and their performance
    best_params = dict(zip(['spectral_radius', 'leak', 'input_scaling', 'n_internal_units', 'n_dim', 'nonlinearity', 'bidir'], result.x))
    best_performance = -result.fun
    print("Mejores hiperparámetros encontrados:", best_params)
    print("Mejor rendimiento (precisión del clustering):", best_performance)
    
    file_path = "best_bayes_hyperparameters.txt"
    with open(file_path, 'w') as file:
        file.write("Mejores hiperparámetros encontrados:\n")
        for key, value in best_params.items():
            file.write(f"{key}: {value}\n")
        file.write(f"Mejor rendimiento (precisión del clustering): {best_performance}\n")

if __name__ == "__main__":
    # Configura los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutar algoritmo genético.')
    parser.add_argument('--ICAflag', action='store_true', default=False, help='Activar eliminación de artefactos ICA')
    parser.add_argument('--n_calls', type=int, default=30, help='Número de evaluaiones')
    parser.add_argument('--direct_Younger', type=str, default='./Younger', help='Directorio para sujetos jóvenes')
    parser.add_argument('--direct_Older', type=str, default='./Older', help='Directorio para sujetos mayores')
    parser.add_argument('--n_y_subject', type=int, default=12, help='Número de sujetos jóvenes')
    parser.add_argument('--n_o_subject', type=int, default=12, help='Número de sujetos mayores')
    parser.add_argument('--verbose', action='store_true', default=False, help='Activar trazas')
    args = parser.parse_args()

    # Llama a la función principal con los argumentos proporcionados
    main(args.ICAflag, args.n_calls, args.direct_Younger, args.direct_Older, args.n_y_subject, args.n_o_subject, args.verbose)