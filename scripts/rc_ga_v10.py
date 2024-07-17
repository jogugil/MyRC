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
from collections import Counter

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
from scipy.spatial.distance import squareform
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


# Inicializar listas para almacenar los puntajes y los hiperparámetros de cada individuo en cada generación
scores = []
parameters_history = []
num_input = 64
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

def predict_clusters(mts_representations, Y_labels):
    words_labels = ["young" if label == 0 else "old" for label in Y_labels]
    Y_cod = [2 if label == "young" else 1 for label in words_labels]

    similarity_matrix = cosine_similarity(mts_representations)
    similarity_matrix = (similarity_matrix + 1.0) / 2.0

    #fig = plt.figure(figsize=(5, 5))
    #h = plt.imshow(similarity_matrix)
    #plt.title("RC similarity matrix")
    #plt.colorbar(h)
    # plt.show()

    kpca = KernelPCA(n_components=2, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(similarity_matrix)
    #plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=Y_labels, s=3)
    #plt.title("PCA embeddings")
    #plt.show()

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

    #fig = plt.figure(figsize=(20, 10))
    dn = dendrogram(Z, color_threshold=best_threshold, labels=words_labels, above_threshold_color='k')
    #plt.axhline(y=best_threshold, c='r', linestyle='--')
    #plt.show()

    num_clusters_dendrogram = count_clusters(dn, best_threshold)
    print(f"N. clusters in dendrogram: {num_clusters_dendrogram}")

    return clust, num_clusters_dendrogram, best_threshold, nmi

def calculate_cluster_accuracy(cluster_ids, young_range, old_range):
    young_cluster = cluster_ids[young_range[0]:young_range[1]+1]
    old_cluster = cluster_ids[old_range[0]:old_range[1]+1]

    y_counts = sum(1 for id in young_cluster if id == 2)
    o_counts = sum(1 for id in old_cluster if id == 1)

    young_accuracy = (y_counts / len(young_cluster)) * 100
    old_accuracy = (o_counts / len(old_cluster)) * 100

    print("Porcentaje de sujetos jóvenes asignados correctamente:", young_accuracy)
    print("Porcentaje de sujetos mayores asignados correctamente:", old_accuracy)

    homogeneity = homogeneity_score([0]*len(young_cluster) + [1]*len(old_cluster), young_cluster + old_cluster)
    print("Homogeneity Score:", homogeneity)

    return young_accuracy, old_accuracy, homogeneity

def extract_features(internal_representations):
    num_subjects, num_samples, num_neurons = internal_representations.shape
    features = np.zeros((num_subjects, num_neurons * 2))
    for i in range(num_subjects):
        subject_data = internal_representations[i]
        mean_features = subject_data.mean(axis=0)
        var_features = subject_data.var(axis=0)
        features[i, :num_neurons] = mean_features
        features[i, num_neurons:] = var_features
    return features

def calculate_performance_metric (mts_representations, Y_labels):
    print (f"************ Y_labels :{Y_labels}")
    clust, num_clusters_dendrogram, best_threshold, nmi = predict_clusters (mts_representations, Y_labels)

    print(f"Predicted clusters: {clust}")
    print(f"Number of clusters in dendrogram: {num_clusters_dendrogram}")
    print(f"Best threshold: {best_threshold}")
    print(f"Normalized Mutual Information (v-score): {nmi}")

    unique_labels = list(set(Y_labels))
   
    label_to_cluster = {unique_labels[0]: 0, unique_labels[1]: 1}
 
    clusters = {}
    for i, label in enumerate (Y_labels):
        cluster_id = label_to_cluster [label]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)

    for cluster_id, subject_ids in clusters.items():
        print(f"Cluster {cluster_id}: {subject_ids}")

    young_range = (0, 11)
    old_range   = (12, 23)
    young_accuracy, old_accuracy, homogeneity = calculate_cluster_accuracy(list(clust), young_range, old_range)
    
    print("old_accuracy Score:", old_accuracy)
    print("young_accuracy Score:", young_accuracy)
    print("Homogeneity Score:", homogeneity)

    return homogeneity

def evaluate_fitness(individual, X, Y, config_clus):
    config_clus['init_std'] = individual[0]
    config_clus['init_mean'] = individual[1]
    config_clus['n_internal_units'] = individual[2]
    config_clus['spectral_radius'] = individual[3]
    config_clus['leak'] = individual[4]
    config_clus['input_scaling'] = individual[5]
    config_clus['connectivity'] = individual[6]
    config_clus['noise_level'] = individual[7]
    config_clus['n_dim'] = individual[8]
    config_clus['bidir'] = bool(individual[9])
    config_clus['nonlinearity'] = individual[10]


    print ("^^^^^^^^^^^^^^ evaluate_fitness")
    print (config_clus)
    # Crear y entrenar el modelo
    # Asumiendo que la función MyESN y MyRC ya están definidas
    model_clus = MyESN (config_clus)
    my_rc_clus = MyRC (model_clus, config_clus)
    print (f"************ Y :{Y}")
    
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
    
    accuracy2 = calculate_performance_metric(mtx_output_rc_layer, Y )
    print(f'accuracy: {accuracy2}')
    accuracy1 = calculate_performance_metric(output_rc_layer, Y )
    print(f'accuracy: {accuracy1}')
    
    feature_representations = extract_features (mtx_rc_state)
    # Evaluar el rendimiento del modelo y devolver la aptitud
    homogeneity = calculate_performance_metric (output_rc_layer, Y) #feature_representations

    # Guardar la homogeneidad como puntaje en la lista de puntajes
    scores.append(homogeneity)


    return homogeneity,

def custom_mutate(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.uniform(0.001, 0.1)  # init_std
    if random.random() < indpb:
        individual[1] = random.uniform(0.0, 0.5)    # init_mean
    if random.random() < indpb:
        individual[2] = random.randint(100, 1000)   # n_internal_units
    if random.random() < indpb:
        individual[3] = random.uniform(0.1, 1.5)    # spectral_radius
    if random.random() < indpb:
        individual[4] = random.uniform(0.01, 1.0)   # leak
    if random.random() < indpb:
        individual[5] = random.uniform(0.01, 1.0)   # input_scaling
    if random.random() < indpb:
        individual[6] = random.uniform(0.01, 1.0)   # connectivity
    if random.random() < indpb:
        individual[7] = random.uniform(0.0, 0.1)    # noise_level
    if random.random() < indpb:
        individual[8] = random.randint(2, 64)       # n_dim
    if random.random() < indpb:
        individual[9] = random.randint(0, 1)        # bidir
    if random.random() < indpb:
        individual[10] = random.choice(['relu', 'tanh'])  # nonlinearity
    return individual,

def optimize_hyperparameters(X, Y, config_clus, generations=10, population_size=20, mutpb = 0.2):
    # Establece la semilla aleatoria para reproducibilidad
    random.seed(config_clus['seed'])

    # Crea una clase de fitness para maximizar (mejorar el rendimiento)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Crea una clase de individuo que es una lista con un atributo de fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    Y_labels = np.where(np.all(Y == [1, 0], axis=1), 'young', 'old')
    Y_labels
    
    # Inicializa la caja de herramientas de DEAP para registrar funciones
    toolbox = base.Toolbox()

    # Registra funciones para inicializar cada hiperparámetro con valores aleatorios dentro de los rangos dados
    toolbox.register("init_std", random.uniform, 0.001, 0.1)          # init_std entre 0.001 y 0.1
    toolbox.register("init_mean", random.uniform, 0.0, 0.5)            # init_mean entre 0.0 y 0.5
    toolbox.register("n_internal_units", random.randint, 100, 1000)    # n_internal_units entre 100 y 1000
    toolbox.register("spectral_radius", random.uniform, 0.1, 1.5)      # spectral_radius entre 0.1 y 1.5
    toolbox.register("leak", random.uniform, 0.01, 1.0)                # leak entre 0.01 y 1.0
    toolbox.register("input_scaling", random.uniform, 0.01, 1.0)       # input_scaling entre 0.01 y 1.0
    toolbox.register("connectivity", random.uniform, 0.01, 1.0)        # connectivity entre 0.01 y 1.0
    toolbox.register("noise_level", random.uniform, 0.0, 0.1)          # noise_level entre 0.0 y 0.1
    toolbox.register("n_dim", random.randint, 2, 64)                   # n_dim entre 2 y 64
    toolbox.register("bidir", random.randint, 0, 1)                    # bidir como 0 o 1 (booleano)
    toolbox.register("nonlinearity", random.choice, ['relu', 'tanh'])  # nonlinearity como 'relu' o 'tanh'



    # Registra la función para crear un individuo combinando los valores de los hiperparámetros
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.init_std, toolbox.init_mean, toolbox.n_internal_units, toolbox.spectral_radius,
                      toolbox.leak, toolbox.input_scaling, toolbox.connectivity, toolbox.noise_level,
                      toolbox.n_dim, toolbox.bidir, toolbox.nonlinearity), n=1)

    # Registra la función para crear una población de individuos
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Registra la función de evaluación de fitness utilizando los datos X, Y y la configuración del modelo
    toolbox.register("evaluate", evaluate_fitness, X=X, Y=Y_labels, config_clus=config_clus)

    # Registra la función de cruce de dos puntos
    toolbox.register("mate", tools.cxTwoPoint)

    # Registra la función de mutación uniforme con rango [0, 1] y una probabilidad de 0.2
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.2)

    # Registra la función de mutación personalizada con una probabilidad individual de 0.2
    toolbox.register("mutate", custom_mutate, indpb=0.2)

    # Registra la función de selección por torneo con tamaño de torneo 3
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crea una población inicial de individuos de tamaño population_size
    pop = toolbox.population(n=population_size)

    # Crea un salón de la fama para almacenar el mejor individuo
    hof = tools.HallOfFame(1, similar=np.array_equal)

    # Define estadísticas para recopilar durante la evolución
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Ejecuta el algoritmo genético simple con la población inicial, caja de herramientas,
    # probabilidad de cruce 0.5, probabilidad de mutación 0.2, durante un número de generaciones
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    # Retorna la población final, las estadísticas y el salón de la fama
    return pop, stats, hof


# Importa tu función de optimización y las funciones auxiliares aquí
def main(ICAflag, population_size, max_generations, mutpb, direct_Younger, direct_Older, n_y_subject, n_o_subject, verbose):

    print("ICA flag:", ICAflag)
    print("Population size:", population_size)
    print("Max generations:", max_generations)
    print("Direct Younger:", direct_Younger)
    print("Direct Older:", direct_Older)
    print("Mutation probability:", mutpb)
    print("Number of younger subjects:", n_y_subject)
    print("Number of older subjects:", n_o_subject)
    print("verbose:", verbose)
    
    ### MAIN: Cargamos los datos; Preprocesamos EEG ; Y Creamos dataset de estudio
    dataset_Younger, dataset_Older = load_data (direct_Younger = direct_Younger,direct_Older = direct_Older, n_y_subject = n_y_subject, n_o_subject = n_o_subject,verbose = verbose)
    X, Y, Y_label = script_procesing_EEG (dataset_Younger, dataset_Older, ICA_flag = ICAflag)

    ######## Eliminamos memoria

    # Obtener un resumen del uso de memoria
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)

    # Imprimir el nuevo resumen
    summary.print_(sum1)

    del dataset_Younger
    del dataset_Older

    # Ejecutar el recolector de basura
    gc.collect()

    # Obtener un nuevo resumen del uso de memoria
    all_objects = muppy.get_objects()
    sum2 = summary.summarize(all_objects)
    # Imprimir el nuevo resumen
    summary.print_(sum2)
 


    ##### Lanzamos GA:


    # Paso 3: Configurar DEAP
    generations     = max_generations
    population_size = population_size
    population, stats, hall_of_fame = optimize_hyperparameters(X, Y, config_clus, generations, population_size, mutpb)

    print("Mejores individuos encontrados:")
    for individual in hall_of_fame:
        print(individual)
    # Guardar los resultados en un archivo de texto
    # Guardar los mejores individuos encontrados en un archivo de texto
    with open('mejores_individuos.txt', 'w') as f:
        f.write("Mejores individuos encontrados:\n")
        for individual in hall_of_fame:
            f.write(str(individual) + '\n')
        # Graficar la evolución del puntaje y los hiperparámetros
        plt.figure(figsize=(12, 6))
    
    # Gráfico de la evolución del puntaje
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(scores) + 1), scores, marker='o', color='b')
    plt.xlabel('Generación')
    plt.ylabel('Puntaje')
    plt.title('Evolución del puntaje')
    
    # Gráfico de la evolución de los hiperparámetros
    parameters_history = np.array(parameters_history)
    num_params = parameters_history.shape[1]  # Obtener el número de hiperparámetros
    num_cols = 2  # Número de columnas para los subgráficos
    num_rows = (num_params + num_cols - 1) // num_cols  # Calcular el número de filas necesario
    
    plt.figure(figsize=(15, 5 * num_rows))  # Ajustar el tamaño de la figura según el número de filas
    
    for i in range(num_params):
        plt.subplot(num_rows, num_cols, i + 1)  # Seleccionar el subgráfico correspondiente
        plt.plot(range(1, len(scores) + 1), parameters_history[:, i])
        plt.xlabel('Generación')
        plt.ylabel(f'Valor de Param{i}')
        plt.title(f'Evolución de Param{i}')
        plt.grid(True)
    
    plt.tight_layout()  # Ajustar automáticamente los espacios entre los subgráficos
    
    # Guardar la gráfica en formato PNG
    plt.savefig('evolucion_hiperparametros.png')
    
    # Mostrar la gráfica
    plt.show()
if __name__ == "__main__":
    # Configura los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutar algoritmo genético.')
    parser.add_argument('--ICAflag', action='store_true', default=False, help='Activar eliminación de artefactos ICA')
    parser.add_argument('--population', type=int, default=30, help='Tamaño de la población en cada generación')
    parser.add_argument('--max_generations', type=int, default=10, help='Número máximo de generaciones')
    parser.add_argument('--mutpb', type=float, default=0.2, help='Probabilidad de mutación de los hiperparámetros')
    parser.add_argument('--direct_Younger', type=str, default='./Younger', help='Directorio para sujetos jóvenes')
    parser.add_argument('--direct_Older', type=str, default='./Older', help='Directorio para sujetos mayores')
    parser.add_argument('--n_y_subject', type=int, default=12, help='Número de sujetos jóvenes')
    parser.add_argument('--n_o_subject', type=int, default=12, help='Número de sujetos mayores')
    parser.add_argument('--verbose', action='store_true', default=False, help='Activar trazas')
    args = parser.parse_args()

    # Llama a la función principal con los argumentos proporcionados
    main(args.ICAflag, args.population, args.max_generations,  args.mutpb, args.direct_Younger, args.direct_Older, args.n_y_subject, args.n_o_subject, args.verbose)