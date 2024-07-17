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

# Listas para almacenar los puntajes y los hiperparámetros de cada individuo en cada generación
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

def predict_clusters(mts_representations, Y_labels):
    words_labels = ["young" if label == 0 else "old" for label in Y_labels]
    Y_cod = [2 if label == "young" else 1 for label in words_labels]

    # Compute a similarity matrix from the cosine similarity of the representations
    similarity_matrix = cosine_similarity(mts_representations)

    # Normalize the similarity in [0,1]
    similarity_matrix = (similarity_matrix + 1.0) / 2.0

    # Plot similarity matrix
    #fig = plt.figure(figsize=(5, 5))
    #h = plt.imshow(similarity_matrix)
    #plt.title("RC similarity matrix")
    #plt.colorbar(h)
    #plt.show()

    # Reducción de dimensionalidad con Kernel PCA
    kpca = KernelPCA(n_components=2, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(similarity_matrix)
    #plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=Y_labels, s=3)
    #plt.title("PCA embeddings")
    #plt.show()

    # Compute Dissimilarity matrix
    Dist = 1.0 - similarity_matrix
    np.fill_diagonal(Dist, 0)  # due to numerical errors, the diagonal might not be 0

    # Hierarchical clustering
    distArray = squareform(Dist)
    Z = linkage(distArray, 'ward')

    # Ordenar las filas de Z por la distancia de fusión
    Z = Z[np.argsort(Z[:, 2]), :]

    # Encontrar el mejor umbral para 2 clusters
    best_threshold = find_best_threshold(Z)

    print(f"optim_t: {best_threshold}")

    # Calcular los clusters usando fcluster con el umbral encontrado
    clust = fcluster(Z, t=best_threshold, criterion='distance')

    print(f"clust: {clust}")
    print("Found N-clust: (%d)" % len(np.unique(clust)))

    nmi = v_measure_score(Y_cod, clust)
    print("Normalized Mutual Information (v-score): %.3f" % nmi)

    # Plot del Dendrograma con el mismo umbral para color_threshold
    #fig = plt.figure(figsize=(20, 10))
    dn = dendrogram(Z, color_threshold=best_threshold, labels=words_labels, above_threshold_color='k')
    #plt.axhline(y=best_threshold, c='r', linestyle='--')  # Línea roja para el umbral
    #plt.show()

    # Contar el número de clusters en el dendrograma
    num_clusters_dendrogram = count_clusters(dn, best_threshold)
    print(f"N. clusters in dendrogram: {num_clusters_dendrogram}")

    return clust, num_clusters_dendrogram, best_threshold, nmi

##########################################################
##########################################################
##########################################################

def calculate_cluster_accuracy(cluster_ids, young_range, old_range):
    young_cluster = cluster_ids [young_range[0]:young_range[1]+1]
    old_cluster   = cluster_ids [old_range[0]:old_range[1]+1]

    # Contar la frecuencia de cada ID en los clusters
    y_counts = 0  # Suponiendo que el ID 1 es para "old" y el ID 2 para "young"
    for id in young_cluster:
       if id == 2: y_counts += 1
    o_counts = 0  # Suponiendo que el ID 1 es para "old" y el ID 2 para "young"
    for id in old_cluster:
       if id == 1: o_counts += 1

    # Obtener el porcentaje de sujetos jóvenes y mayores
    young_accuracy = (y_counts / len(young_cluster)) * 100
    old_accuracy = (o_counts / len(old_cluster)) * 100

    print("Porcentaje de sujetos jóvenes asignados correctamente:", young_accuracy)
    print("Porcentaje de sujetos mayores asignados correctamente:", old_accuracy)

    homogeneity = homogeneity_score([0]*len(young_cluster) + [1]*len(old_cluster), young_cluster + old_cluster)
    print("Homogeneity Score:", homogeneity)

    return young_accuracy, old_accuracy, homogeneity

def extract_features(internal_representations):
    num_subjects, num_samples, num_neurons = internal_representations.shape
    features = np.zeros((num_subjects, num_neurons * 2))  # Media y varianza para cada neurona

    for i in range(num_subjects):
        subject_data = internal_representations[i]
        mean_features = subject_data.mean(axis=0)
        var_features = subject_data.var(axis=0)
        features[i, :num_neurons] = mean_features
        features[i, num_neurons:] = var_features

    return features
def calculate_performance_metric (mts_representations, Y_labels):

  
    clust, num_clusters_dendrogram, best_threshold, nmi = predict_clusters (mts_representations, Y_labels)

    print(f"Predicted clusters: {clust}")
    print(f"Number of clusters in dendrogram: {num_clusters_dendrogram}")
    print(f"Best threshold: {best_threshold}")
    print(f"Normalized Mutual Information (v-score): {nmi}")

     
    unique_labels = list (set (Y_labels))
    homogeneity = 0
    if len (unique_labels)> 1:
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

# Paso 1: Definir la función de aptitud
def evaluate_fitness (individual, X, Y, config_clus):

    # Configurar el modelo con los hiperparámetros dados en 'individual'
    config_clus ['init_std']         = individual [0]
    config_clus ['init_mean']        = individual [1]
    config_clus ['n_internal_units'] = individual [2]
    config_clus ['spectral_radius']  = individual [3]
    config_clus ['leak']             = individual [4]
    config_clus ['input_scaling']    = individual [5]
    config_clus ['connectivity']     = individual [6]
    config_clus ['noise_level']      = individual [7]
    config_clus ['n_dim']            = individual [8]
    config_clus ['bidir']            = bool (individual [9])
    config_clus ['nonlinearity']     = individual [10]

    print ("^^^^^^^^^^^^^^ evaluate_fitness")
    print (config_clus)
    # Crear y entrenar el modelo
    # Asumiendo que la función MyESN y MyRC ya están definidas
    model_clus = MyESN (config_clus)
    my_rc_clus = MyRC (model_clus, config_clus)
    
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

    return (homogeneity,)

# Paso 2: Definir el tipo de aptitud
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# Función de inicialización para crear individuos
def init_individual (min_vals, max_vals):
    print ("^^^^^^^^^^^^^^ init_individual")
    print (min_vals)
    print (max_vals)
    individual = []
    for min_val, max_val in zip(min_vals, max_vals):
        if isinstance (min_val, bool):
            individual.append (random.choice ([True, False]))
        elif isinstance (min_val, float):
            individual.append (random.uniform (min_val, max_val))
        elif isinstance (min_val, int):
            individual.append(random.randint (min_val, max_val))
        elif isinstance (min_val, list):
            individual.append (random.choice (min_val))
    return creator.Individual (individual)

# Función de mutación

def mutate_integer_part(individual, min_vals, max_vals, indpb):
    for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
        if isinstance(min_val, int):
            if random.random() < indpb:
                individual[i] = random.randint(min_val, max_val)

def mutate_float_part(individual, min_vals, max_vals, indpb):
    for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
        if isinstance(min_val, float):
            if random.random() < indpb:
                individual[i] = random.uniform(min_val, max_val)

def mutate_boolean_part(individual, min_vals, max_vals, indpb):
    for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
        if isinstance(min_val, bool):
            if random.random() < indpb:
                individual[i] = not individual[i]  # Cambia el valor booleano

def mutate_list_part(individual, min_vals, max_vals, indpb):
    for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
        if isinstance(min_val, list):
            if random.random() < indpb:
                individual[i] = random.choice(min_val)  # Cambia el valor de la lista seleccionando un elemento aleatorio

def mutate_individual(individual, min_vals, max_vals, indpb):
    print ("^^^^^^^^^^^^^^ mutate_individual")
    print (individual)
    print (min_vals)
    print (max_vals)
    for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
        print (min_val)
        print (max_val)
        if random.random() < indpb:
            if isinstance(min_val, bool):
                individual[i] = not individual[i]
            elif isinstance(min_val, float):
                individual[i] = random.uniform(min_val, max_val)
            elif isinstance(min_val, int):
                individual[i] = random.randint(min_val, max_val)
            elif isinstance(min_val, list):
                individual[i] = random.choice(min_val)
    print (individual)
    return individual,

def evaluate_individual(individual, X, Y, config_clus):
    print ("^^^^^^^^^^^^^^ evaluate_individual")
    fitness = evaluate_fitness(individual, X, Y, config_clus)
    parameters_history.append (list (individual))  # Guarda los hiperparámetros de este individuo
    return fitness

# Definir la función de cruce (mate)
def mate(ind1, ind2):
    # Implementa la operación de cruce aquí
    # Por ejemplo, podrías implementar un cruce de un solo punto:
    crossover_point = random.randint(0, len(ind1) - 1)
    ind1[crossover_point:], ind2[crossover_point:] = ind2[crossover_point:], ind1[crossover_point:]
    return ind1, ind2

def stop_criteria(population, toolbox):
    # Seleccionar el mejor individuo
    best_ind = tools.selBest(population, k=1)[0]
    best_fitness = best_ind.fitness.values[0]

    # Si la aptitud (accuracy) es 1.0, detener el algoritmo
    if best_fitness == 1.0:
        return True
    else:
        return False

# Importa tu función de optimización y las funciones auxiliares aquí
def main(ICAflag, population_size, max_generations, mutpb, direct_Younger, direct_Older, n_y_subject, n_o_subject,verbose):

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
    dataset_Younger, dataset_Older = load_data (direct_Younger = direct_Younger,direct_Older = direct_Older, n_y_subject = n_y_subject, n_o_subject = n_o_subject, verbose = verbose)
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

    Y_labels = np.where(np.all(Y == [1, 0], axis=1), 'young', 'old')
    Y_labels
    
    # Paso 3: Configurar DEAP
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, min_vals=[0.01, 0.01, 100, 0.1, 0.1, 0.1, 0.1, 0.001, 2, False, ['relu', 'tanh']],
                                          max_vals=[0.5, 0.5, 1000, 1.0, 0.5, 1.0, 1.0, 0.1, 10, True, ['relu', 'tanh']])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mutate", tools.cxBlend, alpha=0.5)
    # Registra la función de cruce en tu Toolbox
    toolbox.register("mate", mate)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    # Registra las funciones de mutación específicas
    
    toolbox.register("mutate_boolean_part", mutate_boolean_part)
    toolbox.register("mutate_list_part", mutate_list_part)
    toolbox.register("mutate_integer_part", mutate_integer_part)
    toolbox.register("mutate_float_part", mutate_float_part)


    toolbox.register("mutate", mutate_individual, min_vals=[0.01, 0.01, 100, 0.1, 0.1, 0.1, 0.1, 0.001, 2, False, ['relu', 'tanh']],
                                          max_vals=[0.5, 0.5, 1000, 1.0, 0.5, 1.0, 1.0, 0.1, 10, True, ['relu', 'tanh']], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual, X=X, Y=Y_labels, config_clus=config_clus)


    # Paso 4: Ejecutar el algoritmo genético
    population = toolbox.population(n=population_size)
    max_generations = max_generations
    
    # Crea un salón de la fama para almacenar el mejor individuo
    hof = tools.HallOfFame (1, similar = np.array_equal)
    # Ejecuta el algoritmo genético
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=mutpb, ngen=max_generations, halloffame=hof, verbose=True)


    if not hasattr(creator, "Individual"):
        # Crear el tipo de individuo si no existe
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Obtener el mejor individuo y su aptitud
    best_ind = tools.selBest(population, k=1)[0]
    best_fitness = best_ind.fitness.values[0]

    print("Mejor configuración de hiperparámetros:", best_ind)
    print("Mejor aptitud (precisión):", best_fitness)

    # Guardar los resultados en un archivo de texto
    with open('results.txt', 'w') as f:
        # Obtener el mejor individuo y su aptitud
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = best_ind.fitness.values[0]
    
        # Escribir los resultados en el archivo
        f.write("Mejor configuración de hiperparámetros: {}\n".format(best_ind))
        f.write("Mejor aptitud (precisión): {}\n".format(best_fitness))
    
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
    #parser.add_argument('--ICAflag', action='store', default='False', help='Activar eliminación de artefactos ICA')
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