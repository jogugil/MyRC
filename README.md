# MyRC And MyESN Project (Reservoir Computing ESN)


Este proyecto implementa un modelo de Reservoir Computing Echo State Network (ESN) para su estudio e implementación. Se ha creado una API para evaluar el uso de este modelo en el procesamiento de señales temporales presentes en los canales de un EEG. El objetivo es la reconstrucción y predicción de señales, la obtención no supervisada de estados neuronales, y la clasificación de tipos de sujetos según los patrones de la dinámica temporal que conserva el estado final del RC para cada sujeto. En concreto, procesamos diferentes sujetos que se clasifican en jóvenes adultos y mayores.
Descripción del Proyecto

# Generación de Datos Sintéticos

Para llevar a cabo los experimentos, primero generamos datos sintéticos que simulan los EEG de diferentes sujetos, creando dos poblaciones distintas: una de jóvenes adultos y otra de mayores. Se consideraron las ondas cerebrales típicas en este tipo de señales, diferenciando su magnitud frecuencial y amplitud según sea un joven adulto o una persona mayor.

# Evaluación con Datos Reales

Después de implementar y probar el modelo con datos sintéticos, se probó con un banco de datos reales de diferentes sujetos. Por motivos de privacidad, estos datos reales no se han subido, pero se incluyen los notebooks y scripts utilizados para el procesamiento de dichos datos.

# Procesamiento de Señales EEG

Las señales EEG procesadas en este proyecto permiten la reconstrucción y predicción de patrones temporales. Además, se utiliza tanto el aprendizaje supervisado como no supervisado para extraer características significativas y realizar la clasificación de sujetos.

# API Construida

Se construyó una API configurada mediante un diccionario config. Este diccionario contiene diferentes parámetros que se transforman en hiperparámetros para el modelo, permitiendo una fácil personalización y ajuste del modelo a diferentes necesidades experimentales.

## Ejemplo de Diccionario de Configuración

      config = {
          'seed': 1,
          'init_type':'orthogonal',
          'init_std':0.01,
          'init_mean':0,
          'input_size':10,
          'n_internal_units': 480,
          'spectral_radius': 0.59,
          'leak': 0.4,
          'input_scaling':0.1,
          'nonlinearity':'relu', # 'relu','tanh'
          'connectivity': 0.6,
          'noise_level': 0.1,
          'n_drop': 100,
          'washout':'init',
          'use_input_bias':True,
          'use_input_layer':True,
          'use_output_bias':True,
          'use_bias':True,
          'readout_type': None,
          'plasticity_synaptic':None,
          'plasticity_intrinsic':None,
          'threshold':0.5,
           'svm_kernel': 'linear',
          'svm_gamma': 0.005,
          'svm_C': 5.0,
          'w_ridge': 5.0,
          'num_epochs': 2000,
          'mlp_layout': (10, 10),
          'w_l2': 0.001,
          'learning_rate':0.9,
          'max_depth':12,
          'n_estimators':100,
          'min_samples_split':1,
          'min_samples_leaf':1,
          'random_state':1,
          'w_ridge_embedding':1.0,
          'mts_rep':'reservoir',
          'bidir': True,
          'circle': False,
          'dimred_method': 'tenpca',
          'n_dim': 44,
          'plasticity_synaptic':'hebb', # 'hebb'.'oja', 'covariance'
          'theta_m':0.01,
          'plasticity_intrinsic':'excitability', # 'excitability', 'activation_function'
          'new_activation_function':'tanh',
          'excitability_factor':0.01,
          'device': 'cpu'
      }
  
# Hiperparámetros del Modelo


# Estructura del Proyecto

    MyRCClass.py: Implementación de la clase MyRCClass.
    fit_evaluate.py: Script para evaluar el rendimiento del modelo.
    data/: Carpeta con datos de prueba y entrenamiento.
    notebooks/: Notebooks de Jupyter para experimentos y análisis.
    README.md: Descripción del proyecto.
    presentation/: Presentación del proyecto.

# Instalación

Para instalar las dependencias del proyecto, ejecuta:
      
      pip install -r requirements.txt

# Uso

Para ejecutar la evaluación, usa el siguiente comando:

# Resultados

Los resultados de las evaluaciones se guardan en la carpeta results/.

# Contacto

Para cualquier consulta, contacta a jogugil@gmail.com/jogugil@alumni.uv.es

