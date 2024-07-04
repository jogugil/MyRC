import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
 
import re
import os
import time
import random
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from base.tensorPCA import tensorPCA

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.svm import SVC
from scipy.io import loadmat 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score, classification_report)
'''
### **Valores de configuración del Recervoir:**

- **n_internal_units**: Tamaño del reservorio, indicando el número de unidades internas o nodos en el reservorio.

- **spectral_radius**: El mayor valor propio del reservorio. Es una medida de la dispersión de las activaciones en el reservorio y puede afectar las propiedades dinámicas del reservorio.

- **leak**: Cantidad de fuga en la actualización del estado del reservorio. Si es None o 1.0, no hay fuga. Un valor menor que 1.0 introduce un decaimiento en el estado del reservorio. Notar que 'la cantidad de fuga' controla cuánta información del estado anterior del reservorio se mantiene en cada iteración, y cómo se combina con la nueva información para actualizar el estado del reservorio. Este parámetro puede afectar la dinámica y la capacidad de memoria del reservorio en tareas específicas.

- **connectivity**: Porcentaje de conexiones no nulas en el reservorio. Controla la dispersión del reservorio.

- **input_scaling**: Escala de los pesos de entrada. Controla la fuerza de la señal de entrada.

- **noise_level**: Nivel de ruido en la actualización del estado del reservorio. Introduce perturbaciones aleatorias en la dinámica del reservorio.

- **n_drop**: Estados transitorios a eliminar. Los estados iniciales del reservorio a menudo están afectados por las condiciones iniciales y pueden no ser representativos de la verdadera dinámica del reservorio.

- **bidir**: Si es True, usa un reservorio bidireccional. Los reservorios bidireccionales incorporan información de ambos estados pasados y futuros. (idem lstm)

- **circ**: Usa un reservorio con topología circular. Si es True, las conexiones del reservorio forman un círculo.

- **dimred_method**: Método de reducción de dimensionalidad. Las opciones incluyen None (sin reducción de dimensionalidad), 'pca' (análisis de componentes principales) y 'tenpca' (análisis de componentes principales tensorial).

- **n_dim**: Número de dimensiones resultantes después del procedimiento de reducción de dimensionalidad.

- **mts_rep**: Tipo de representación de MTS (series temporales multivariadas). Las opciones incluyen 'last', 'mean', 'output' y 'reservoir'.Cada opción tiene sus propias implicaciones en términos de qué información se incluye en la representación y cómo se utiliza para la tarea específica que estás abordando. La elección dependerá del contexto y de la naturaleza de los datos y la tarea.


        * 'last': La representación de la serie temporal multivariada (MTS) se toma como la última salida generada por el reservorio. En otras palabras, para cada secuencia de entrada, el valor final del estado del reservorio se utiliza como representación de la MTS.

        * 'mean': La representación se obtiene calculando la media de las salidas generadas por el reservorio a lo largo de toda la secuencia de entrada. Este enfoque podría proporcionar una representación más suavizada de la MTS al considerar el promedio de las activaciones del reservorio.

        * 'output': La representación se toma directamente de la salida del reservorio. Esto significa que cada paso de tiempo de la salida del reservorio se utiliza como parte de la representación de la MTS.

        * 'reservoir': La representación se toma de alguna característica específica del reservorio. Esto puede implicar la utilización de ciertas propiedades o activaciones internas del reservorio como representación de la MTS. El detalle específico de qué característica del reservorio se utiliza depende de la implementación del modelo.

- **w_ridge_embedding**: Parámetro de regularización de la regresión de ridge para la incrustación.

- **readout_type**: Tipo de lectura utilizado para la clasificación. Las opciones incluyen 'lin' (lineal), 'mlp' (perceptrón multicapa) y 'svm' (máquina de soporte vectorial).

- **w_ridge**: Regularización de la regresión de ridge para la lectura lineal.

- **svm_gamma**: Ancho del kernel de función de base radial (RBF) para la lectura SVM.

- **svm_C**: Regularización para el hiperplano SVM.

- **mlp_layout**: Neuronas en cada capa del perceptrón multicapa para la lectura MLP.

- **num_epochs**: Número de épocas para entrenar la lectura MLP.

- **w_l2**: Peso de la regularización L2 para la lectura MLP.

- **nonlinearity**: Tipo de función de activación para la lectura MLP. Las opciones incluyen 'relu', 'tanh', 'logistic' e 'identity'.

Después de imprimir la configuración, el código carga un conjunto de datos utilizando el nombre del conjunto de datos especificado y realiza la **codificación one-hot para las etiquetas (Ytr e Yte)**

En un Reservoir Computing, el estado del reservorio se actualiza multiplicando 
    el estado actual del reservorio por una matriz de pesos (o conexiones)  (aplicación
    capa de entrada)y luego aplicando una función no lineal, comúnmente una tangente 
    hiperbólica (tanh). 
    
    1. Concatenación de la entrada con el estado actual del reservorio. De esta manera,
    introducimos la historia pasada de la entrada en la computación del reservorio.
    
    En un ESN, el reservorio tiene conexiones recurrentes que permiten que la información 
    se propague y se almacene en el estado del reservorio a lo largo del tiempo. Concatenar 
    la entrada actual con el estado actual del reservorio antes de aplicar la capa de entrada
    permite que la información de la entrada y la información histórica del reservorio se 
    combinen de manera efectiva.
    
    Al concatenar la entrada actual con el estado del reservorio, estás permitiendo que la 
    red considere tanto la entrada actual como la información almacenada en el estado del 
    reservorio, lo que puede ser crucial para la captura de patrones temporales complejos.
    
    2. Aplicar la capa de entrada del reservorio a la entrada combinada. Esta capa de entrada
    transforma la entrada combinada utilizando pesos sinápticos, y la salida se convierte en 
    el nuevo estado del reservorio.
        
    3. Función de Activación: Aplica una función de activación no lineal a la salida de la 
    capa de entrada. Esto introduce no linealidades en el proceso y permite que el reservorio
    capture patrones complejos.
    
    4. Actualización del Estado del Reservorio: El resultado de la función de activación se 
    convierte en el nuevo estado del reservorio.

    El objetivo de este proceso es permitir que el reservorio capture y mantenga información 
    relevante a lo largo del tiempo, lo que es crucial para tareas que involucran secuencias 
    temporales.
    
        x = x.expand(reservoir_state.size())  # Expande las dimensiones de x para que coincidan con reservoir_state

        # Reemplaza la línea con el error con la siguiente
        reservoir_input = torch.cat((x, reservoir_state), dim=1)
        input_reservoir_transformed = torch.tanh(self.input_layer(reservoir_input))

        # Luego, en la actualización del estado del reservorio
        reservoir_state = (1 - self.config['leak']) * reservoir_state + \
                          self.config['leak'] * input_reservoir_transformed + \
                          self.config['noise_level'] * torch.randn_like(reservoir_state)
                          
    Nota: La última operación  representa la actualización del estado del reservorio, 
    considerando la fuga del estado anterior, la influencia de la nueva entrada transformada 
    y la adición de ruido. Esto refleja el comportamiento dinámico de un Reservoir Computing, 
    donde el estado del reservorio evoluciona en el tiempo en respuesta a las entradas y la 
    dinámica interna.
    
        - Leakage (Fuga):
              
             + self.config['leak'] es el parámetro que controla la cantidad de "fuga" en el 
              estado del reservorio.
             +  (1 - self.config['leak']) representa el término de "no fuga", lo que significa 
              que es la parte del estado del reservorio que se mantiene sin cambios.
        - Entrada transformada:
            + input_reservoir_transformed es la salida transformada de la capa de entrada del reservorio, 
              que ya incluye la aplicación de la función de activación (torch.tanh) a la combinación 
              lineal de la entrada y el estado del reservorio.
            + self.config['leak'] controla la cantidad de influencia que tiene esta nueva entrada transformada 
             en el estado del reservorio.  
        - Ruido:
            + self.config['noise_level'] es el parámetro que controla el nivel de ruido que 
            se agrega al estado del reservorio.
            + torch.randn_like(reservoir_state) genera un tensor de números aleatorios de la 
            misma forma que reservoir_state, introduciendo así ruido.
            
            
        "Descartar transitorios" implica que se está desechando un número determinado de 
        estados iniciales del reservorio antes de comenzar el entrenamiento real o la 
        predicción. Este enfoque es común en Reservoir Computing para asegurarse de que el 
        sistema haya convergido a un estado estable y estacionario antes de utilizar los 
        datos.
        
        
            - reservoir_input: Se concatena la entrada actual x con el estado actual del 
            reservorio reservoir_state a lo largo de la dimensión 1 utilizando 
                        torch.cat((x, reservoir_state), dim=1). 
            Esto crea un nuevo tensor reservoir_input que combina la entrada con el estado 
            del reservorio.

            - Transformación de la entrada del reservorio: Se aplica la función tangente 
            hiperbólica (torch.tanh) a la salida de la capa de entrada del reservorio 
            (self.input_layer(reservoir_input)). Esto transforma la entrada combinada y 
            la lleva a través de una función de activación no lineal.

            - Actualización del estado del reservorio: Se actualiza el estado del reservorio 
            utilizando la regla del reservorio. La nueva actualización se calcula como sigue:
            
                    a) La componente actual del estado del reservorio reservoir_state 
                    se multiplica por 1 - self.config['leak'].
                   b) A esta cantidad se le suma la componente transformada de la entrada 
                   del reservorio multiplicada por self.config['leak'].
                    c) Se añade ruido gaussiano multiplicado por self.config['noise_level'] 
                    al estado del reservorio.
                    
        En un RC de tipo Echo State Network (ESN), tiene como objetivo eliminar los efectos g
        iniciales y no representativos del estado del reservorio. Cuando se inicia el 
        procesamiento de una serie temporal, el estado del reservorio puede estar en un 
        estado inicial arbitrario y no necesariamente representativo de la dinámica que 
        queremos capturar. El descarte de transitorios ayuda a estabilizar el comportamiento 
        del reservorio y a comenzar desde un estado más consistente antes de utilizar los
        datos de entrada.

        En términos prácticos, el descarte de transitorios se realiza durante las primeras 
        iteraciones del procesamiento de datos. Durante este período, el estado del 
        reservorio se ajusta gradualmente a la entrada y a las condiciones iniciales, 
        eliminando así la influencia de cualquier estado inicial no representativo. 
        Después de descartar estos transitorios, el reservorio debería encontrarse en un 
        estado más "equilibrado" o "representativo", y a partir de ahí, el sistema puede 
        comenzar a aprender y modelar la dinámica de la serie temporal de manera más efectiva.
        
        Se busca, por tanto, garantizar que el reservorio comience desde un estado 
        consistente antes de procesar datos de entrada, mejorando así la capacidad del 
        modelo para capturar patrones y realizar predicciones precisas.

'''
class MyESN (nn.Module):
    '''
   
        El propósito de multiplicar la matriz de conexiones del reservorio por el estado anterior es permitir que las interacciones entre los nodos del reservorio,
        mediadas por las conexiones, afecten la evolución del estado del reservorio en el siguiente paso de tiempo. Esto introduce una dinámica no lineal en el sistema,
        que es fundamental para el poder de procesamiento del Reservoir Computing.
        
        Es importante recordar que la capacidad de memoria y procesamiento en las ESN proviene de la dinámica no lineal de los nodos internos y  las conexiones recurrentes en la red, lo que permite que la información se procese y se mantenga en el tiempo para generar comportamientos
        complejos y adaptativos.
        
        * Plasticidad Sináptica (Reglas de Aprendizaje):
        
        - Hebb: Esta regla fortalece las conexiones entre neuronas que se activan simultáneamente. Es útil cuando deseas que la red aprenda patrones específicos o asociaciones entre entradas.
        - Covarianza: Esta regla modifica las conexiones en función de la covarianza entre las activaciones de las neuronas pre y postsinápticas. Puede ser útil para aprender estadísticas de segundo orden de los datos.
        - Oja: Esta regla es una versión normalizada de la regla de Hebb, que puede prevenir el crecimiento excesivo de los pesos. Es útil para extraer componentes principales de los datos.
        
        * Plasticidad Intrínseca:
        
        - Excitabilidad: Modificar la excitabilidad de las neuronas puede afectar la dinámica de la red y su capacidad para procesar diferentes tipos de entrada. Puede aumentar la capacidad de adaptación de la red a diferentes condiciones.
        - Función de Activación: Cambiar las funciones de activación puede alterar la forma en que las neuronas responden a las entradas y cómo se propagan las señales a través de la red. Puede influir en la capacidad de la red para aprender patrones complejos o para adaptarse a diferentes tipos de datos. 
    
    '''

    def __init__ (self, config):
        super(MyESN, self).__init__()
        self.init_type               = config ['init_type']
        self.init_std                = config ['init_std']
        self.init_mean               = config ['init_mean']
        self.connectivity            = config ['connectivity']
        self.input_size              = config ['input_size']
        self.nonlinearity            = config ['nonlinearity']
        self.use_input_bias          = config ['use_input_bias']
        self.use_input_layer         = config ['use_input_layer']
        self.reservoir_size          = config ['n_internal_units']
        self.spectral_radius         = config ['spectral_radius']
        self.input_scaling           = config ['input_scaling']
        self.plasticity_synaptic     = config ['plasticity_synaptic']   # Configuración de la plasticidad sináptica
        self.plasticity_intrinsic    = config ['plasticity_intrinsic']  # Configuración de la plasticidad intrínseca
        self.new_activation_function = config ['new_activation_function'] 
        self.excitability_factor     = config ['excitability_factor']
        self.leak                    = config ['leak']
        self.noise_level             = config ['noise_level']
        self.learning_rate           = config ['learning_rate']
        self.circle                  = config ['circle']
        self.theta_m                 = config ['theta_m']

        self.config = config
        if config ['device'] is not None and (config ['device'] == 'gpu' or config ['device'] == 'cuda'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            

        
        self.new_activations = None # inicializamos función activación en plasticidad
        
        # Generate internal weights y reservoir_size
        self._initialize_weights ()
        
        # Si se usa Capa lineal inicial para los datos de entrada, se crea la capa lineal torch.
        # Notar que si no e usa una capa lineal torch se simulará dicha capa pasando los vectores
        # de entrada a todas las neuronas de la capa interna mediante una multiplicación matricial
        # simulando la combinación lineal de una capa lineal torch
        
        if self.use_input_layer:
            # Inicialización de la capa de entrada solo para los datos de entrada
            self.input_layer = nn.Linear (self.input_size, self.reservoir_size, bias = self.use_input_bias)
            # Asignar los pesos inicializados a la capa lineal
            #self.input_layer.weight.data = self.input_weights
            # Inicializar el bias si es necesario
            #if self.use_input_bias:
                #init.zeros_(self.input_layer.bias)
                
        # Normaliza las matriz de pesos del reservorio (neuronas internas) con radio espectral
        if self.circle:
            self.reservoir_weights = self._spectral_reservoir_weights_circ (self.reservoir_size, self.spectral_radius)
        else:
            self.reservoir_weights = self._spectral_reservoir_weights ()
       
        # Move to device
        self.to (self.device)
        
    def forward (self, input_data, reservoir_state, t):
        ''' 
            Método forward que calcula el estado del reservorio en el instante t basado en la entrada y el estado previo.
            
            Funcionalidad:
            
            1. Transforma la entrada:
               - Si se ha habilitado una capa de entrada lineal, se aplica esta transformación inicial.
               - En caso contrario, se multiplica la entrada por los pesos de entrada.
        
            2. Obtiene el estado previo del reservorio:
               - Si t > 0, se toma el estado del reservorio en el instante anterior (t-1).
               - Si t == 0, se inicializa como un tensor de ceros.
        
            3. Calcula la representación interna del reservorio:
               - Multiplica el estado anterior por los pesos del reservorio, permitiendo la propagación de la información a través del mismo.
               - Suma la representación transformada por la entrada.
               - Añade ruido a la señal para introducir aleatoriedad.
        
            4. Aplica la no linealidad:
               - Utiliza la función tangente hiperbólica para generar la salida no lineal del reservorio.
        
            5. Aplica la plasticidad sináptica e intrínseca si están habilitadas:
               - La plasticidad sináptica ajusta los pesos de las sinapsis.
               - La plasticidad intrínseca ajusta los nodos del reservorio.
        
            6. Actualiza el estado del reservorio:
               - Interpola entre el estado anterior y la representación transformada por la entrada usando la tasa de fuga.
               - Almacena el estado actualizado del reservorio en el instante t.
        
            Parámetros:
            - input_data: Tensor con los datos de entrada.
            - reservoir_state: Tensor con el estado del reservorio.
            - t: Instante de tiempo actual.
            
            Retorno:
            - El estado actualizado del reservorio.
        '''
        
        # print(f'input_data:{input_data.shape}')
        input_data      = input_data.to(self.device)
        reservoir_state = reservoir_state.to(self.device)
        
        # Aplicar transformación lineal a la entrada si se usa una capa lineal inicial
        # o una combinación lineal con los pesos asociados a cada uno de los nodos internos

        if self.use_input_layer:
            reservoir_input = self.input_layer (input_data)
        else:
            reservoir_input    = torch.matmul (input_data, self.input_weights.to(self.device))

        # obtenemos el estado actual trasnmitiendo al información de entrada a los nodos
        # del RC atendiendo a la matriz de conectividad (reservoir_state)
        current_rc_state = None
        if t > 0:
            current_rc_state = reservoir_state [:, (t-1), :]
        else:
            # current_rc_state = torch.zeros_like (reservoir_input).to(self.device)
            initial_state    = torch.ones_like (reservoir_input).to(self.device)
            current_rc_state = torch.matmul (initial_state, self.reservoir_weights.to (self.device))
        
        # Esta multiplicación por la matriz de conexiones del reservorio no solo transmite
        # la información a los nodos vecinos, sino que también permite que la información
        # se propague a través de múltiples conexiones en el reservorio, generando así
        # dinámicas complejas y no lineales.
        
        # representación interna compleja y no lineal en el reservorio
        recervoire_state_w   = torch.matmul (current_rc_state, self.reservoir_weights.to (self.device))   
        state_before_tanh    = recervoire_state_w + reservoir_input
        # Añadir ruido
        state_before_tanh += self.noise_level * torch.randn_like (state_before_tanh)
        # Aplicar la no linealidad
        reservoir_state_tanh = self._f_nonlinearity (state_before_tanh) 

        # Aplica la plasticidad sináptica si está habilitada
        if self.plasticity_synaptic is not None:
            self._apply_synaptic_plasticity (input_data, reservoir_state, t)

        # Aplica la plasticidad intrínseca si está habilitada
        if self.plasticity_intrinsic is not None:
            self._apply_intrinsic_plasticity (reservoir_state, t)
            
        #print (f'reservoir_state_tanh:{reservoir_state_tanh.shape}')
        #print (f'reservoir_state:{current_rc_state.shape}')
        # Actualizar el estado del reservorio con fuga y ruido. Para mantener la memoria a corto plazo en el sistema.
        reservoir_state [:,t,:] = (1.0 - self.leak)*current_rc_state + self.leak * reservoir_state_tanh


        return reservoir_state

    def _f_nonlinearity(self, rc_data):
        '''
            Aplica una función de no linealidad a los datos del reservorio según la configuración.
            
            Funcionalidad:
            
            - Si se ha definido `new_activations`, aplica esta función a `rc_data`.
            - Si no, aplica la función de no linealidad especificada (`tanh`, `relu` o `id`) a `rc_data`.
            
            Parámetros:
            - rc_data: Tensor de datos del reservorio a los que aplicar la no linealidad.
            
            Retorno:
            - Tensor con los datos del reservorio transformados según la función de no linealidad.
        '''
        if self.new_activations is not None:
            rc_data_nonlinearity = self.new_activations(rc_data)
        else:
            if self.nonlinearity == 'tanh':
                rc_data_nonlinearity = torch.tanh(rc_data)
            elif self.nonlinearity == 'relu':
                rc_data_nonlinearity = torch.relu(rc_data)
            elif self.nonlinearity == 'id':
                rc_data_nonlinearity = rc_data
            else:
                raise ValueError("Invalid nonlinearity. Use 'tanh', 'relu', or 'id'.")
                
        return rc_data_nonlinearity
        
    def _apply_intrinsic_plasticity (self, reservoir_state, t):
        '''
            Aplica reglas de plasticidad intrínseca al estado del reservorio en el instante t.
            
            Funcionalidad:
            
            - Si la plasticidad intrínseca es 'excitability', ajusta la excitabilidad de las neuronas multiplicando por `excitability_factor`.
            - Si la plasticidad intrínseca es 'activation_function', cambia la función de activación del reservorio según `new_activation_function`.
            
            Parámetros:
            - reservoir_state: Tensor con el estado del reservorio.
            - t: Instante de tiempo actual.
        '''        
        
        # Implementación de la plasticidad intrínseca
        if self.plasticity_intrinsic == 'excitability':
            # Modifica la excitabilidad de las neuronas en el reservorio
            # Actualizar la excitabilidad de las neuronas
            reservoir_state[:, t, :] *= self.excitability_factor
        elif self.plasticity_intrinsic== 'activation_function':
            # Modifica las funciones de activación de las neuronas
            if self.new_activation_function == 'tanh':
                self.new_activations = torch.tanh 
            elif self.new_activation_function == 'relu':
                self.new_activations = torch.relu 
            elif self.new_activation_function == 'id':
                self.new_activations = lambda x: x  # Identidad
            else:
                raise ValueError("Invalid new activation function. Only 'tanh', 'relu' or 'id' ")
        else:
            raise ValueError("Invalid intrinsic plasticity rule. Only 'excitability' or 'activation_function' ")

    def _apply_synaptic_plasticity (self, input_data, reservoir_state, t):
        '''
            Aplica reglas de plasticidad sináptica al estado del reservorio en el instante t.
            
            Funcionalidad:
            
            - Si la plasticidad sináptica es 'hebb', aplica la regla de Hebb.
            - Si la plasticidad sináptica es 'covariance', calcula y actualiza los pesos según la regla de Covarianza.
            - Si la plasticidad sináptica es 'oja', aplica la regla de Oja para ajustar los pesos.
            
            Parámetros:
            - input_data: Tensor con los datos de entrada.
            - reservoir_state: Tensor con el estado del reservorio.
            - t: Instante de tiempo actual.
        '''
        # Implementación de la plasticidad sináptica
        if self.plasticity_synaptic is not None:
            if self.plasticity_synaptic == 'bcm':
                # Regla de Hebb
                # Obtener la activación de las neuronas en el reservorio
                self._apply_synaptic_plasticity_bcm (input_data, reservoir_state, t)
            elif self.plasticity_synaptic == 'hebb':
                # Regla de Hebb
                # Obtener la activación de las neuronas en el reservorio
                self._apply_synaptic_plasticity_hebb (input_data, reservoir_state, t)
            elif self.plasticity_synaptic == 'covariance':
                # Regla de Covarianza
                # Obtener la activación de las neuronas en el reservorio
                activations = reservoir_state [:, t, :]
                # Calcular la matriz de covarianza
                covariance_matrix = torch.matmul (activations.t(), activations) / activations.shape[0]
                # Actualizar los pesos con la regla de Covarianza
                self.reservoir_weights += self.learning_rate * covariance_matrix
            elif self.plasticity_synaptic == 'oja':
                # Regla de Oja
                self._apply_ojas_rule (input_data, reservoir_state,  t)
            else:
                raise ValueError("Invalid synaptic plasticity rule.") 

    def _apply_ojas_rule (self, input_data, reservoir_state, t):
        """
            La regla de Oja es una modificación de la regla de Hebb que incluye un término de normalización 
            para evitar que los pesos crezcan sin límites. Implementar la plasticidad sináptica con la regla de Oja 
            en tu sistema de Reservoir Computing puede ayudar a ajustar los pesos del reservorio durante el entrenamiento.
        
            Regla de Oja:
            La regla de Oja se puede formular como:
            Δw = η ⋅ (y ⋅ x - y² ⋅ w)
        
            donde:
                Δw es el cambio en los pesos.
                η es la tasa de aprendizaje.
                y es la salida de la neurona (estado del reservorio después de la no linealidad).
                x es la entrada de la neurona (estado del reservorio antes de la no linealidad).
                w son los pesos sinápticos.
        
            Este método aplica la regla de Oja para actualizar los pesos del reservorio.
        
            Args:
                input_data (torch.Tensor): Los datos de entrada al sistema.
                reservoir_state (torch.Tensor): El estado del reservorio en todos los tiempos.
                t (int): El tiempo actual del entrenamiento.
        """
        # Obtener el estado actual del reservorio en el tiempo t
        current_rc_state = reservoir_state [:, t, :]  # Dimensiones: (batch_size, reservoir_size)
    
        # Calcular la activación de las neuronas en el reservorio en el tiempo t
        # Calcular el cambio en los pesos según la regla de Oja
        delta_w = self.learning_rate * (torch.matmul (current_rc_state.unsqueeze (2), current_rc_state.unsqueeze (1)) - \
                                    torch.pow (current_rc_state.unsqueeze (2), 2) * self.reservoir_weights.unsqueeze (0))

        # Actualizar los pesos sinápticos del reservorio
        self.reservoir_weights += delta_w.mean (dim = 0)
    def _apply_synaptic_plasticity_hebb(self, input_data, reservoir_state, t):
        """
            La regla de Hebb es un principio fundamental de la neurociencia que sugiere que las conexiones entre 
            neuronas se fortalecen cuando ambas neuronas están activas simultáneamente. Implementar la plasticidad 
            sináptica con la regla de Hebb en tu sistema de Reservoir Computing puede ayudar a ajustar los pesos 
            del reservorio durante el entrenamiento.
        
            Regla de Hebb:
            La regla de Hebb se puede formular como:
            Δw = η ⋅ y ⋅ x
        
            donde:
                Δw es el cambio en los pesos.
                η es la tasa de aprendizaje.
                y es la salida de la neurona (estado del reservorio después de la no linealidad).
                x es la entrada de la neurona (estado del reservorio antes de la no linealidad).
        
            Este método aplica la regla de Hebb para actualizar los pesos del reservorio.
        
            Args:
                input_data (torch.Tensor): Los datos de entrada al sistema.
                reservoir_state (torch.Tensor): El estado del reservorio en todos los tiempos.
                t (int): El tiempo actual del entrenamiento.
        """
        # Obtener el estado actual
        current_rc_state = reservoir_state[:, t, :]
    
        # Calcular el cambio en los pesos según la regla de Hebb
        delta_w = self.learning_rate * torch.matmul(current_rc_state.unsqueeze(2), current_rc_state.unsqueeze(1))
    
        # Actualizar los pesos del reservorio
        self.reservoir_weights += delta_w.mean(dim=0)
    def _apply_synaptic_plasticity_bcm (self, input_data, reservoir_state, t):
        """
            La regla BCM (Bienenstock, Cooper y Munro) es una extensión de la regla de Hebb que introduce un 
            umbral de plasticidad dependiente de la actividad promedio de la neurona. Esta regla puede ser 
            más estable y capaz de captar dependencias temporales en los datos.
        
            Regla BCM:
            La regla BCM se puede formular como:
            Δw = η ⋅ (avg_activation ⋅ avg_activation - θ_m ⋅ w²)
        
            donde:
                Δw es el cambio en los pesos.
                η es la tasa de aprendizaje.
                avg_activation es la activación promedio de la neurona.
                θ_m es el umbral de modificación.
                w son los pesos sinápticos.
        
            Este método aplica la regla BCM para actualizar los pesos del reservorio.
        
            Args:
                input_data (torch.Tensor): Los datos de entrada al sistema.
                reservoir_state (torch.Tensor): El estado del reservorio en todos los tiempos.
                t (int): El tiempo actual del entrenamiento.
        """
        # Obtener el estado actual del reservorio en el tiempo t
        reservoir_state_t = reservoir_state[:, t, :]
    
        # Calcular la activación promedio de las neuronas en el reservorio en el tiempo t
        avg_activation = reservoir_state_t.mean(dim=0)
    
        # Calcular el cambio en los pesos sinápticos utilizando la regla BCM
        delta_weights = self.learning_rate * (torch.outer(avg_activation, avg_activation) - self.theta_m * torch.pow(self.reservoir_weights, 2))
    
        # Actualizar los pesos sinápticos del reservorio
        self.reservoir_weights += delta_weights.to(self.device)
    
        # Opcionalmente, actualizar el umbral de modificación
        self.theta_m = reservoir_state_t.norm(dim=1).mean()


    def _spectral_reservoir_weights_circ (self, n_internal_units, spectral_radius):
        '''
            Construye una matriz de pesos del reservorio con topología circular y ajusta el radio espectral.
            
            Funcionalidad:
            
            - Construye una matriz de pesos del reservorio con topología circular.
            - Ajusta el radio espectral de la matriz de pesos del reservorio.
            
            Parámetros:
            - n_internal_units: Número de unidades internas en el reservorio.
            - spectral_radius: Radio espectral deseado para la matriz de pesos del reservorio.
            
            Retorno:
            - Tensor con la matriz de pesos del reservorio ajustada.
        '''
        # Construct reservoir with circular topology
        reservoir_weights = np.zeros((n_internal_units, n_internal_units))
        reservoir_weights [0,-1] = 1.0
        for i in range(n_internal_units-1):
            reservoir_weights [i+1,i] = 1.0

        # Adjust the spectral radius.
        E, _ = np.linalg.eig (reservoir_weights)
        e_max = np.max(np.abs(E))
        reservoir_weights /= np.abs(e_max)/spectral_radius 
        return torch.tensor (reservoir_weights, dtype = torch.float32).to (self.device)

        
    def _spectral_reservoir_weights (self):
        '''
            Ajusta el radio espectral de la matriz de pesos del reservorio y aplica conectividad esparsa.
            
            Funcionalidad:
            
            - Ajusta el radio espectral de la matriz de pesos del reservorio.
            - Aplica una máscara de conectividad esparsa basada en el parámetro `connectivity`.
            
            Retorno:
            - Tensor con la matriz de pesos del reservorio ajustada y con conectividad esparsa.
        '''
        # print (f'reservoir_weights 0:{self.reservoir_weights.device}')
        
        # print (f'reservoir_weights 1:{self.reservoir_weights.device}')
        if not isinstance (self.reservoir_weights, torch.Tensor):
            self.reservoir_weights = torch.tensor (self.reservoir_weights, dtype = torch.float32).to (self.device)
        self.reservoir_weights = self.reservoir_weights.to(self.device)
        # print (f'reservoir_weights 2:{self.reservoir_weights.device}')
        self.reservoir_weights *= self.spectral_radius / torch.max (torch.abs (torch.linalg.eigvals (self.reservoir_weights)))
        # print (f'reservoir_weights 3:{self.reservoir_weights.device}')
        mask = torch.rand (self.reservoir_size, self.reservoir_size).float()  < self.connectivity
        # print (f'mask 0:{mask.device}')
        # print (f'reservoir_weights 4:{self.reservoir_weights.device}')
        mask = mask.to(self.device) # Move mask to the same device
        # print (f'mask 1:{mask.device}')
        self.reservoir_weights *= mask 
        return self.reservoir_weights
        
    def _initialize_weights (self):
        '''
            Inicializa los pesos de entrada y del reservorio según el tipo de inicialización especificado.
            
            Funcionalidad:
            
                - Inicializa los pesos de entrada (`input_weights`) y del reservorio (`reservoir_weights`).
                - Permite la inicialización aleatoria uniforme, ortogonal, truncada normal o binomial.
                
                Tipos de inicialización:
                
                - 'rand': Inicialización aleatoria uniforme. Los pesos se generan aleatoriamente entre -0.5 y 0.5 y se escalan por `input_scaling`.
                
                - 'orthogonal': Inicialización ortogonal. Los pesos se inicializan con una matriz ortogonal generada por PyTorch y se escalan por 0.5 y `input_scaling`.
                
                - 'trunc_normal': Inicialización truncada normal. Los pesos se generan aleatoriamente a partir de una distribución normal truncada con media `init_mean`, desviación estándar `init_std`, y luego se escalan por `input_scaling`.
                
                - 'binorm': Inicialización binomial. Los pesos se generan utilizando una distribución binomial centrada en 0 y 1, luego se escalan entre -1.0 y 1.0 por `input_scaling`.
                
                Raises:
                    ValueError: Si el tipo de inicialización especificado no es válido (no es 'rand', 'orthogonal', 'trunc_normal' o 'binorm').
        '''
        # Define las matrices de pesos para la entrada y del reservorio
        # Inicialización de los pesos de entrada
        if self.init_type == 'rand':
            self.input_weights     = (torch.rand (self.input_size, self.reservoir_size) - 0.5) * self.input_scaling
            self.reservoir_weights = (torch.rand (self.reservoir_size, self.reservoir_size) - 0.5)
        elif self.init_type == 'orthogonal':
            self.input_weights     = (init.orthogonal_ (torch.empty (self.input_size, self.reservoir_size))*0.5) * self.input_scaling
            self.reservoir_weights = (init.orthogonal_ (torch.empty (self.reservoir_size, self.reservoir_size))*0.5)
        elif self.init_type == 'trunc_normal':
            self.input_weights = torch.empty (self.input_size, self.reservoir_size) * self.input_scaling
            init.trunc_normal_ (self.input_weights, mean = self.init_mean, std = self.init_std)
            self.reservoir_weights = torch.empty (self.reservoir_size, self.reservoir_size)
            init.trunc_normal_ (self.reservoir_weights, mean = self.init_mean, std = self.init_std)
        elif self.init_type == 'binorm':
            self.input_weights = torch.tensor((2.0 * np.random.binomial(1, 0.5, [self.input_size, self.reservoir_size]) - 1.0) * self.input_scaling, dtype=torch.float32)
            self.reservoir_weights = torch.tensor((2.0 * np.random.binomial(1, 0.5, [self.reservoir_size, self.reservoir_size]) - 1.0), dtype=torch.float32)

            print(f'{type(self.reservoir_weights)}') 
        else:
            raise ValueError("Invalid initialization type. Use 'rand', 'orthogonal', or 'trunc_normal'.")
            
    def plot_spectral_graph (self):
        '''
                función auxiliar para graficar la representación interna de conexiones del RC.
        '''
        # print (f' max:{torch.max(self.reservoir_weights)} - min :{torch.min(self.reservoir_weights)}')

        # Calcular los eigenvalores
        eigenvalues = torch.linalg.eigvals (self.reservoir_weights)
        # Separar la parte real e imaginaria de los eigenvalores
        real_part      = eigenvalues.real
        imaginary_part = eigenvalues.imag
        # Graficar el gráfico espectral
        plt.scatter (real_part, imaginary_part, s=5)
        plt.xlabel ('Parte Real')
        plt.ylabel ('Parte Imaginaria')
        plt.title ('Gráfico Espectral de Conexiones')
        plt.grid (True)
        plt.show ()
##################################   
class MyRC:
    model           = None  # Arquitectura y sistema ESN para mi RC
    config          = None  # Mantiene lso hiperparámetros del RC
    readout         = None  # Almacena la ultima fase del RC 
    output_rc       = None  # Salida del RC en el último transitorio
    readout_type    = None  # Tipo de capa de salida deseada para el RC (por configuración)
    redim_state     = None  # 
    reservoir_state = None  # Estado del RC en el último transitorio 
    ridge_embedding = None  # Regresión ridge 
    output_rc_layer = None  # Almacenará el resultado de la fase de readout
    mtx_rc_state    = None  # Almacenará los trnasitorios que deseamos que almacene, según la configuración del RC
    
    def __init__ (self, model,  config): #readout,
        self.model                  = model
        self.readout                = None # readout
        self.config                 = config
        self.readout_type           = self.config ['readout_type']
        self.mts_rep                = self.config ['mts_rep']
        self.w_ridge_embedding      = self.config ['w_ridge_embedding']
        self.w_ridge                = self.config ['w_ridge']
        self.w_l2                   = self.config ['w_l2']
        self.svm_C                  = self.config ['svm_C']
        self.svm_kernel             = self.config ['svm_kernel'] 
        self.svm_gamma              = self.config ['svm_gamma']
        self.mlp_layout             = self.config ['mlp_layout']
        self.mlp_batch_size         = self.config ['mlp_batch_size']
        self.mlp_learning_rate      = self.config ['mlp_learning_rate']
        self.mlp_learning_rate_type = self.config ['mlp_learning_rate_type']
        self.num_epochs             = self.config ['num_epochs']
        self.nonlinearity           = self.config ['nonlinearity']
        self.mts_rep                = self.config ['mts_rep']
        self.dimred_method          = self.config ['dimred_method']
        self.n_dim                  = self.config ['n_dim']
        self.bidir                  = self.config ['bidir']
        self.threshold              = self.config ['threshold']
        self.washout                = self.config ['washout']
        nc_drop                     = self.config ['n_drop']
        
        # Determinar el número de muestras a eliminar
        if nc_drop is None:
            self.n_drop = 0
        elif nc_drop == 'all':
            self.n_drop = -1
        else:
            self.n_drop = int (nc_drop)
            
        # Initialize readout type
        if self.readout_type is not None:
            if self.readout_type == 'lin': # Ridge regression
                self.readout = Ridge (alpha = self.w_ridge, random_state = 0)
            elif self.readout_type == 'svm': # SVM readout
                self.readout = SVC (C=self.svm_C, kernel='precomputed')
            elif self.readout_type == 'ovr': # SVM readout
                svm_classifier = SVC (kernel = self.svm_kernel, C=self.svm_C,gamma = self.svm_gamma, probability=True)
                # Utilizar OneVsRestClassifier con el clasificador SVM
                self.readout = OneVsRestClassifier (svm_classifier)
            elif self.readout_type == 'mlp': # MLP (deep readout)
                # pass
                self.readout = MLPClassifier(
                    hidden_layer_sizes  = self.mlp_layout,
                    activation          = self.nonlinearity,
                    alpha               = self.w_l2,
                    batch_size          = self.mlp_batch_size,
                    learning_rate       = self.mlp_learning_rate_type, # 'constant' or 'adaptive'
                    learning_rate_init  = self.mlp_learning_rate,
                    max_iter            = self.num_epochs,
                    early_stopping      = True, # if True, set validation_fraction > 0
                    validation_fraction = 0.001 # used for early stopping
                    )
            else:
                raise RuntimeError('Invalid readout type. Only (lin, svm, ovr or mlp')
                
        # print (f'* self.mts_rep : {self.mts_rep}')
        
        # Initialize ridge regression model
        if self.mts_rep == 'output' or self.mts_rep == 'reservoir':
            self._ridge_embedding = Ridge (alpha = self.w_ridge_embedding, fit_intercept = True, random_state = 0)
            
        # Initialize dimensionality reduction method
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components = self.n_dim)            
            elif self.dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components = self.n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID') 
                
    def _process_transient_ESN (self, X, transients_to_drop = None):
        """
            Procesa los transitorios de una Red de Estado Reservado (ESN).
        
            Args:
                - X: Datos de entrada que contienen las señales de la ESN.
                - transients_to_drop (lista o None): Lista de índices de los transitorios a eliminar.
        
            Returns:
                - output  : Representación del último transitorio de la ESN.
                - response: Estado de la ESN con todos los transitorios.
        """
        n_serial = X.size (1)  # número de instantes de la serie temporal
        # print (f'transients_to_drop:{transients_to_drop}')
        # Calcular el número de transitorios a eliminar
        num_transients_to_drop = len(transients_to_drop) if transients_to_drop is not None else 0
        # print (f'num_transients_to_drop:{num_transients_to_drop}')
        # print (f'num_transients_to_drop:{num_transients_to_drop}') 
        # Calcular el tamaño de reservoir_state considerando los transitorios a eliminar
        reservoir_state = torch.zeros((X.size(0), n_serial  , self.model.reservoir_size), dtype= torch.float32)
        rc_state = torch.zeros((X.size(0), n_serial - num_transients_to_drop, self.model.reservoir_size), dtype= torch.float32)
        # print (f'rc_state:{rc_state.shape}')
        # print (f'reservoir_state:{reservoir_state.size()}')
        # print (f'rc_state:{rc_state.size()}')
        reservoir_index = 0  # Índice para seguir el estado del reservorio
        # print (f'transients_to_drop:{transients_to_drop}')
        for t in range (n_serial):
            input_sequence = X[:, t, :].float()
            # print (f't:{t}')
            
            if transients_to_drop is not None and t in transients_to_drop:
                continue # Skip processing this transient
            
            # Process input_sequence with the ESN model
            reservoir_state = self.model (input_sequence, reservoir_state, t)
            # print (f'* reservoir_state:{reservoir_state.size()}')
            
            rc_state [:,reservoir_index,:] = reservoir_state [:,t,:]
            reservoir_index += 1
            
            # print (f'* reservoir_index:{reservoir_index}')
            
        if len(rc_state) == 0:
            return None
        # print (f'* rc_state:{rc_state.shape}') 
        return rc_state    
    def _determine_transients_to_drop (self, n_drop, input_timeline_size):
        """
            Determina los índices de los transitorios a eliminar según la configuración.
        
            Args:
            - n_drop (int): Número de transitorios a eliminar.
            - input_timeline_size (int): Tamaño total de la línea temporal de entrada.
        
            Returns:
            - transients_to_drop (list): Lista de índices de transitorios a eliminar.
        """
        transients_to_drop = []
    
        if n_drop > 0:
            if self.washout == 'init':
                transients_to_drop = list (range (n_drop))
            elif self.washout == 'rand':
                transients_to_drop = np.random.choice (input_timeline_size, size = n_drop, replace = False)
            else:
                transients_to_drop = list (range (n_drop))
        elif n_drop < 0:
            # Eliminar todos menos el último transitorio
            transients_to_drop = list (range (input_timeline_size - 1))
        else:
            # No eliminar ningún transitorio
            transients_to_drop = None
    
        return transients_to_drop
        
    def _get_states (self, input_data, birdir = False, evaluate = False):
        """
            Obtener los estados de la Red de Estado de Eco (ESN) a partir de señales EEG de cada canal.
            
            Args:
            - input_data: Datos de entrada que contienen las señales EEG de cada canal.
            - bidir: Bandera que indica si procesar los datos en dirección bidireccional (por defecto: False).
            - evaluate: Bandera que indica si estamos en fase de evaluación (por defecto: False).
            
            Returns:
            - out_rc: El último transitorio (último estado de ESN).
            - mts_rep: Representación de todos los transitorios.
            - all_rc: Todos los transitorios excepto los especificados para eliminación según la configuración.
        """
        # Convertir listas a tensores
        if not isinstance (input_data, torch.Tensor):
            input_data   = torch.tensor(input_data, dtype = torch.float32)
     
        # ============ Obtenemos los indies de los trnasitorios a eliminar ==============
        n_drop             = 0
        transients_to_drop = None
        if not evaluate: # En fase de test o evaluación no se eliminan transitorios
            n_drop = min (self.n_drop, input_data.size(1) - 1)
            transients_to_drop = self._determine_transients_to_drop (n_drop, input_data.size (1)) 
        # ============ Buscamos el estado del RC al procesar la series tmeporales ============
        reservoir_state  = self._process_transient_ESN (input_data, transients_to_drop)
        
        # print (f'* get_states : reservoir_state : {reservoir_state.shape}')
        # print (f'* get_states : birdir : {birdir}')
        # ============ Si activamos una estado bidireccional del estado interno del ESN ============
        if birdir:
            input_data_b       = torch.tensor(input_data.numpy () [:, ::-1, :].copy(), dtype = torch.float32)
            reservoir_state_b  = self._process_transient_ESN (input_data_b, transients_to_drop)
            
            print (f'* get_states : reservoir_state_b : {reservoir_state_b.shape}')
            all_rc = np.concatenate((reservoir_state, reservoir_state_b), axis=1)
        else:
            all_rc = reservoir_state

        # print ('Procesamiento ESN completado.')
        # print (f'* get_states : all_rc : {all_rc.shape}')

        return  all_rc, transients_to_drop
        
    def _apply_dimensionality_reduction (self, reservoir_state):
        """
            Aplica reducción de dimensionalidad al estado del reservorio según el método especificado.
            
            Args:
            - reservoir_state (numpy.ndarray): Estado del reservorio a reducir dimensionalidad.
            
            Returns:
            - red_states (numpy.ndarray): Estado del reservorio con reducción de dimensionalidad aplicada, si corresponde.
        """
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # matricize
                n_samples = reservoir_state.shape [0]
                r_states  = reservoir_state.reshape (-1, reservoir_state.shape [2])                   
                # ..transform..
                red_states = self._dim_red.fit_transform (r_states)          
                # ..and put back in tensor form
                red_states = red_states.reshape (n_samples, -1, red_states.shape [1])          
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform (reservoir_state)       
            else: # Skip dimensionality reduction
                red_states = reservoir_state
        else: # Skip dimensionality reduction
                red_states = reservoir_state
                
        red_states = np.real (red_states) # Occasionally I get complex matrices that the rest of the code does not support
        return red_states

    def _compute_output_represtn (self, red_states, input_data, n_tr_drop):
        """
            Calcula la representación interna de salida basada en los estados reducidos del reservorio y los datos de entrada.
            
            Args:
            - red_states (numpy.ndarray): Los estados reducidos del reservorio.
            - input_data (numpy.ndarray or torch.Tensor): Los datos de entrada.
            - n_tr_drop (list): Lista de índices de transitorios a eliminar.
            
            Returns:
            - numpy.ndarray: La representación interna de salida.
        """
        # Eliminar las mismas columnas de red_states que se eliminaron de input_data
        if n_tr_drop is not None:
            input_data = np.delete (input_data, n_tr_drop, axis = 1)

        # Si es bidireccional, concatenar los datos de entrada con su versión invertida
        if self.bidir:
            if torch.is_tensor (input_data):
                reversed_input = torch.flip (input_data, dims=[1])
                input_data = torch.cat((input_data, reversed_input), dim=1)
            else:
                input_data = np.concatenate((input_data, input_data[:, ::-1, :]), axis=1)

        coeff_tr, biases_tr = [], []
    
        # Ajuste de _ridge_embedding para cada ejemplo en input_data
        for i in range(input_data.shape[0]):
            self._ridge_embedding.fit(red_states[i, :-1, :], input_data[i, :-1, :])
            coeff_tr.append(self._ridge_embedding.coef_.ravel())
            biases_tr.append(self._ridge_embedding.intercept_.ravel())
    
        # Concatenar coeficientes y sesgos
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr
            
    def _compute_reservoir_represtn (self, red_states, input_data, n_tr_drop):
        """
            Calcula la representación del reservorio basada en los estados de reservorio reducidos y los datos de entrada.
            
            Args:
            - red_states (numpy.ndarray): Los estados de reservorio reducidos.
            - input_data (numpy.ndarray): Los datos de entrada.
            - n_tr_drop (list): Lista de índices de transitorios a eliminar.
            
            Returns:
            - numpy.ndarray: La salida de representación del reservorio.
        """
        coeff_tr  = []
        biases_tr = []
        
        # Eliminar las mismas columnas de red_states que se eliminaron de input_data
        if n_tr_drop is not None:
            input_data = np.delete (input_data, n_tr_drop, axis = 1)
        
        for i in range(input_data.shape[0]):
            # print (f'red_states:{red_states [i].shape}')
            self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeff_tr.append(self._ridge_embedding.coef_.ravel())
            biases_tr.append(self._ridge_embedding.intercept_.ravel())
        
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
    
        return input_repr
        
    def _compute_readout_layer (self, input_repr, target_data = None):
        """
            Calcula la capa de salida basada en la representación interna y los datos objetivo.
            
            Args:
            - input_repr (numpy.ndarray): La representación interna.
            - target_data (numpy.ndarray): Los datos objetivo para entrenar la capa de salida. Por defecto es None.
            
            Returns:
            - object: La capa de salida entrenada.
        """
        output_rc_layer = None
        if target_data is None or self.readout_type is None:
            # Just store the input all representations (all RC state)
            output_rc_layer = input_repr
        elif self.readout_type == 'lin':
            # Ridge regression
            output_rc_layer = self.readout.fit (input_repr, target_data)
        elif self.readout_type == 'svm':
            # SVM readout
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean'))
            Ktr = np.exp(-self.svm_gamma * Ktr)
            output_rc_layer = self.readout.fit (Ktr, np.argmax(target_data, axis=1))
        elif self.readout_type == 'ovr':
            # SVM ovr readout
            self.readout.fit (input_repr, target_data)
        elif self.readout_type == 'mlp':
            # MLP (deep readout)
            output_rc_layer = self.readout.fit(input_repr, target_data)
     
        return output_rc_layer
    
    def _convert_to_one_hot (self, pred_class_max, num_classes = 2):
        """
            Convierte las predicciones de clases maximizadas en codificación one-hot.
    
            Args:
            - pred_class_max (numpy.ndarray): Predicciones de clases maximizadas (números enteros).
            - num_classes (int): Número total de clases. Por defecto es 2.
    
            Returns:
            - numpy.ndarray: Matriz codificada en one-hot.
        """
        num_samples = len(pred_class_max)
        one_hot = np.zeros((num_samples, num_classes), dtype=np.int32)

        for i in range(num_samples):
            if pred_class_max[i] == 0:
                one_hot[i, 0] = 1  # Clase 0: 10
            elif pred_class_max[i] == 1:
                one_hot[i, 1] = 1  # Clase 1: 01

        return one_hot
    def _compute_multilabel_test_scores(self, y_pred, Yte, multi_label=True):
        """
            Calcula las métricas de evaluación multilabel para la predicción y los datos de prueba.
        
            Args:
                - y_pred (numpy.ndarray): Las predicciones del modelo.
                - Yte (numpy.ndarray): Las etiquetas verdaderas de los datos de prueba.
                - multi_label (bool): Indica si las etiquetas son multilabel o no. Por defecto es True.
        
            Returns:
                - float: F1-Score ponderado o macro/micro según el promedio especificado.
                - float: Precisión del modelo.
                - numpy.ndarray: Matriz de confusión multilabel.
        """
        # Comprobar si son multilabel
        if Yte.shape[1] > 2:
            true_class = np.argmax(Yte, axis=1)
            accuracy = accuracy_score(true_class, y_pred, multi_label=multi_label)
            f1 = f1_score(true_class, y_pred, average='weighted')
        else:
            true_class = Yte[:, 0]  # Asignar las etiquetas binarias
            f1 = f1_score(true_class, y_pred, average='binary')
            accuracy = accuracy_score(true_class, y_pred)
    
        # Calcular la matriz de confusión multilabel
        confusion_matrices = multilabel_confusion_matrix(Yte, y_pred)
    
        # Imprimir resultados
        print("Matriz de Confusión Multilabel:")
        print(confusion_matrices)
        print("F1-Score Multilabel:", f1)
        print("Accuracy-Score Multilabel:", accuracy)
    
        return f1, accuracy, confusion_matrices
    def _evaluate_readout(self, labels, labels_pred, multi_label=True):
        """
        Calcula y muestra varias métricas de evaluación para el clustering, incluyendo la matriz de confusión,
        frecuencia, precisión, recall, F1 score, ROC AUC score y la curva ROC. También muestra el reporte de clasificación.
    
        Args:
            labels (array-like): Etiquetas reales.
            labels_pred (array-like): Etiquetas predichas por el modelo de clustering.
            multi_label (bool): Indica si las métricas deben ser calculadas en un contexto multietiqueta.
        Returns:
            - report: Resumen de las métricas (Accuracy, f1, recall..)
        """
        
        if multi_label:
            # Calcular la matriz de confusión multietiqueta
            cm = multilabel_confusion_matrix(labels, labels_pred)
            print("Multilabel Confusion Matrix:")
            print(cm)
    
            # Graficar la matriz de confusión para cada etiqueta
            for i, conf_matrix in enumerate(cm):
                plt.figure(figsize=(10, 7))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix for label {i}')
                plt.show()
    
            # Calcular otras métricas multietiqueta
            accuracy = accuracy_score(labels, labels_pred)
            precision = precision_score(labels, labels_pred, average='weighted')
            recall = recall_score(labels, labels_pred, average='weighted')
            f1 = f1_score(labels, labels_pred, average='weighted')
            roc_auc = roc_auc_score(labels, labels_pred, average='weighted')
        else:
            # Calcular la matriz de confusión normal
            cm = confusion_matrix(labels, labels_pred)
            print("Confusion Matrix:")
            print(cm)
    
            # Graficar la matriz de confusión
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(labels), yticklabels=np.unique(labels))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
    
            # Calcular otras métricas
            accuracy = accuracy_score(labels, labels_pred)
            precision = precision_score(labels, labels_pred, average='weighted')
            recall = recall_score(labels, labels_pred, average='weighted')
            f1 = f1_score(labels, labels_pred, average='weighted')
            roc_auc = roc_auc_score(labels, labels_pred)
    
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)
    
        if not multi_label:
            # Calcular la curva ROC solo si no es multietiqueta
            fpr, tpr, _ = roc_curve(labels, labels_pred)
    
            # Graficar la curva ROC
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
    
        # Reporte de clasificación
        report = classification_report(labels, labels_pred)
        print("Classification Report:")
        print(report)
    
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "classification_report": report
        }
  
    def fit (self, input_data, target_data = None):
        '''
            Entrena el modelo ESN con los datos de entrada y opcionalmente con los datos de destino.

            Argumentos:
                input_data: numpy.ndarray
                    Los datos de entrada para entrenar el ESN.
                target_data: numpy.ndarray, opcional
                    Los datos de destino para entrenar el ESN. Si no se proporcionan, se usan solo los datos de entrada.

            Devoluciones:
                output_rc_layer: numpy.ndarray
                    La capa de salida del ESN.
                reservoir_state: numpy.ndarray
                    El estado del reservorio del ESN.
                mtx_rc_state: numpy.ndarray
                    Los transitorios de la matriz de estados del reservorio.
                red_states: numpy.ndarray
                    Los estados del reservorio reducidos dimensionalmente.
        '''
        representations = []
         #print (f'fit :  input_data: {input_data.shape}')

        with torch.no_grad ():
            mts_rep_state, n_tr_drop = self._get_states (input_data, self.bidir, evaluate = False)

        rc_state = mts_rep_state  # Almacenamos aquellos transitorios que indicamos en configuración
        
        # print (f'fit :  self.mts_rep_state: {self.mts_rep_state.shape}')

        # ============ Dimensionality reduction of the reservoir states ============  
        
        # print (f'fit :  self.dimred_method: {self.dimred_method}')
        if self.dimred_method is not None:
            rc_dim_states = self._apply_dimensionality_reduction (rc_state)
        else:
            rc_dim_states = rc_state
        #print (f'fit : self.rc_dim_states:{self.rc_dim_states.shape}') 
        
        # ============ Generate representation of the MTS ============
        if self.mts_rep == 'output':
            input_repr = self._compute_output_represtn (rc_dim_states , input_data, n_tr_drop)  
        elif self.mts_rep == 'reservoir':
            input_repr = self._compute_reservoir_represtn (rc_dim_states, input_data, n_tr_drop) 
        elif self.mts_rep == 'last':
            input_repr = rc_state [:, -1, :]
        elif self.mts_rep == 'mean':
            if isinstance(rc_state, torch.Tensor):
                input_repr = torch.mean (rc_state, dim = 1)
            else:
                input_repr = np.mean (rc_state, axis = 1)
        elif self.mts_rep == 'id':
            if isinstance(rc_dim_states, torch.Tensor):
                input_repr = rc_dim_states.view (rc_dim_states.size (0), -1)
            else:
                input_repr = rc_dim_states.reshape (rc_dim_states.shape [0], -1)
        elif self.mts_rep == 'state':
            if isinstance(rc_state, torch.Tensor):
                input_repr = rc_state.view (rc_state.size (0), -1)
            else:
                input_repr = rc_state.reshape (rc_state.shape [0], -1)  
        else:
            raise RuntimeError('Invalid representation ID: output, reservoir, state, id, last o mean')
       # print (f'fit : input_repr:{input_repr.shape}')
        self.input_repr_tr = input_repr
        # ============ Apply readout ============
        if self.readout_type is not None:
            output_redout_layer = self._compute_readout_layer (input_repr, target_data)
        else:
            output_redout_layer = input_repr
        
        # Devolvemos: Estados internos neuronas, Etados internos reducidos a PCA, Representación estados internos, Readout
        return  rc_state, rc_dim_states, input_repr, output_redout_layer

    def fit_evaluate (self, Xte, Yte):
        '''
            Evalúa el modelo ESN utilizando datos de prueba y calcula la precisión y la puntuación F1.

            Argumentos:
                Xte: numpy.ndarray
                    Datos de entrada de prueba.
                Yte: numpy.ndarray

            Devoluciones:
                accuracy: float
                    Precisión del modelo.
                f1: float
                    Puntuación F1 del modelo.
                pred_class: numpy.ndarray
                    Clases predichas por el modelo.
        '''
        print (f'fit_evaluate :Yte: {Yte}') 
        # Nota: En fase de evaluación no se eliminan transitorios
        # ============ Compute reservoir states ============
        with torch.no_grad ():
            mts_rep_state_xte, n_tr_drop = self._get_states (Xte, self.bidir, evaluate = True)

        print (f'fit_evaluate :mts_rep_state_xte: {mts_rep_state_xte.shape}')    
        print (f'fit_evaluate :n_tr_drop: {n_tr_drop}')  

        # ============ Dimensionality reduction of the reservoir states   ============
        if self.dimred_method is not None:
            rc_dim_states_xte = self._apply_dimensionality_reduction (mts_rep_state_xte)
        else:
            rc_dim_states_xte = mts_rep_state_xte
            
        # ============ Generate representation of the MTS ============
        if self.mts_rep == 'output':
            input_repr_xte = self._compute_output_represtn (rc_dim_states_xte, Xte, n_tr_drop)  
        elif self.mts_rep == 'reservoir':
            input_repr_xte = self._compute_reservoir_represtn (rc_dim_states_xte, Xte, n_tr_drop) 
        elif self.mts_rep == 'last':
            input_repr_xte = mts_rep_state_xte [:, -1, :]
        elif self.mts_rep == 'mean':
            if isinstance(mts_rep_state_xte, torch.Tensor):
                input_repr_xte = torch.mean (mts_rep_state_xte, dim = 1)
            else:
                input_repr_xte = np.mean (mts_rep_state_xte, axis = 1)
        elif self.mts_rep == 'id':
            if isinstance(rc_dim_states, torch.Tensor):
                input_repr_xte = rc_dim_states_xte.view (rc_dim_states_xte.size (0), -1)
            else:
                input_repr_xte = rc_dim_states_xte.reshape (rc_dim_states_xte.shape [0], -1)
        elif self.mts_rep == 'state':
            if isinstance(rc_dim_states_xte, torch.Tensor):
                input_repr_xte = rc_dim_states_xte.view (rc_dim_states_xte.size (0), -1)
            else:
                input_repr_xte = rc_dim_states_xte.reshape (rc_dim_states_xte.shape [0], -1) 
        else:
            raise RuntimeError('Invalid representation ID: output, reservoir, last o mean') 
            
        # ============ Apply readout ============
        print (f'fit_evaluate :lin :fit_evaluate :self.readout_type : {self.readout_type }')
        Y_test_int = Yte.astype (int)
        pred_class = None
        if self.readout_type == 'lin':  # Linear regression
            logits = self.readout.predict (input_repr_xte)
            pred_prob      = 1 / (1 + np.exp(-logits))
            pred_class_max = np.argmax (logits, axis=1)
            pred_class     = self._convert_to_one_hot (pred_class_max)
             
            print(f'fit_evaluate :lin :logits : {logits}')
            print(f'fit_evaluate :lin :pred_class : {pred_class}')
            print(f'fit_evaluate :lin :pred_prob : {pred_prob}')
            print(f'fit_evaluate :lin :pred_class_max : {pred_class_max}')
        elif self.readout_type == 'svm':  # SVM readout
            Kte = cdist (input_repr_xte, self.input_repr_tr, metric = 'sqeuclidean')
            Kte = np.exp (-self.svm_gamma * Kte)
            
            pred_class            = self.readout.predict (Kte)
            pred_class_multilabel = np.zeros_like (Y_test_int)
            
            for i, label in enumerate(pred_class):
                pred_class_multilabel [i, label] = 1
                
            pred_class = pred_class_multilabel
            
            print(f'fit_evaluate _SVM:Yte : {Y_test_int}')
            print(f'fit_evaluate _SVM:pred_class : {pred_class}')
        
        elif self.readout_type == 'ovr':  # One-vs-Rest Classifier
            pred_prob = self.readout.predict_proba (input_repr_xte)
            print(f'fit_evaluate :ovr :pred_prob : {pred_prob}')
            pred_class = (pred_prob > self.threshold).astype(int)
            
            print(f'fit_evaluate :Yte : {Y_test_int}')
            print(f'fit_evaluate :pred_class : {pred_class}')
        
        elif self.readout_type == 'mlp':  # MLP (deep readout)
            pred_prob      = self.readout.predict_proba (input_repr_xte)
            pred_class_max = np.argmax (pred_prob, axis = 1)
            pred_class     = self._convert_to_one_hot (pred_class_max)
            
            print(f'fit_evaluate :mlp :pred_prob : {pred_prob}')
            print(f'fit_evaluate :mlp :pred_class_max : {pred_class_max}')
            print(f'fit_evaluate :mlp :pred_class : {pred_class}')
        else:
            print("Error while evaluating. When evaluating, we need to have a correct readout (lin, svm, ovr, or mlp).")
            return None, None, None

        if pred_class is not None:
            print(f'fit_evaluate :Yte : {Y_test_int}')
            print(f'fit_evaluate :pred_class : {pred_class}')
            
            # f1, accuracy, confusion_matrices = self._compute_multilabel_test_scores (pred_class, Y_test_int)
            # return pred_class, f1, accuracy, confusion_matrices
            # {
             #"accuracy": accuracy,
             #"precision": precision,
             #"recall": recall,
             #"f1": f1,
             #"roc_auc": roc_auc,
             #"classification_report": report
             # }
            metrics  = self._evaluate_readout (Y_test_int, pred_class)
            return metrics ["classification_report"]
        else:
            # return None, None, None, None
            return None
            
    def generate_representation (self, val_data):
        '''
            Calcular la representación utilizando los estados del reservorio
        '''
        #  Calcular la representación utilizando los estados del reservorio
        val_repr = np.dot (val_data, self.reservoir_state)  # Ejemplo de representación básica: producto punto entre los datos de validación y los estados del reservorio

        return val_repr


    def train_validate_predict (self, train_data, val_data, train_states):
        # Predicción utilizando el estado interno del RC ya entrenado
        # (Esta funcion se utiliza una vez se haya entrenado el modelo y guardado los estados del reservorio)
        train_states = self.reservoir_state  #  Los estados del reservorio durante el entrenamiento

        # Generar la representación de la serie temporal de validación utilizando los estados del reservorio
        # (suponiendo que ya has definido una función para generar la representación)
        val_repr = self.generate_representation (val_data)  # Esta función debería generar la representación de la serie temporal de validación

        # Predecir las etiquetas de salida para la parte de validación utilizando la capa de salida entrenada
        predictions = self.readout.predict (val_repr)  # Suponiendo que ya has entrenado tu capa de salida (readout) durante el entrenamiento

        # Visualizar las series temporales de entrenamiento, validación y predicciones
        plt.figure (figsize = (10, 5))

        # Serie temporal de entrenamiento en azul
        plt.plot (train_data  [:, :, 0].numpy(), color = 'blue', label = 'Train')

        # Serie temporal de validación en verde
        # plt.plot (np.arange (train_size, len (mts)), val_data[:, :, 0].numpy(), color='green', label='Validation')

        # Predicciones en rojo
        # plt.plot (np.arange (train_size, len (mts)), predictions[:, :, 0], color='red', label='Predictions')

        plt.title ('Serie Temporal de Entrenamiento, Validación y Predicciones')
        plt.xlabel ('Tiempo')
        plt.ylabel ('Valor')
        plt.legend ()
        plt.show ()   

    
############################# 
## Funciones auxiliares
#############################
 

def print_prediction_results (ground_truth, predicted):
    """
    Print results of different metrics in a table format.

    Parameters:
    ground_truth (numpy.ndarray): Ground truth series.
    predicted (numpy.ndarray): Predicted series.
    """
    # Calculate metrics
    pearson_corr = calculate_pearson_correlation(ground_truth, predicted)
    mse = calculate_mean_squared_error(ground_truth, predicted)
    mae = calculate_mean_absolute_error(ground_truth, predicted)
    #dtw_dist = calculate_dtw_distance(ground_truth, predicted)

    # Print results in a table format
    print("Metric\t\t\t\tResult")
    print("-" * 30)
    print(f"Pearson Correlation\t{pearson_corr}")
    print(f"Mean Squared Error\t{mse}")
    print(f"Mean Absolute Error\t{mae}")
    #print(f"DTW Distance\t\t{dtw_dist}")


def calculate_dtw_distance(ground_truth, predicted):
    """
    Calculate Dynamic Time Warping (DTW) distance between two series.

    Parameters:
    ground_truth (numpy.ndarray): Ground truth series.
    predicted (numpy.ndarray): Predicted series.

    Returns:
    float: DTW distance.
    """
    # Desenrollar los datos de las series temporales
    series_desenrolladas_gt = ground_truth.reshape(-1, ground_truth.shape[2])
    series_desenrolladas_pred = predicted.reshape(-1, predicted.shape[2])

    # Calcular la distancia DTW
    dtw_distance, _ = dtw(series_desenrolladas_gt, series_desenrolladas_pred)
    return dtw_distance

def calculate_mean_absolute_error(ground_truth, predicted):
    """
    Calculate Mean Absolute Error (MAE) between two series.

    Parameters:
    ground_truth (numpy.ndarray): Ground truth series.
    predicted (numpy.ndarray): Predicted series.

    Returns:
    float: Mean Absolute Error.
    """
    mae = mean_absolute_error(ground_truth.flatten(), predicted.flatten())
    return mae

def calculate_mean_squared_error(ground_truth, predicted):
    """
    Calculate Mean Squared Error (MSE) between two series.

    Parameters:
    ground_truth (numpy.ndarray): Ground truth series.
    predicted (numpy.ndarray): Predicted series.

    Returns:
    float: Mean Squared Error.
    """
    mse = mean_squared_error(ground_truth.flatten(), predicted.flatten())
    return mse

def calculate_pearson_correlation(ground_truth, predicted):
    """
    Calculate Pearson correlation coefficient between two series.

    Parameters:
    ground_truth (numpy.ndarray): Ground truth series.
    predicted (numpy.ndarray): Predicted series.

    Returns:
    float: Pearson correlation coefficient.
    """
    correlation_coefficient, _ = pearsonr(ground_truth.flatten(), predicted.flatten())
    return correlation_coefficient


def normalize_time_series (time_series_data):
    """
    Normaliza las series temporales a lo largo del eje de las características utilizando la normalización min-max.

    Parámetros:
        - time_series_data: numpy.ndarray, matriz tridimensional de series temporales con forma (num_series, num_samples, num_features).

    Retorna:
        - normalized_time_series_data: numpy.ndarray, matriz tridimensional de series temporales normalizadas con forma (num_series, num_samples, num_features).
    """
    # Calcular los valores mínimos y máximos a lo largo del eje de las características
    min_values = np.min(time_series_data, axis=(0, 1))  # Mínimos por características
    max_values = np.max(time_series_data, axis=(0, 1))  # Máximos por características

    # Normalizar las series temporales utilizando la normalización min-max
    normalized_time_series_data = (time_series_data - min_values) / (max_values - min_values)

    return normalized_time_series_data


def summary(model, input_size):
    """
    Función para imprimir un resumen de parámetros de un modelo de red neuronal.

    Args:
    - model (nn.Module): El modelo de red neuronal PyTorch.
    - input_size (tuple): El tamaño de la entrada esperada para el modelo.

    Prints:
    - Imprime por consola el total de parámetros y los parámetros entrenables del modelo.
    """

    total_params = 0
    trainable_params = 0
    input_shape = input_size

    def register_hook(module):
        """
        Función interna para registrar ganchos (hooks) en cada módulo del modelo.

        Args:
        - module (nn.Module): El módulo del modelo para registrar el gancho.

        Returns:
        - None
        """
        def hook(module, input, output):
            """
            Función de gancho para calcular el número total y entrenable de parámetros.

            Args:
            - module (nn.Module): El módulo del modelo.
            - input (tuple): Entrada del módulo.
            - output (Tensor): Salida del módulo.

            Returns:
            - None
            """
            nonlocal total_params, trainable_params
            total_params += sum(p.numel() for p in module.parameters())
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
            # Imprimir información opcionalmente
            # print(f"{module.__class__.__name__}:")
            # print(f"  Input shape: {input[0].shape}")
            # print(f"  Output shape: {output.shape}")

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    # Crear un tensor de estado de reservorio con ceros del tamaño adecuado
    reservoir_state = torch.zeros((1, model.reservoir_size), dtype=torch.float32)
    try:
        # Ejecutar un pase hacia adelante de ejemplo para obtener formas de entrada y salida
        with torch.no_grad():
            model(torch.tensor(1, 2), reservoir_state)  # Aquí asumí que 1 y 2 son ejemplos de datos ficticios

    finally:
        for hook in hooks:
            hook.remove()

    print(f"Total de parámetros: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")