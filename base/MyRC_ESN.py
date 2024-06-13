import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import re
import os
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from scipy.io import loadmat 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
 
from scipy.spatial.distance import pdist, cdist, squareform

from base.tensorPCA import tensorPCA

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
        self.use_output_bias         = config ['use_output_bias']
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
            
        torch.manual_seed (0)
        np.random.seed (0)
        
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
            Se transforma la entrada mediante una capa lineal inicial o una multiplicación con los pesos de entrada,
            dependiendo de si se ha habilitado la capa de entrada lineal.
            
            Se obtiene el estado anterior del reservorio correspondiente al instante t-1.
            
            Si t > 0, se toma como estado anterior el estado del reservorio en el instante anterior.
            
            En caso contrario, se inicializa como un tensor de ceros.
            
            Se calcula la representación interna del reservorio multiplicando el estado anterior por los pesos del reservorio.
            
            Luego, se suma la representación transformada por la entrada y se añade ruido para introducir aleatoriedad.
            A continuación, se aplica la función de no linealidad (tangente hiperbólica) para generar la salida del reservorio.
            Si se ha habilitado la plasticidad sináptica, se aplica para ajustar los pesos de los sinapsis.
            Si se ha habilitado la plasticidad intrínseca, se aplica para ajustar los nodos del reservorio.
            Finalmente, se actualiza el estado del reservorio en el instante t, interpolando entre el estado anterior
            y la representación transformada por la entrada, con la tasa de fuga como factor de interpolación.
            El estado actualizado del reservorio se devuelve como salida del método.
        '''
        
        # print(f'input_data:{input_data.shape}')
        input_data      = input_data.to(self.device)
        reservoir_state = reservoir_state.to(self.device)
        
        # Aplicar transformación lineal a la entrada si se usa una capa lineal inicial
        # o una combinación lineal con los pesos asociados a cada uno de los nodos internos

        if self.use_input_layer:
            reservoir_input = self.input_layer(input_data)
        else:
            reservoir_input = torch.matmul(input_data, self.input_weights.to(self.device))

        # obtenemos el estado actual trasnmitiendo al información de entrada a los nodos
        # del RC atendiendo a la matriz de conectividad (reservoir_state)
        current_rc_state = None
        if t > 0:
            current_rc_state = reservoir_state [:, (t-1), :]
        else:
            current_rc_state = torch.zeros_like (reservoir_input).to(self.device)
        
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
        # Implementación de la plasticidad sináptica
        if self.plasticity_synaptic is not None:
            if self.plasticity_synaptic == 'hebb':
                # Regla de Hebb
                # Obtener la activación de las neuronas en el reservorio
                self._apply_bcm_rule (reservoir_state, t)
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
                self._apply_ojas_rule (reservoir_state, input_data, t)
            else:
                raise ValueError("Invalid synaptic plasticity rule.") 

    def _apply_ojas_rule (self, reservoir_state, input_data, t):
        # Obtener el estado actual del reservorio en el tiempo t
        reservoir_state_t = reservoir_state [:, t, :]  # Dimensiones: (batch_size, reservoir_size)
    
        # Calcular la activación de las neuronas en el reservorio en el tiempo t
        activations = reservoir_state_t  # Dimensiones: (batch_size, reservoir_size)
    
        # Calcular la proyección de la entrada a través de los pesos del reservorio
        projection = torch.matmul (input_data, self.reservoir_weights.t ().to (self.device)) # Dimensiones: (batch_size, reservoir_size)
        # Calcular la diferencia entre la entrada y la proyección
        difference = input_data - projection  # Dimensiones: (batch_size, reservoir_size)
    
        # Calcular el cambio en los pesos sinápticos utilizando la regla de Oja
        # Nota: Usamos una versión de aprendizaje de Oja adecuada para batch processing
        delta_weights = self.learning_rate * torch.bmm (activations.unsqueeze (2), difference.unsqueeze (1)).mean (dim = 0)
    
        # Actualizar los pesos sinápticos del reservorio
        self.reservoir_weights += delta_weights.to (self.device)

    def _apply_bcm_rule (self, reservoir_state, t):
        # Obtener el estado actual del reservorio en el tiempo t
        reservoir_state_t = reservoir_state [:, t, :]

        # Calcular la activación promedio de las neuronas en el reservorio en el tiempo t
        avg_activation = reservoir_state_t.mean (dim = 0)

        # Calcular el cambio en los pesos sinápticos utilizando la regla BCM
        delta_weights = self.learning_rate * (torch.outer(avg_activation, avg_activation) - self.theta_m * torch.pow(self.reservoir_weights, 2))

        # Actualizar los pesos sinápticos del reservorio
        self.reservoir_weights += delta_weights.to(self.device)
        
        # Opcionalmente, actualizar el umbral de modificación
        self.theta_m = reservoir_state_t.norm(dim = 1).mean()


    def _spectral_reservoir_weights_circ (self, n_internal_units, spectral_radius):

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
        print (f'reservoir_weights 0:{self.reservoir_weights.device}')
        self.reservoir_weights = self.reservoir_weights.to(self.device)
        print (f'reservoir_weights 1:{self.reservoir_weights.device}')
        if not isinstance (self.reservoir_weights, torch.Tensor):
            self.reservoir_weights = torch.tensor (self.reservoir_weights, dtype = torch.float32).to (self.device)
        print (f'reservoir_weights 2:{self.reservoir_weights.device}')
        self.reservoir_weights *= self.spectral_radius / torch.max (torch.abs (torch.linalg.eigvals (self.reservoir_weights)))
        print (f'reservoir_weights 3:{self.reservoir_weights.device}')
        mask = torch.rand (self.reservoir_size, self.reservoir_size).float()  < self.connectivity
        print (f'mask 0:{mask.device}')
        print (f'reservoir_weights 4:{self.reservoir_weights.device}')
        mask = mask.to(self.device) # Move mask to the same device
        print (f'mask 1:{mask.device}')
        self.reservoir_weights *= mask 
        return self.reservoir_weights
        
    def _initialize_weights (self):
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
          self.input_weights     = (2.0 * np.random.binomial(1, 0.5, [self.input_size, self.reservoir_size]) - 1.0) * self.input_scaling
          self.reservoir_weights = (2.0 * np.random.binomial(1, 0.5, [self.reservoir_size, self.reservoir_size]) - 1.0)
          print(f'{type(self.reservoir_weights)}') 
        else:
            raise ValueError("Invalid initialization type. Use 'rand', 'orthogonal', or 'trunc_normal'.")
            
    def plot_spectral_graph (self):
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
        self.model             = model
        self.readout           = None # readout
        self.config            = config
        self.readout_type      = self.config ['readout_type']
        self.mts_rep           = self.config ['mts_rep']
        self.w_ridge_embedding = self.config ['w_ridge_embedding']
        self.w_ridge           = self.config ['w_ridge']
        self.w_l2              = self.config ['w_l2']
        self.svm_C             = self.config ['svm_C']
        self.svm_kernel        = self.config ['svm_kernel'] 
        self.svm_gamma         = self.config ['svm_gamma']
        self.mlp_layout        = self.config ['mlp_layout']
        self.num_epochs        = self.config ['num_epochs']
        self.nonlinearity      = self.config ['nonlinearity']
        self.mts_rep           = self.config ['mts_rep']
        self.dimred_method     = self.config ['dimred_method']
        self.n_dim             = self.config ['n_dim']
        self.bidir             = self.config ['bidir']
        self.threshold         = self.config ['threshold']
        self.washout           = self.config ['washout']
        nc_drop                = self.config ['n_drop']
        
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
                self.readout = Ridge (alpha = self.w_ridge)
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
                    batch_size          = 32,
                    learning_rate       = 'adaptive', # 'constant' or 'adaptive'
                    learning_rate_init  = 0.001,
                    max_iter            = self.num_epochs,
                    early_stopping      = False, # if True, set validation_fraction > 0
                    validation_fraction = 0.0 # used for early stopping
                    )
            else:
                raise RuntimeError('Invalid readout type. Only (lin, svm, ovr or mlp')
                
        print (f'* self.mts_rep : {self.mts_rep}')
        
        # Initialize ridge regression model
        if self.mts_rep == 'output' or self.mts_rep == 'reservoir':
            self._ridge_embedding = Ridge (alpha = self.w_ridge_embedding, fit_intercept = True)
            
        # Initialize dimensionality reduction method
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components = self.n_dim)            
            elif self.dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components = self.n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID') 
                
    def _process_transient_ESN (self, X):
        """
            Process the transients of an Echo State Network (ESN).

            Args:
            - data: Input data containing the signals from the ESN.

            Returns:
            - output: Representation of the last transient of the ESN.
            - response: State of the ESN with all transients.
        """
        # print (f'X:{X.shape}')
        n_serial = X.size (1) # número de instantes de la serie temporal
          
        reservoir_state = torch.zeros ((X.size(0), X.size(1), self.model.reservoir_size), dtype = torch.float32)
        # print (f'* _process_transient_ESN : reservoir_state : {reservoir_state.shape}')
        for t in range (n_serial):
            # print (f' n_serial : t : {t}')
            input_sequence = X [ :, t, :] # input_data[trial, t:t+1, :]
            input_sequence = input_sequence.float()
            
            #if t % 10 == 0:
                #print (f'* _process_transient_ESN : n_sample process {t}/{n_serial}')
            
            reservoir_state  = self.model (input_sequence, reservoir_state, t)
  
        return reservoir_state
        
    def remove_random_transients (self, input_data, reservoir_state):
        transient_indices = None
        num_transients = min (self.n_drop, input_data.shape [1] - 1)
        if num_transients > 0:
            transient_indices = np.random.choice (input_data.shape [1], size = (input_data.shape[1] - num_transients), replace = False)
            state_matrix = reservoir_state [:, transient_indices, :]
        else:
            state_matrix = reservoir_state
        return state_matrix, transient_indices

    def remove_initial_transients (self, input_data, reservoir_state):
        T = input_data.shape[1]
        #state_matrix = reservoir_state[:, T - self.n_drop:, :]
        state_matrix = reservoir_state[:, self.n_drop:, :]
        return state_matrix, np.arange(0, self.n_drop)

    def not_remove (self, input_data, reservoir_state):
        # print('not_remove')
        # print(f'reservoir_state:{reservoir_state.shape}')
        
        # Número total de columnas en input_data
        num_columns = input_data.shape [1]
        
        # Calcular los índices de las columnas a eliminar, excluyendo las dos últimas
        n_drop_indices = np.arange (num_columns - 2)
        #print(f'n_drop_indices:{n_drop_indices}')
        
        if len(n_drop_indices) > 0:
            # Copiar el estado del reservorio
            reservoir_state_copy = np.copy (reservoir_state)
            
            # Verificar si hay más de una columna en la segunda dimensión
            if reservoir_state.shape[1] > 1:
                # print(f'reservoir_state (before deletion):{reservoir_state.shape}')
                
                # Eliminar las columnas especificadas en n_drop_indices
                reservoir_state_copy = np.delete (reservoir_state_copy, n_drop_indices, axis = 1)
                # print(f'reservoir_state (after deletion):{reservoir_state_copy.shape}')
            
            # Asignar el resultado a state_matrix
            state_matrix = reservoir_state_copy
        else:
            # Si no hay índices a eliminar, state_matrix es igual a reservoir_state
            state_matrix = reservoir_state
        #print(f'np.arange(0, n_drop_indices):{n_drop_indices}')
        #print(f'state_matrix:{state_matrix.shape}')
        return state_matrix, n_drop_indices
        
    def _get_states (self, input_data, birdir = False, evaluate = False):
        """
            Get the states of the Echo State Network (ESN) from EEG signals of each channel.

            Args:
            - input_data: Input data containing the EEG signals from each channel.
            - birdir: Flag indicating whether to process the data bidirectionally (default: False).

            Returns:
            - out_rc: The last transient (last state of ESN).
            - mts_rep: Representation of all transients.
            - all_rc: All transients except those specified for deletion by configuration.
            
            NNote:- The '--123--' marker is used to indicate when all functionality has been tested. Once verified, 'mts_rep' will be removed, leaving only 'all_rc'. 
        """
        # Convertir listas a tensores
        if not isinstance (input_data, torch.Tensor):
            input_data   = torch.tensor(input_data, dtype = torch.float32)
     
        # ============ Buscamos el estado del RC al procesar la series tmeporales ============
        
        reservoir_state  = self._process_transient_ESN (input_data)
        # print (f'* _get_states : reservoir_state : {reservoir_state.shape}')
        
        # ============ Determinar el número de transitorios a eliminar y los elimina segun self.n_drop ============
        n_drop = 0
        if evaluate: # En fase de test o evaluación no se eliminan transitorios
            n_drop = min (self.n_drop, input_data.size(1))

        state_matrix              = None
        n_drop_indices            = None
        n_drop_indices_b_adjusted = 0
        if n_drop > 0:
            if self.washout == 'init':
                state_matrix, n_drop_indices  = self.remove_initial_transients (input_data, reservoir_state)
            elif self.washout == 'rand':
                state_matrix, n_drop_indices  = self.remove_random_transients (input_data, reservoir_state)
            else:
                state_matrix, n_drop_indices  = self.remove_initial_transients (input_data, reservoir_state)
        elif n_drop < 0:    
            state_matrix, n_drop_indices  = self.not_remove (input_data, reservoir_state)
        else:
            state_matrix   = reservoir_state
            n_drop_indices = 0
            
        # print (f'* get_states : state_matrix : {state_matrix.shape}')
        # print (f'* get_states : birdir : {birdir}')
        # ============ Si activamos una estado bidireccional del estado interno del ESN ============
        if birdir:
            input_data_b       = torch.tensor(input_data.numpy () [:, ::-1, :].copy(), dtype = torch.float32)
            reservoir_state_b  = self._process_transient_ESN (input_data_b)
            
            # print (f'* get_states : reservoir_state_b : {reservoir_state_b.shape}')
            
            state_matrix_b   = None
            n_drop_indices_b = None
            if n_drop > 0:
                if self.washout == 'init':
                    state_matrix_b, n_drop_indices_b = self.remove_initial_transients (input_data_b, reservoir_state_b)
                elif self.washout == 'rand':
                    state_matrix_b, n_drop_indices_b  = self.remove_random_transients (input_data_b, reservoir_state_b)
                else:
                    state_matrix_b, n_drop_indices_b  = self.remove_initial_transients (input_data_b, reservoir_state_b)
            elif n_drop < 0:    
                state_matrix_b, n_drop_indices_b  = self.not_remove (input_data_b, reservoir_state_b)
            else:
                state_matrix_b   = reservoir_state_b

            # print (f'* get_states : state_matrix_b : {state_matrix_b.shape}')
            # Ajustar índices de eliminación para la secuencia inversa
            if n_drop_indices_b is None:
                n_drop_indices_b_adjusted = 0
            else:
                n_drop_indices_b_adjusted = input_data.size(1) - np.array(n_drop_indices_b) - 1
            
            # Convertir tensores a numpy arrays
            if torch.is_tensor (state_matrix_b):
                state_matrix_b = state_matrix_b.detach().numpy()
                
            if torch.is_tensor (state_matrix):
                state_matrix = state_matrix.detach().numpy()   
                
            if torch.is_tensor (reservoir_state):
                reservoir_state = reservoir_state.detach().numpy()
                
            if torch.is_tensor (reservoir_state_b):
                reservoir_state_b = reservoir_state_b.detach().numpy()    
                
            mts_rep_state = np.concatenate((state_matrix, state_matrix_b), axis=1)
            all_rc        = np.concatenate((reservoir_state, reservoir_state_b), axis=1)
        else:
            mts_rep_state = state_matrix
            all_rc        = reservoir_state

        # print ('Procesamiento ESN completado.')
        # print (f'* get_states : all_rc : {all_rc.shape}')
        # print (f'* get_states : mts_rep_state : {mts_rep_state.shape}')
        # print (f'* get_states : n_drop_indices : {n_drop_indices}')
        # print (f'* get_states : n_drop_indices_b_adjusted : {n_drop_indices_b_adjusted}')
        return mts_rep_state, all_rc, n_drop_indices, n_drop_indices_b_adjusted 
        
    def apply_dimensionality_reduction (self, reservoir_state):
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
    
        red_states = np.real (red_states) # Occasionally I get complex matrices that the rest of the code does not support
        return red_states

    def compute_internal_representation_output(self, red_states, input_data):
        """
        Computes the internal representation output based on the reduced reservoir states and input data.
    
        Args:
        - red_states (numpy.ndarray): The reduced reservoir states.
        - input_data (numpy.ndarray or torch.Tensor): The input data.
    
        Returns:
        - numpy.ndarray: The internal representation output.
        """
        # Si es bidireccional, concatenar los datos de entrada con su versión invertida
        if self.bidir:
            if torch.is_tensor(input_data):
                reversed_input = torch.flip(input_data, dims=[1])
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
            
    def compute_internal_representation_output_mtx(self, red_states, input_data, n_drop_indices, n_drop_indices_b):
        """
        Computes the internal representation output based on the reduced reservoir states and input data.
    
        Args:
        - red_states (numpy.ndarray): The reduced reservoir states.
        - input_data (numpy.ndarray or torch.Tensor): The input data.
    
        Returns:
        - numpy.ndarray: The internal representation output.
        """
        if self.bidir:
            if torch.is_tensor(input_data):
                reversed_input = torch.flip(input_data, dims=[1])
                input_data = torch.cat((input_data, reversed_input), dim=1)
            else:
                input_data = np.concatenate((input_data, input_data[:, ::-1, :]), axis=1)

        if not torch.is_tensor(input_data):
            input_data = torch.tensor(input_data)
    
        if self.n_drop > 0:
            mask = torch.ones(input_data.size(1), dtype=torch.bool)
            indices_to_drop = set(n_drop_indices)
            if self.bidir:
                indices_to_drop.update(n_drop_indices_b)
            mask[list(indices_to_drop)] = False
            input_data_remove = input_data[:, mask, :]
        else:
            input_data_remove = input_data
    
        coeff_tr, biases_tr = [], []
    
        for i in range(input_data.shape[0]):
            self._ridge_embedding.fit(red_states[i, :-1, :], input_data_remove[i, :-1, :])
            coeff_tr.append(self._ridge_embedding.coef_.ravel())
            biases_tr.append(self._ridge_embedding.intercept_.ravel())
    
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr
        
    def fit_ridge_regression (self, red_states, input_data):
        coeff_tr  = []
        biases_tr = []
        
        for i in range(input_data.shape[0]):
            # print (f'red_states:{red_states [i].shape}')
            self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeff_tr.append(self._ridge_embedding.coef_.ravel())
            biases_tr.append(self._ridge_embedding.intercept_.ravel())
        
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
    
        return input_repr
    def compute_output_layer(self, input_repr, target_data = None):
        """
        Computes the output layer based on the internal representation and target data.
    
        Args:
        - input_repr (numpy.ndarray): The internal representation.
        - target_data (numpy.ndarray): The target data for training the output layer. Default is None.
    
        Returns:
        - object: The trained output layer.
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
            # get_states (model, input_data, target_data, reservoir_state, config):
            mts_rep_state, rc_state, n_drop_indices, n_drop_indices_b = self._get_states (input_data, self.bidir, evaluate = False)

        # print ('fit :', mts_rep_state [:,-1,:])
        self.reservoir_state  = rc_state       # Almacenamos todos los transitorios sin eliminar n_drop 
        self.mtx_rc_state     = mts_rep_state  # Almacenamos aquellos transitoriso que hemos dicho que se almacene 
        
        # print (f'fit :  self.mtx_rc_state: {self.mtx_rc_state.shape}')
        # print (f'fit :  self.reservoir_state: {self.reservoir_state.shape}')

        # ============ Dimensionality reduction of the reservoir states ============  
        
        # print (f'fit :  self.dimred_method: {self.dimred_method}')
        
        red_states     = self.apply_dimensionality_reduction (self.reservoir_state)
        mtx_red_states = self.apply_dimensionality_reduction (self.mtx_rc_state)
        
        #print (f'fit : red_states:{red_states.shape}')  
        #print (f'fit : mtx_red_states:{mtx_red_states.shape}') 
        
        # ============ Generate representation of the MTS ============
        # Output model space representation
        if self.mts_rep == 'output':
            input_repr     = self.compute_internal_representation_output (red_states, input_data)  
            mtx_input_repr = self.compute_internal_representation_output_mtx (mtx_red_states, input_data, n_drop_indices, n_drop_indices_b)
        # Reservoir model space representation
        elif self.mts_rep == 'reservoir':
            input_repr     = self.fit_ridge_regression (red_states, input_data) 
            mtx_input_repr = self.fit_ridge_regression (mtx_red_states, input_data) 
         # Last state representation        
        elif self.mts_rep == 'last':
            input_repr     = red_states [:, -1, :]
            mtx_input_repr = mtx_red_states [:, -1, :]
         # Mean state representation        
        elif self.mts_rep == 'mean':
            input_repr     = np.mean (red_states, axis = 1)
            mtx_input_repr = np.mean (mtx_red_states, axis = 1)
        elif self.mts_rep == 'id':
            input_repr     = red_states
            mtx_input_repr = mtx_red_states
        else:
            raise RuntimeError('Invalid representation ID: output, reservoir, last o mean')  
        
        self.mtx_input_repr = mtx_input_repr
        self.input_repr     = input_repr
        
        # print (f'fit : mtx_input_repr:{mtx_input_repr.shape}')
        # print (f'fit : input_repr:{input_repr.shape}')
        # ============ Apply readout ============
        self.output_rc_layer     = self.compute_output_layer (self.input_repr, target_data)
        self.mtx_output_rc_layer = self.compute_output_layer (self.mtx_input_repr, target_data)
        
        return self.output_rc_layer, self.mtx_output_rc_layer, self.reservoir_state, self.mtx_rc_state, self.input_repr, self.mtx_input_repr, red_states, mtx_red_states

    def fit_evaluate (self, Xte, Y_test):
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
        # Nota: En fase de evaluación no se eliminan transitorios
        # ============ Compute reservoir states ============
        mts_rep_te, rc_state_te, n_drop_indices, n_drop_indices_b  = self._get_states (Xte, self.bidir, evaluate = True)

        print (f'fit_evaluate :mts_rep_te: {mts_rep_te.shape}')    
        print (f'fit_evaluate :rc_state_te: {rc_state_te.shape}')  
        print (f'fit_evaluate :n_drop_indices: {n_drop_indices}') 
        print (f'fit_evaluate :n_drop_indices_b: {n_drop_indices_b}') 
        
        # ============ Dimensionality reduction of the reservoir states   ============
        if self.dimred_method.lower() == 'pca':
            n_samples  = mts_rep_te.shape [0]
            res_states = mts_rep_te.reshape  (-1, mts_rep_te.shape [2])                   
            # ..transform..mts_rep_te
            red_states = self._dim_red.fit_transform(res_states)          
            # ..and put back in tensor form
            red_states_te = red_states.reshape(n_samples,-1,red_states.shape [1])          
        elif self.dimred_method.lower() == 'tenpca':
            if torch.is_tensor (mts_rep_te):
                mts_rep_te = mts_rep_te.detach().numpy()
            red_states_te = self._dim_red.fit_transform (mts_rep_te)       
        else: # Skip dimensionality reduction
            red_states_te = mts_rep_te
            
        print (f'fit_evaluate : red_states_te:{red_states_te.shape}')          
        # ============ Generate representation of the MTS ============
        coeff_te  = []
        biases_te = []
        
        if torch.is_tensor(red_states_te):
            red_states_te = red_states_te.detach().numpy()
        if torch.is_tensor(mts_rep_te):
            mts_rep_te = mts_rep_te.detach().numpy()   
        
        # Output model space representation
        if self.mts_rep=='output':
            if self.bidir:
                Xte = np.concatenate ((Xte, Xte[:, ::-1, :]), axis = 1)  
 
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, 0:-1, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # Reservoir model space representation
        elif self.mts_rep=='reservoir':    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
        # Last state representation        
        elif self.mts_rep=='last':
            input_repr_te = red_states_te[:, -1, :]
        # Mean state representation        
        elif self.mts_rep=='mean':
            input_repr_te = np.mean(red_states_te, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID: output, reservoir, last, mean') 

        print (f'fit_evaluate :input_repr_te: {input_repr_te.shape}')
        self.input_repr_te = input_repr_te
        # ============ Apply readout ============
        print (f'fit_evaluate :lin :fit_evaluate :self.readout_type : {self.readout_type }')
        pred_class = None
        if self.readout_type == 'lin':  # Linear regression
            logits = self.readout.predict(input_repr_te)
            pred_prob = 1 / (1 + np.exp(-logits))
            pred_class_max = np.argmax(logits, axis=1)
            pred_class = self.convert_to_one_hot(pred_class_max)
            print(f'fit_evaluate :lin :logits : {logits}')
            print(f'fit_evaluate :lin :pred_class : {pred_class}')
            print(f'fit_evaluate :lin :pred_prob : {pred_prob}')
            print(f'fit_evaluate :lin :pred_class_max : {pred_class_max}')
            Y_test_int = Y_test.astype(int)
        elif self.readout_type == 'svm':  # SVM readout
            Kte = cdist(input_repr_te, self.input_repr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma * Kte)
            pred_class = self.readout.predict(Kte)
            pred_class_multilabel = np.zeros_like(Y_test)
            for i, label in enumerate(pred_class):
                pred_class_multilabel[i, label] = 1
            pred_class = pred_class_multilabel
            Y_test_int = Y_test.astype(int)
            print(f'fit_evaluate _SVM:Yte : {Y_test_int}')
            print(f'fit_evaluate _SVM:pred_class : {pred_class}')
        
        elif self.readout_type == 'ovr':  # One-vs-Rest Classifier
            pred_prob = self.readout.predict_proba(input_repr_te)
            print(f'fit_evaluate :ovr :pred_prob : {pred_prob}')
            pred_class = (pred_prob > self.threshold).astype(int)
            Y_test_int = Y_test.astype(int)
            print(f'fit_evaluate :Yte : {Y_test_int}')
            print(f'fit_evaluate :pred_class : {pred_class}')
        
        elif self.readout_type == 'mlp':  # MLP (deep readout)
            pred_prob = self.readout.predict_proba(input_repr_te)
            pred_class_max = np.argmax(pred_prob, axis=1)
            pred_class = self.convert_to_one_hot(pred_class_max)
            print(f'fit_evaluate :mlp :pred_class : {pred_class}')
            print(f'fit_evaluate :mlp :pred_prob : {pred_prob}')
            print(f'fit_evaluate :mlp :pred_class_max : {pred_class_max}')
            Y_test_int = Y_test.astype(int)
        else:
            print("Error while evaluating. When evaluating, we need to have a correct readout (lin, svm, ovr, or mlp).")
            return None, None, None

        if pred_class is not None:
            Y_test_int = Y_test.astype(int)
            print(f'fit_evaluate :Yte : {Y_test_int}')
            print(f'fit_evaluate :pred_class : {pred_class}')
            if Y_test.shape[1] > 1:
                f1, c_matrix = self.compute_multilabel_test_scores(pred_class, Y_test_int)
                return pred_class, f1, c_matrix
            else:
                f1 = f1_score(Y_test_int, pred_class, average='micro')  # o 'macro', 'weighted'
                c_matrix = multilabel_confusion_matrix(Y_test_int, pred_class)
                print("F1-score:", f1)
                print("Confusion Matrix:\n", c_matrix)
                accuracy, f1, c_matrix = self.compute_test_scores(pred_class, Y_test_int)
                return pred_class, accuracy, f1, c_matrix
        else:
            return None, None, None

            
    def compute_test_scores_ (self, pred_class, Yte):
        """
        Compute classification accuracy and F1 score for multilabel classification.
        """
        accuracy = accuracy_score(Yte, pred_class, multi_label=True)
        f1 = f1_score(Yte, pred_class, average='weighted', multi_label=True)
    
        return accuracy, f1
        
    def compute_test_scores (self, pred_class, Yte):
        """
        Wrapper to compute classification accuracy and F1 score
        """
        true_class = np.argmax(Yte, axis=1)
    
        accuracy = accuracy_score(true_class, pred_class)
        if Yte.shape[1] > 2:
            f1 = f1_score(true_class, pred_class, average='weighted')
        else:
            f1 = f1_score(true_class, pred_class, average='binary')
    
        return accuracy, f1
    def generate_representation (self, val_data):
        #  Calcular la representación utilizando los estados del reservorio
        val_repr = np.dot (val_data, self.reservoir_state)  # Ejemplo de representación básica: producto punto entre los datos de validación y los estados del reservorio

        return val_repr
    def convert_to_one_hot (self, pred_class_max, num_classes = 2):
        num_samples = len(pred_class_max)
        one_hot = np.zeros((num_samples, num_classes), dtype=np.int32)

        for i in range(num_samples):
            if pred_class_max[i] == 0:
                one_hot[i, 0] = 1  # Clase 0: 10
            elif pred_class_max[i] == 1:
                one_hot[i, 1] = 1  # Clase 1: 01

        return one_hot

    
    def compute_multilabel_test_scores (self, y_pred, Yte):
        # Comprobar si son multilabel
        # Calcular la matriz de confusión multilabel
        confusion_matrices = multilabel_confusion_matrix(Yte, y_pred)

        # Calcular el F1-score multilabel
        f1 = f1_score(Yte, y_pred, average='weighted')

        # Imprimir resultados
        print("Matriz de Confusión Multilabel:")
        print(confusion_matrices)
        print("F1-Score Multilabel:", f1)
        
        return f1, confusion_matrices
 
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

    
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from tslearn.metrics import dtw

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
    total_params = 0
    trainable_params = 0
    input_shape = input_size

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params, trainable_params
            total_params += sum(p.numel() for p in module.parameters())
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
            # print(f"{module.__class__.__name__}:")
            # print(f"  Input shape: {input[0].shape}")
            # print(f"  Output shape: {output.shape}")

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)
    # Crear un tensor de estado de reservorio con ceros del tamaño adecuado
    reservoir_state = torch.zeros((1, model.reservoir_size), dtype=torch.float32)
    model(torch.tensor (1,2) , reservoir_state)  # Pasar el tensor de estado de reservorio como argumento adicional

    for hook in hooks:
        hook.remove()

    print(f"Total de parámetros: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")

