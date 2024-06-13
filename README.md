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

Puedes utilizar tanto el código que hay en los notebooks como los diferentes scripts existentes en el directorio para realizar la evaluación.
Para ejecutar la evaluación, usa el siguiente comando:


## Características de las Señales EEG

Las señales de electroencefalografía (EEG) son una representación directa de la actividad eléctrica del cerebro. Estas señales se generan principalmente por los potenciales postsinápticos de las neuronas piramidales en la corteza cerebral, que crean campos eléctricos detectables en el cuero cabelludo. El EEG es una herramienta no invasiva que permite registrar estas señales a través de electrodos colocados en la superficie del cuero cabelludo.

Las señales EEG se forman mediante la superposición de múltiples corrientes sinápticas generadas por la actividad sincronizada de grandes poblaciones neuronales. Las corrientes sinápticas crean dipolos eléctricos cuyos campos se suman y pueden ser detectados a nivel macroelectrodo. La amplitud y frecuencia de las ondas EEG reflejan la dinámica de la actividad neuronal subyacente.

Las ondas EEG se clasifican en varios tipos, cada una con características de frecuencia y amplitud específicas:

* **Ondas Delta (0.5-4 Hz):** Predominan durante el sueño profundo y están asociadas con actividades regenerativas del cerebro.
* **Ondas Theta (4-8 Hz):** Comunes en estados de somnolencia, meditación y etapas tempranas del sueño. También se observan durante tareas de memoria y aprendizaje.
* **Ondas Alfa (8-12 Hz):** Asociadas con estados de relajación y alerta tranquila. Se generan principalmente en las regiones occipitales durante la vigilia relajada con los ojos cerrados.
* **Ondas Beta (12-30 Hz):** Relacionadas con la actividad mental activa, atención y procesamiento de información. Se observan en estados de alerta y concentración.
* **Ondas Gamma (30-100 Hz):** Asociadas con el procesamiento de información de alto nivel, la percepción consciente y la integración de diferentes modalidades sensoriales.

El análisis de las señales EEG permite identificar distintos tipos de ondas cerebrales, cada una asociada con diferentes estados mentales y actividades. Estas ondas se clasifican según sus frecuencias y características específicas, proporcionando información valiosa sobre la actividad cerebral:

* **Ondas Delta (1-5 Hz):** Asociadas con el sueño profundo y son predominantes en bebés.
* **Ondas Theta (4-7 Hz):** Asociadas con creatividad y espontaneidad, pero también con distracción, ensoñaciones, depresión y otros trastornos emocionales. Los niños presentan amplitudes de ondas Theta mayores que los adultos.
* **Ondas Alfa (8-12 Hz):** Relacionadas con la meditación y la sensación de calma interior. Son prominentes en las áreas posteriores del cerebro; sin embargo, su predominio en las zonas frontales puede indicar TDA-H, depresión y otros trastornos.
* **Ondas Beta (13-21 Hz):** Comprenden la actividad de ondas rápidas y se asocian con concentración, orientación hacia el exterior o estados de reflexión. Las frecuencias dominantes de Beta son mayores en adultos que en niños, con máxima amplitud normalmente en regiones frontales del cerebro.
* **Ondas Hi-Beta o Beta rápida (20-32 Hz):** Asociadas con la actividad cognitiva (resolver problemas, estudiar), pero también con preocupaciones, ansiedad y obsesiones.

Las ondas EEG tienen una distribución específica de energía que puede variar según el estado mental y las actividades del individuo:

* Valores altos de onda delta junto con valores bajos en otras bandas de frecuencia pueden indicar un estado de reposo o sueño, mientras que una distribución más equilibrada de energía entre diferentes bandas podría indicar actividad mental.
* La onda alfa está relacionada con la falta de actividad cerebral y se genera durante el descanso, meditación o paseos.
* Las ondas beta se generan cuando el cerebro está activo y concentrado en actividades mentales, como durante conversaciones intensas, debates o enseñanza.
* Las ondas theta surgen cuando un individuo está tan relajado que comienza a soñar despierto, como durante actividades automáticas como conducir o ducharse.
* Las ondas delta se originan durante el sueño profundo.
* La onda gamma es la más rápida y está relacionada con la conciencia y experiencias de meditación.


## Preprocesamiento de Señales EEG

Las señales EEG proporcionadas están en su forma cruda, lo que requiere un preprocesamiento antes de su análisis. Este preprocesamiento incluye varias etapas esenciales para asegurar la calidad y usabilidad de los datos:

### Filtrado de Datos

Las señales EEG son inherentemente ruidosas, por lo que se debe aplicar un filtrado para eliminar el ruido de alta y baja frecuencia no deseado. Esto puede incluir el uso de filtros pasa banda para mantener las frecuencias de interés y eliminar artefactos de frecuencia fuera del rango típico de la actividad cerebral.

### Eliminación de Artefactos

Los artefactos en las señales EEG, como los movimientos oculares y la actividad muscular, deben ser identificados y eliminados. Para esto, se puede utilizar la librería MNE-Python, que proporciona herramientas avanzadas para la detección y eliminación de artefactos mediante técnicas como el Análisis de Componentes Independientes (ICA).

### Segmentación de Datos

Para el experimento donde el estudio se realiza a través de la extracción de características, se realizará una segmentación de las señales temporales. Las señales EEG se segmentan en ventanas temporales para su análisis. Esta segmentación permite una mejor gestión de los datos y facilita la aplicación de técnicas de procesamiento y análisis posteriores.

# Reservoir Computing (RC)

![imagen](https://github.com/jogugil/MyRC/assets/15160072/26da01fe-6d7f-4ac5-a498-d1ca99be794e)

El Reservoir Computing (RC) es una técnica avanzada para el análisis de señales EEG que puede identificar patrones dinámicos directamente de las series temporales en bruto, sin necesidad de extracción previa de características. A diferencia de las redes recurrentes tradicionales, el RC puede distinguir entre jóvenes adultos y adultos mayores y clasificar sus EEG según su grupo. Combinando su capacidad de análisis y predicción de patrones con la información de las señales EEG, se espera identificar diferencias significativas en la actividad cerebral entre estos dos grupos en reposo. Evaluar estas capacidades permitirá determinar la eficacia del RC en discernir y clasificar diferencias cerebrales relacionadas con la edad, profundizando en la relación entre actividad cerebral y envejecimiento.

El RC utiliza redes neuronales recurrentes con una arquitectura única que incluye una capa de reservorio de neuronas recurrentes con conexiones aleatorias y una capa de salida entrenable. Esta configuración permite al reservorio capturar y procesar información de manera eficiente, siendo especialmente útil para aplicaciones con datos altamente dinámicos y no lineales, como el procesamiento de señales neuronales. Para lograr sus objetivos, se emplearán tanto enfoques no supervisados como supervisados y técnicas de imagen, como las gráficas de recurrencia, para identificar los patrones dinámicos procesados por el RC.

#  Arquitectura del RC

![imagen](https://github.com/jogugil/MyRC/assets/15160072/d9630b14-0aa8-4726-932e-9732f91881ec)

## Capa de Entrada

Esta capa recibe las señales de entrada y las presenta al reservorio. Puede incluir la codificación de la información temporal y espacial de las señales, como en el caso del procesamiento de señales EEG.

$$
x(t) = W_{in} u(t) + b_{in}
$$

donde:
- \( x(t) \) es el estado de las neuronas de entrada en el tiempo \( t \).
- \( W_{in} \) es la matriz de pesos de la capa de entrada.
- \( u(t) \) es el vector de entrada en el tiempo \( t \).
- \( b_{in} \) es el vector de sesgo de la capa de entrada.

## Reservorio

Esta capa está formada por un conjunto de neuronas recurrentes interconectadas con conexiones aleatorias. El reservorio actúa como una memoria dinámica que captura la información temporal de las entradas.

La ecuación que describe la evolución temporal del estado del reservorio es la siguiente:

$$
r(t + 1) = \sigma(W_{res} r(t) + W_{in} u(t) + b_{res})
$$

donde:
- \( r(t) \) es el estado del reservorio en el tiempo \( t \).
- \( W_{res} \) es la matriz de pesos recurrentes del reservorio.
- \( \sigma \) es la función de activación no lineal (generalmente se suele usar tangente hiperbólica).
- \( b_{res} \) es el vector de sesgo del reservorio.

Las conexiones recurrentes permiten que el reservorio mantenga una memoria a corto plazo de las entradas pasadas, lo cual es crucial para capturar la dinámica temporal de las señales EEG. Las conexiones de retroalimentación en esta capa permiten que las salidas anteriores del modelo se retroalimenten como entradas adicionales al reservorio, mejorando la modelización de la dinámica temporal.

## Funciones de Activación

En Reservoir Computing, la elección de la función de activación en la capa de reservorio juega un papel crucial en el rendimiento y la capacidad de generalización del modelo. Aquí se analizan las características específicas de las funciones de activación tanh y ReLU en el contexto de Reservoir Computing:

### Tanh (Tangente Hiperbólica)

**Ventajas:**
- Proporciona una no linealidad suave que ayuda a capturar relaciones complejas en los datos temporales.
- La salida de tanh está acotada en el rango \([-1, 1]\), lo que puede ayudar a mantener la estabilidad dinámica del reservorio.
- La tangente hiperbólica facilita la generación de dinámicas no lineales, lo que es esencial para la memoria a corto plazo y la representación de patrones temporales en Reservoir Computing.

**Desventajas:**
- La función tanh puede sufrir de saturación en los extremos, lo que puede afectar la capacidad del reservorio para capturar la variabilidad en los datos.
- La tangente hiperbólica no permite la activación dispersa, lo que podría limitar la capacidad del modelo para manejar datos con patrones dispersos.

### ReLU (Unidad Lineal Rectificada)

**Ventajas:**
- La ReLU proporciona una activación no lineal en la región positiva, lo que puede mejorar la capacidad del reservorio para capturar señales con variabilidad dinámica.
- La función ReLU permite la activación dispersa, lo que podría ser beneficioso para reducir la redundancia en los datos de entrada y mejorar la eficiencia computacional del modelo.
- Al no sufrir de saturación en los extremos, la ReLU podría ser más efectiva en la representación de señales con amplitudes variables.

**Desventajas:**
- La ReLU puede causar el fenómeno de "neuronas muertas" (dead neurons) donde las unidades de activación permanecen inactivas para entradas negativas, lo que podría afectar la dinámica del reservorio.
- Algunos estudios han sugerido que la ReLU puede ser más sensible al ruido en comparación con la tangente hiperbólica, lo que podría impactar la robustez del modelo en entornos con alta variabilidad.

La elección entre tanh y ReLU en Reservoir Computing depende del contexto específico del problema, la naturaleza de los datos y las características del reservorio. Es crucial realizar experimentos y ajustar los parámetros del modelo para optimizar el rendimiento en la tarea deseada.

## Capa de Salida (Readout)

La capa de salida está formada por neuronas entrenables que reciben la información del reservorio y generan las predicciones o clasificaciones finales.

$$
y(t) = W_{out} r(t) + b_{out}
$$

donde:
- \( y(t) \) es el vector de salida en el tiempo \( t \).
- \( W_{out} \) es la matriz de pesos de la capa de salida.
- \( b_{out} \) es el vector de sesgo de la capa de salida.

A diferencia del reservorio, los pesos \( W_{out} \) en la capa de salida son entrenables. Este entrenamiento se realiza típicamente usando métodos de regresión lineal, lo cual es computacionalmente eficiente. La capa de salida convierte la representación interna de las señales EEG, mantenida en el reservorio, en una predicción o clasificación interpretable. En los siguientes apartados desarrollaremos esta capa de salida más en profundidad.

## Justificación de la Arquitectura

La arquitectura del RC ofrece varias ventajas específicas:

- **Simplicidad y Eficiencia:** La capa de entrada y el reservorio no requieren entrenamiento, lo que reduce significativamente la complejidad computacional. Solo la capa de salida es entrenable, lo que simplifica el proceso de ajuste del modelo.
- **Captura de Dinámica Temporal:** Gracias a las conexiones recurrentes y de retroalimentación en el reservorio, el RC puede capturar y modelar eficazmente la dinámica temporal de las señales EEG, que es crucial para comprender los patrones de actividad cerebral.
- **Proyección a Espacios de Mayor Dimensión:** La proyección de las señales de entrada a un espacio de mayor dimensión en el reservorio facilita la separación de características complejas y no lineales, mejorando la capacidad del modelo para distinguir entre diferentes patrones de señal.
- **Eficiencia en la Extracción de Características:** La capacidad del RC para trabajar con las señales en bruto, sin necesidad de una extracción de características previa, permite una modelización más directa y potencialmente más precisa de las señales EEG.
  
# Estudio Interno de Hiperparámetros
## Hiperparámetros en las Ecuaciones Diferenciales del Reservoir Computing

### 1. Cantidad de Neuronas en el Reservorio (\(N\))

La cantidad de neuronas en el reservorio afecta la dimensionalidad del estado del reservorio \(x(t)\) en las ecuaciones diferenciales del RC. Matemáticamente, podemos representarlo como:

$$
x(t) \in \mathbb{R}^N
$$

Donde \(N\) es el número de neuronas en el reservorio. Un mayor número de neuronas puede aumentar la capacidad del reservorio para capturar características complejas, pero también puede incrementar la complejidad computacional.

### 2. Conectividad del Reservorio (\(W_{res}\) y \(W_{fb}\))

La conectividad del reservorio influye en las matrices de peso \(W_{res}\) y \(W_{fb}\) en las ecuaciones diferenciales del RC. Estas matrices definen cómo las neuronas están conectadas entre sí. Podemos expresarlo como:

$$
W_{res} \text{ y } W_{fb} \text{ dependen de la conectividad del reservorio}
$$

Donde \(W_{res}\) es la matriz de pesos que representa las conexiones entre las neuronas dentro del reservorio, y \(W_{fb}\) es la matriz de pesos que representa las conexiones de retroalimentación. La densidad de estas conexiones puede afectar la capacidad del reservorio para almacenar y procesar información temporal.

### 3. Función de Activación (\(f(\cdot)\))

La función de activación \(f(\cdot)\) controla la no linealidad de las dinámicas del reservorio en las ecuaciones diferenciales del RC. Podemos relacionarlo como:

$$
\text{La elección de } f(\cdot) \text{ afecta la transformación de la entrada en la salida del reservorio}
$$

Funciones de activación comunes incluyen la tangente hiperbólica (\(\tanh\)) y la Unidad Lineal Rectificada (ReLU). La elección de la función de activación puede influir significativamente en el comportamiento dinámico del reservorio.

### 4. Parámetros de Regularización

Los parámetros de regularización ayudan a controlar la complejidad del modelo y pueden influir en las matrices de peso del reservorio y de salida en las ecuaciones de entrenamiento del RC. Podemos expresarlo como:

$$
\text{Los parámetros de regularización influyen en } W_{res} \text{ y } W_{out}
$$

Donde \(W_{out}\) es la matriz de pesos que transforma las actividades del reservorio en la salida del sistema. La regularización puede incluir términos como la norma \(L2\) para prevenir el sobreajuste y mejorar la generalización del modelo.

### 5. Parámetro de Fuga (\(\alpha\))

El parámetro de fuga controla la tasa a la que la actividad de las neuronas en el reservorio decae con el tiempo. Matemáticamente, este parámetro se incluye en las ecuaciones de estado de las neuronas recurrentes como un término de fuga. Se puede expresar como:

$$
\frac{dx_i(t)}{dt} = -\alpha x_i(t) + \sum_j W_{ij} f(x_j(t)) + I_i(t)
$$

Donde \(\alpha\) representa el parámetro de fuga. Este parámetro determina cuánto de la información pasada se retiene en cada actualización, afectando la capacidad del reservorio para manejar dependencias temporales largas.

### 6. Parámetro de Ruido (\(\sigma\))

El parámetro de ruido introduce una componente estocástica en las ecuaciones de estado de las neuronas, lo que puede ayudar a mejorar la capacidad del modelo para generalizar a nuevos datos y prevenir el sobreajuste. Matemáticamente, este parámetro se agrega como una fuente de ruido en las ecuaciones de estado de las neuronas, y puede tener la forma de una señal de ruido blanco gaussiano. La ecuación de estado modificada con el parámetro de ruido se ve así:

$$
\frac{dx_i(t)}{dt} = -x_i(t) + \sum_j W_{ij} f(x_j(t)) + I_i(t) + \sigma \xi(t)
$$

Donde \(\sigma\) representa el parámetro de ruido y \(\xi(t)\) es una variable aleatoria que sigue una distribución gaussiana con media cero y varianza unitaria.

---

Los parámetros de fuga y el parámetro de ruido son importantes para ajustar la dinámica del reservorio y la capacidad del modelo para manejar la incertidumbre en los datos de entrada. Su inclusión en las ecuaciones diferenciales del Reservoir Computing amplía la capacidad del modelo para capturar la complejidad de los datos y mejorar el rendimiento general del sistema.

### Consideraciones Adicionales

#### Escalado de Entradas

Es crucial escalar las entradas al reservorio para asegurar que las señales sean adecuadamente procesadas por las funciones de activación. Un escalado inapropiado puede llevar a saturación en las funciones de activación o a una dinámica ineficiente en el reservorio.

#### Estabilidad del Reservorio

Para garantizar que el reservorio no se vuelva caótico y mantenga una dinámica estable, es importante ajustar los hiperparámetros como la matriz de pesos \(W_{res}\) y los parámetros de regularización. Mantener el espectro de la matriz \(W_{res}\) dentro de ciertos límites (como un radio espectral menor que uno) puede ayudar a mantener la estabilidad.

#### Optimización de Hiperparámetros

La selección óptima de hiperparámetros requiere una validación cuidadosa y posiblemente el uso de técnicas de optimización, como la búsqueda en rejilla o la optimización bayesiana, para encontrar el conjunto de parámetros que maximicen el rendimiento del modelo en la tarea específica.

---

Al comprender y ajustar estos hiperparámetros, se puede optimizar el rendimiento del Reservoir Computing en diversas aplicaciones, como el análisis de señales EEG, mejorando tanto la precisión como la eficiencia del modelo.
## Datos Sintéticos

Los datos sintéticos son generados artificialmente para simular situaciones específicas o para llenar lagunas en conjuntos de datos reales. En esta sección, se explica cómo se generaron datos sintéticos para simular señales temporales de los diferentes electrodos en cada EEG de cada sujeto. Se describen los métodos y algoritmos utilizados para generar estos datos sintéticos y cómo se ajustan a las características de los datos reales, incluyendo la adición de ruido para simular condiciones más realistas.

### Generación de Datos Sintéticos de EEG

Para generar datos sintéticos de EEG que simulen las señales de jóvenes adultos y adultos mayores, se utilizaron varios métodos basados en modelos generativos y técnicas de simulación. Los pasos clave en la generación incluyen:

1. **Modelo Generativo de Señales EEG**: Utilización de un modelo generativo basado en procesos estocásticos para simular la actividad cerebral, imitando la dinámica temporal y espacial observada en señales EEG reales mediante modelos de autorregresión (AR) y procesos de Gauss.
   
2. **Simulación de Diferentes Electrodos**: Generación de múltiples canales de datos sintéticos con características específicas para cada región del cerebro, reflejando la diversidad de la actividad cerebral en diferentes áreas.

3. **Incorporación de Patrones Específicos de Edad**: Diferenciación entre jóvenes adultos y adultos mayores mediante la incorporación de patrones específicos en las señales sintéticas. Los jóvenes adultos presentan ondas alfa (8-12 Hz) y beta (13-30 Hz) más prominentes, mientras que los adultos mayores muestran una disminución en estas bandas y un incremento en las ondas delta (0.5-4 Hz) y theta (4-7 Hz).

4. **Adición de Ruido Realista**: Inclusión de ruido en las señales sintéticas para simular condiciones más realistas, como artefactos comunes en las señales EEG (interferencias electromagnéticas, movimientos oculares y actividad muscular).

5. **Validación de Datos Sintéticos**: Validación de los datos sintéticos generados comparándolos con datos reales para asegurar que las características principales de las señales EEG se mantengan. Se emplearon técnicas de análisis estadístico y visualización para verificar la similitud.

### Métodos y Algoritmos Utilizados

En la generación de datos sintéticos de EEG, se utilizaron los siguientes métodos y algoritmos:

- **Modelos Autoregresivos (AR)**: Para generar series temporales que imitan la dinámica de las señales EEG, utilizando un modelo AR de orden 2.
  
- **Procesos de Gauss**: Para modelar la correlación espacial y temporal en las señales EEG, generando datos con una estructura de correlación similar a la observada en las señales reales.
  
- **Simulación de Artefactos**: Implementación de algoritmos específicos para simular artefactos comunes en las señales EEG, como movimientos oculares y actividad muscular.
  
- **Ajuste de Parámetros**: Ajuste de los parámetros de los modelos generativos para asegurar que las señales sintéticas tengan características similares a las señales EEG reales.

### Generación de Datos Sintéticos de EEG: Enfoque Basado en Bandas de Frecuencia

Además de los métodos anteriores, se desarrolló un enfoque basado en la combinación de señales sinusoidales en distintas bandas de frecuencia para generar datos que simulen de manera realista las señales de EEG de jóvenes adultos y adultos mayores. Las bandas de frecuencia consideradas son: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz) y Gamma (30-100 Hz).

- **Definición de Bandas de Frecuencia**: Creación de un diccionario de bandas de frecuencia que contiene las bandas asociadas con diferentes tipos de ondas cerebrales.
  
- **Generación de Señales Sinusoidales con Ruido para EEG**: Inicialización de un arreglo de datos para almacenar los datos sintéticos generados, representando las señales EEG para cada sujeto y grupo.
  
- **Generación de Señales EEG para Cada Sujeto y Grupo**: Iteración sobre cada grupo de sujetos (jóvenes adultos y adultos mayores) para generar señales EEG con características típicas de cada grupo. Para jóvenes, se simulan variaciones aleatorias en frecuencia y amplitud; para adultos mayores, se reflejan cambios asociados con el envejecimiento.
  
- **Adición de Ruido**: Agregación de ruido gaussiano a las señales generadas para simular condiciones más realistas.

Este enfoque asegura que los datos sintéticos sean representativos de las señales EEG reales y puedan utilizarse para estudios comparativos y validación de modelos en análisis de señales EEG.


# Resultados

Los resultados de las evaluaciones se guardan en la carpeta results/.

# Contacto

Para cualquier consulta, contacta a jogugil@gmail.com/jogugil@alumni.uv.es

