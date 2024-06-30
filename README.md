# MyRC And MyESN Project (Reservoir Computing ESN)


Este proyecto implementa un modelo de Reservoir Computing Echo State Network (ESN) para su estudio e implementación. Se ha creado una API para evaluar el uso de este modelo en el procesamiento de señales temporales presentes en los canales de un EEG. El objetivo es la reconstrucción y predicción de señales, la obtención no supervisada de estados neuronales, y la clasificación de tipos de sujetos según los patrones de la dinámica temporal que conserva el estado final del RC para cada sujeto. En concreto, procesamos diferentes sujetos que se clasifican en jóvenes adultos y mayores.

![imagen](https://github.com/jogugil/MyRC/assets/15160072/63c28d95-1af2-46b4-87f2-736ee564df76)

 

# Generación de Datos Sintéticos

Para llevar a cabo los experimentos, primero generamos datos sintéticos que simulan los EEG de diferentes sujetos, creando dos poblaciones distintas: una de jóvenes adultos y otra de mayores. Se consideraron las ondas cerebrales típicas en este tipo de señales, diferenciando su magnitud frecuencial y amplitud según sea un joven adulto o una persona mayor.
# Generación de Datos Sintéticos

Para llevar a cabo los experimentos, primero generamos datos sintéticos que simulan los EEG de diferentes sujetos, creando dos poblaciones distintas: una de jóvenes adultos y otra de mayores. La generación de estos datos se basa en modelos estocásticos y técnicas de simulación que permiten reproducir características clave de las señales EEG reales.

### Modelo Generativo Estocástico

El proceso de generación de datos sintéticos de EEG que se ha utilizado (función 'generate_synthetic_eeg_data' del módulo 'eeg.py') sigue los siguientes pasos clave:

1. **Bandas de Frecuencia**: Se simulan las cinco bandas de frecuencia típicas de las señales EEG: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz) y Gamma (30-100 Hz). Cada banda tiene diferentes implicaciones neurológicas y se comporta de manera distinta en sujetos jóvenes versus mayores.

2. **Diferencias entre Grupos**: Para diferenciar las señales de los jóvenes y los mayores, se asignan características específicas a cada grupo:
    - **Sujetos Jóvenes**: Estos sujetos muestran un pico de amplitud en la banda Beta, lo cual se asocia con una mayor actividad cognitiva y vigilia. Las señales tienen mayor amplitud y menos ruido.
    - **Sujetos Mayores**: En contraste, las señales de los mayores son de menor amplitud y presentan más ruido, reflejando la disminución en la actividad cerebral y el incremento de la variabilidad neuronal con la edad.

3. **Dinámica Temporal**:
    - **Proceso Autoregresivo**: Para capturar la naturaleza temporal de las señales EEG, se agrega un proceso autoregresivo. Este modelo ayuda a imitar las dependencias temporales presentes en las señales EEG reales.
    - **Proceso Gaussiano**: Además, se incorpora un proceso gaussiano para introducir variabilidad adicional y complejidad en las señales, ayudando a simular condiciones más realistas.

4. **Interacción entre Canales**:
    - Las señales EEG son multivariantes y presentan interacciones entre diferentes canales (electrodos). Para simular esto, las señales generadas para cada canal incluyen componentes de ruido y variabilidad compartida, reflejando la correlación natural entre diferentes regiones del cerebro.

5. **Ruido Gaussiano**: Finalmente, se añade ruido gaussiano a las señales para imitar las condiciones de grabación reales, donde siempre existe una cierta cantidad de ruido de fondo.

### Justificación del Proceso

La elección de este proceso de generación de datos se basa en varios aspectos críticos:

- **Realismo**: Al incorporar bandas de frecuencia específicas, procesos autoregresivos y gaussianos, y ruido gaussiano, se busca que las señales sintéticas sean lo más realistas posible, permitiendo que los experimentos y análisis realizados sobre estos datos sean relevantes y aplicables a señales EEG reales.

- **Diferenciación entre Grupos**: Las diferencias claras entre las señales de los jóvenes y los mayores permiten estudiar y validar técnicas de clasificación y análisis, asegurando que las características distintivas de cada grupo sean detectables y medibles.

- **Simulación Completa**: La inclusión de dinámicas temporales y la interacción entre canales asegura que las señales generadas no solo sean realistas en términos de frecuencia y amplitud, sino también en su comportamiento temporal y multicanal, lo cual es crucial para cualquier análisis de EEG.

*Este modelo generativo estocástico ofrece una manera robusta de simular señales EEG realistas, proporcionando una base sólida para llevar a cabo experimentos controlados y análisis detallados de las diferencias entre poblaciones de jóvenes adultos y mayores.*

### Utilización de Deep Learning para la Generación de Señales EEG

La generación de señales EEG multivariantes a través de técnicas de deep learning ha ganado atención en los últimos años debido a su capacidad para modelar la complejidad y la estructura temporal de las señales. Modelos como TimeGAN y cosci-GAN son ejemplos prominentes en este campo.

#### TimeGAN

[TimeGAN](https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) (Time-series Generative Adversarial Network) es un modelo avanzado que combina las capacidades de los modelos generativos adversariales (GAN) con la estructura temporal inherente a las series de tiempo. Este modelo no solo se enfoca en generar datos realistas, sino que también mantiene la coherencia temporal de las señales, lo cual es crucial para los datos EEG.

#### cosci.GAN

[cosci.GAN](https://openreview.net/pdf?id=RP1CtZhEmR) es otro ejemplo de aplicación de GANs para la generación de señales temporales multiovariantes. Este modelo ha demostrado ser efectivo en la generación de datos complejos y multivariantes, como las señales EEG, al capturar la dinámica temporal y la interacción entre múltiples canales.

#### Eficacia y Limitaciones

Aunque estos métodos basados en deep learning han mostrado resultados prometedores, su implementación y entrenamiento pueden ser computacionalmente intensivos y requieren grandes volúmenes de datos para lograr un rendimiento óptimo. Además, la interpretación de los modelos generados y la garantía de su realismo sigue siendo un desafío.

### Justificación del Proceso Utilizado Frente al Uso de TimeGAN o cosci.GAN

Aunque los métodos de deep learning como TimeGAN y cosci.GAN tienen el potencial de generar señales EEG multivariantes realistas, hemos optado por un modelo generativo estocástico por varias razones:

1. **Simplicidad y Eficiencia**: El modelo estocástico utilizado es más sencillo y menos exigente computacionalmente en comparación con los métodos de deep learning. Esto permite una generación de datos más rápida y accesible sin la necesidad de grandes infraestructuras de cómputo.

2. **Objetivo del Proyecto**: El objetivo principal del proyecto es el uso de Reservoir Computing para la obtención de la dinámica temporal de las señales y los patrones distintivos entre jóvenes adultos y mayores en reposo. El modelo generativo estocástico nos proporciona un control más directo sobre las características de las señales generadas, asegurando que las diferencias entre los grupos sean claramente definidas y mensurables.

3. **Interpretabilidad**: Los modelos estocásticos permiten una mayor interpretabilidad de los resultados, ya que se puede rastrear y ajustar fácilmente cada componente del proceso de generación. Esto es particularmente útil para validar y ajustar las señales sintéticas de acuerdo a los conocimientos previos sobre las características de las señales EEG reales.

Por ello, aunque los modelos de deep learning como TimeGAN y cosci.GAN son herramientas poderosas para la generación de datos, el enfoque estocástico seleccionado proporciona una solución más práctica y adecuada para los objetivos específicos de nuestro proyecto.

# Evaluación con Datos Reales

Después de implementar y probar el modelo con datos sintéticos, se probó con un banco de datos reales de diferentes sujetos. Por motivos de privacidad, estos datos reales no se han subido, pero se incluyen los notebooks y scripts utilizados para el procesamiento de dichos datos.
Como no se añaden los datos raw orifinales, podriamos generar datos sintéticos con los canales que contienen los datos reales como son:
## Todos los Canales
```python
all_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5',
                'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
                'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
                'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4',
                'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
                'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
                'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8',
                'PO4', 'O2', 'UP', 'DOWN', 'LEFT', 'RIGHT',
                'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']

eeg_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

eog_channels = ["UP", "DOWN", "LEFT","RIGHT"]
exg_channels = ['EXG5', 'EXG6', 'EXG7', 'EXG8']
```
# Procesamiento de Señales EEG

Las señales EEG procesadas en este proyecto permiten la reconstrucción y predicción de patrones temporales. Además, se utiliza tanto el aprendizaje supervisado como no supervisado para extraer características significativas y realizar la clasificación de sujetos.

Como preprocesamiento de los datos aplicamos un filtro paso banda que mantenga las frecuencias de las ondas cerebrales (0.2-30)Khz, dejando fuera las ondas Gamma que llegan hasta 100Khz. 

![image](https://github.com/jogugil/MyRC/assets/15160072/16efe94c-a71b-486e-a973-7c53004b8168)


Despues del filtrado que permite eliminar ruido y artefactos de alta frecuencia, además de los artefactos asociados a la corriente eléctrica (50khz y sus armónicos), podemos aplicar un proceso de eliminación de artefactos mediante un modelo ICA: 

#### Diagrama de Proceso para la Función `remove_ica_components_artifact`

##### Descripción General

La función `remove_ica_components_artifact` se utiliza para eliminar componentes ICA identificados como artefactos en datos EEG. A continuación se detalla el proceso paso a paso.

***Pasos del Proceso***

1. **Copiar Datos EEG**
  
2. **Inicializar ICA**

3. **Ajustar ICA a los Datos**

4. **Aplicar ICA y Rechazar Segmentos**
   - Eliminamos los artefactos asociados a los movimientos oculares, cardiacos y/o musculares

5. **Detectar y Excluir Componentes ICA de Alta Varianza**
   - Identificar componentes ICA con alta varianza, amplitud y picos altos.

6. **Excluir Componentes ICA  mediante histograma**
   - Identificar componentes  ICA excluyendo los componentes cuya densidad del histograma de amplitudes no se aproxima a una normal. Se ha estudiado que este tipo de señales mantiene una distribución normal. Los componentes que no mantienen esta distribución son aquellas que aportan artefactos con picos de señal con amplitudes extradamente elevados.
     ![image](https://github.com/jogugil/MyRC/assets/15160072/41126ac3-db67-402b-8ff4-97a340745e92)

![image](https://github.com/jogugil/MyRC/assets/15160072/d0a9fce4-eb93-463f-b9e1-ee9534341c55)

7. **Actualizar Datos Filtrados y con limpieza de artefactos**


Después de eliminar o no los artefactos (será algo opcional) debemos crear una matriz tridimensional {número sujetos, tamaño señales, nñumero canales}. Creando series temporales multivariante por sujeto. Fijamos para todos los sujetos y canales el mismo tamaño de señal.

Y finalmente lo ideal es normalizar los datos bien mediante una estandarización o una normalización (min-max).
 
# API Construida

Se construyó una API configurada mediante un diccionario config. Este diccionario contiene diferentes parámetros que se transforman en hiperparámetros para el modelo, permitiendo una fácil personalización y ajuste del modelo a diferentes necesidades experimentales.

# Hiperparámetros del Modelo implementado: 

## Ejemplo de Diccionario de Configuración para los hiperparámetros:

      config = {
              'seed': 1,
              'init_type': 'rand',
              'init_std': 0.01,
              'init_mean': 0,
              'input_size': 17,
              'input_scaling': 1.5,
              'w_l2': 0.0005,
              'n_internal_units': 170, #  700,
              'spectral_radius': 0.85,
              'leak': 0.65,
              'nonlinearity': 'relu',
              'connectivity': 0.35,
              'noise_level': 0.1,
              'n_drop': None,
              'washout': 'init',
              'bidir': False,
              'dimred_method': 'tenpca',
              'n_dim': 40,
              'mts_rep': 'reservoir',
              'w_ridge_embedding': 10.0,
              'circle': False,
              'plasticity_synaptic': None,
              'theta_m': 0.01,
              'plasticity_intrinsic': None,
              'learning_rate': 0.9,
              'new_activation_function': 'relu',
              'excitability_factor': 0.01,
              'use_input_bias': True,
              'use_input_layer': True,
              'readout_type': None,
              'threshold': 0.5,
              'svm_kernel': 'linear',
              'svm_gamma': 0.005,
              'svm_C': 5.0,
              'w_ridge': 5.0,
              'num_epochs': 2000,
              'mlp_layout': (100, 100),
              'mlp_batch_size': 32,
              'mlp_learning_rate': 0.01,
              'mlp_learning_rate_type': 'constant',
              'device': 'cpu'
      }

## Comentarios sobre los Parámetros de Configuración
  
  ### 1. `seed`
  - **Descripción**: Semilla para la inicialización de números aleatorios.
  - **Posibles Valores**: Entero (e.g., `1`, `42`).
  - **Comentario**: Útil para asegurar la reproducibilidad de los experimentos.
  
  ### 2. `init_type`
  - **Descripción**: Tipo de inicialización de los pesos.
  - **Posibles Valores**: `'orthogonal'`, `'uniform'`, `'normal'`.
  - **Comentario**: Diferentes métodos de inicialización pueden afectar la convergencia del modelo.
  
  ### 3. `init_std`
  - **Descripción**: Desviación estándar para la inicialización normal de los pesos.
  - **Posibles Valores**: Flotante (e.g., `0.01`).
  - **Comentario**: Controla la magnitud inicial de los pesos cuando se utiliza inicialización normal.
  
  ### 4. `init_mean`
  - **Descripción**: Media para la inicialización normal de los pesos.
  - **Posibles Valores**: Flotante (e.g., `0`).
  - **Comentario**: Establece la media de los pesos iniciales en una distribución normal.
  
  ### 5. `input_size`
  - **Descripción**: Tamaño de la entrada.
  - **Posibles Valores**: Entero (e.g., `10`).
  - **Comentario**: Define la dimensionalidad de los datos de entrada.
  
  ### 6. `n_internal_units`
  - **Descripción**: Tamaño del reservorio, indicando el número de unidades internas o nodos en el reservorio.
  - **Posibles Valores**: Entero (e.g., `480`).
  - **Comentario**: Un número mayor de unidades internas puede aumentar la capacidad del modelo para capturar dinámicas complejas, pero también incrementa el costo computacional.
  
  ### 7. `spectral_radius`
  - **Descripción**: El mayor valor propio del reservorio.
  - **Posibles Valores**: Flotante (e.g., `0.59`).
  - **Comentario**: Es una medida de la dispersión de las activaciones en el reservorio y puede afectar las propiedades dinámicas del reservorio.
  
  ### 8. `leak`
  - **Descripción**: Cantidad de fuga en la actualización del estado del reservorio.
  - **Posibles Valores**: Flotante entre `0` y `1` (e.g., `0.4`).
  - **Comentario**: Controla cuánta información del estado anterior del reservorio se mantiene en cada iteración. Un valor menor que 1 introduce un decaimiento en el estado del reservorio, afectando la dinámica y la capacidad de memoria del reservorio.
  
  ### 9. `connectivity`
  - **Descripción**: Porcentaje de conexiones no nulas en el reservorio.
  - **Posibles Valores**: Flotante entre `0` y `1` (e.g., `0.6`).
  - **Comentario**: Controla la dispersión del reservorio.
  
  ### 10. `input_scaling`
  - **Descripción**: Escala de los pesos de entrada.
  - **Posibles Valores**: Flotante (e.g., `0.1`).
  - **Comentario**: Controla la fuerza de la señal de entrada.
  
  ### 11. `noise_level`
  - **Descripción**: Nivel de ruido en la actualización del estado del reservorio.
  - **Posibles Valores**: Flotante (e.g., `0.1`).
  - **Comentario**: Introduce perturbaciones aleatorias en la dinámica del reservorio.
  
  ### 12. `n_drop`
  - **Descripción**: Estados transitorios a eliminar.
  - **Posibles Valores**: Entero (e.g., `100`).
  - **Comentario**: Los estados iniciales del reservorio a menudo están afectados por las condiciones iniciales y pueden no ser representativos de la verdadera dinámica del reservorio.
  
  ### 13. `washout`
  - **Descripción**: Método de lavado del reservorio.
  - **Posibles Valores**: `'init'`, `'rand'`.
  - **Comentario**: Controla cómo se maneja el estado inicial del reservorio.
  
  ### 14. `use_input_bias`
  - **Descripción**: Si se usa sesgo en la capa de entrada.
  - **Posibles Valores**: `True`, `False`.
  - **Comentario**


# Estructura del Proyecto

    synthetic_eeg_v10.ipynb: Notebook donde se implementan datos sinteticos EEG y se utiliza el modelo MyRC para sus procesamiento.
                . Reconstrucciíon y predicción de señales temporales de los datos sintetigcos EEG
                . Clustering y clasifiación por métodos no supervisados de los datos sinteticos EEG
                . Clasificación por mñetodos supervisados de los datos sinteticos EEG
    data/: Carpeta con datos de prueba y entrenamiento.No se tienen por temas de privacidad
    base/: librerias base donde seimplementa el modelo RC ESN y otra funciones auxiliares.
    README.md: Descripción del proyecto.EN CREACIÓN
    presentation/: Presentación del proyecto.EN CREACIÓN

# Instalación

Para instalar las dependencias del proyecto, ejecuta:
      
      pip install -r requirements.txt
Además hay que tener en cuena que se debe instarlar torch, dependerá del sistema operativo :
   https://pytorch.org/get-started/locally/
   
# Uso

Puedes utilizar tanto el código que hay en los notebooks como los diferentes scripts existentes en el directorio para realizar la evaluación.
Para ejecutar la evaluación, usa el siguiente comando:


## Características de las Señales EEG

Las señales de electroencefalografía (EEG) son una representación directa de la actividad eléctrica del cerebro. Estas señales se generan principalmente por los potenciales postsinápticos de las neuronas piramidales en la corteza cerebral, que crean campos eléctricos detectables en el cuero cabelludo. El EEG es una herramienta no invasiva que permite registrar estas señales a través de electrodos colocados en la superficie del cuero cabelludo. 

![imagen](https://github.com/jogugil/MyRC/assets/15160072/64fd9a4a-99d1-4904-a285-3c9548122e8a)


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

![imagen](https://github.com/jogugil/MyRC/assets/15160072/715a2b6b-d36f-4021-b083-5def6ae65973)


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

![imagen](https://github.com/jogugil/MyRC/assets/15160072/d1687db7-6f64-40a6-aad9-8aff465bbe56)


La arquitectura del RC ofrece varias ventajas específicas:

- **Simplicidad y Eficiencia:** La capa de entrada y el reservorio no requieren entrenamiento, lo que reduce significativamente la complejidad computacional. Solo la capa de salida es entrenable, lo que simplifica el proceso de ajuste del modelo.
- **Captura de Dinámica Temporal:** Gracias a las conexiones recurrentes y de retroalimentación en el reservorio, el RC puede capturar y modelar eficazmente la dinámica temporal de las señales EEG, que es crucial para comprender los patrones de actividad cerebral.
- **Proyección a Espacios de Mayor Dimensión:** La proyección de las señales de entrada a un espacio de mayor dimensión en el reservorio facilita la separación de características complejas y no lineales, mejorando la capacidad del modelo para distinguir entre diferentes patrones de señal.
- **Eficiencia en la Extracción de Características:** La capacidad del RC para trabajar con las señales en bruto, sin necesidad de una extracción de características previa, permite una modelización más directa y potencialmente más precisa de las señales EEG.
  
# Hiperparámetros
 
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

### Deep ESN

Se entrega un notebook que contiene una posible implementación de un modelo DeepESN con el modelo ESN implementado en el proyecto. Solo se implementa el modelo DeepESN pero no se ha utilizado en el proyecto ni se ha estudiado una posible optimización de la arquitectura. Se deja como trabajo futuro.

![imagen](https://github.com/jogugil/MyRC/assets/15160072/ba457b14-1ed1-4ddf-b2b3-4b8df825851c)

![imagen](https://github.com/jogugil/MyRC/assets/15160072/84ff6c2c-0926-4d35-a774-da1ecf57ecc3)


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

![imagen](https://github.com/jogugil/MyRC/assets/15160072/af2a6ab2-602c-4c89-9cb0-23203c2c4f17)

Los resultados de las evaluaciones se guardan en la carpeta results/.

# Contacto

Para cualquier consulta, contacta a jogugil@gmail.com/jogugil@alumni.uv.es

