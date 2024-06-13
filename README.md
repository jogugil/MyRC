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
