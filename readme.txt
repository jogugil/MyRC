Instrucciones de instalación y uso

1. Instalación de dependencias

	Para ejecutar el algoritmo genético y otros scripts relacionados, primero debes instalar las siguientes bibliotecas de Python. Puedes hacerlo ejecutando el siguiente comando en tu terminal:
		```
			pip install -r requirements.txt
		```
	Asegúrate de estar en el directorio que contiene el archivo requirements.txt antes de ejecutar el comando.

	Si no tienes instaldo pip en el ssitema puedes utilizar conda: Para instalar las bibliotecas Python necesarias, puedes utilizar Conda. Si no tienes instalado pip en tu sistema, puedes probar con Conda siguiendo estos pasos:

		1. **Crear un entorno Conda (opcional)**: Si lo deseas, puedes crear un entorno Conda antes de instalar las bibliotecas. Esto te permite aislar las bibliotecas del resto del sistema. Puedes crear un entorno con el siguiente comando:

		    ```
    			conda create --name mi_entorno
    		    ```

    			Reemplaza `mi_entorno` con el nombre que desees para tu entorno.

		2. **Activar el entorno Conda (opcional)**: Si creaste un entorno Conda en el paso anterior, actívalo con el siguiente comando:

		    ```
    			conda activate mi_entorno
    		    ```

		    Nuevamente, reemplaza `mi_entorno` con el nombre de tu entorno.

		3. **Instalar las bibliotecas con Conda**: Utiliza el siguiente comando para instalar las bibliotecas especificadas en tu archivo `requirements.txt`:

		    ```
    			conda install --file requirements.txt
   		    ```

		4. **Esperar a que se completen las instalaciones**: Conda descargará e instalará todas las bibliotecas especificadas en el archivo `requirements.txt`.

		Una vez completados estos pasos, todas las bibliotecas especificadas en tu archivo `requirements.txt` estarán instaladas y podrás usarlas en tu proyecto de Python.

2. Uso del algoritmo genético

	Una vez que todas las dependencias estén instaladas, puedes ejecutar el script del algoritmo genético desde la línea de comandos. El script acepta varios parámetros para configurar el algoritmo. A continuación, se muestra un ejemplo de cómo puedes llamar al script:
		```
			python algoritmo_genetico.py --ICAflag --population 20 --max_generations 10 --direct_Younger "./Younger" --direct_Older "./Older" --n_y_subject 12 --n_o_subject 12 --mutpb 0.2
		```
o
		```
			python3 algoritmo_genetico.py --ICAflag --population 20 --max_generations 10 --direct_Younger "./Younger" --direct_Older "./Older" --n_y_subject 12 --n_o_subject 12 --mutpb 0.2
		```
Y en segundo plano:
		```
			python3 algoritmo_genetico.py --ICAflag --population 20 --max_generations 10 --direct_Younger "./Younger" --direct_Older "./Older" --n_y_subject 12 --n_o_subject 12 --mutpb 0.2 > output_ga00.log 2>&1 &
		```
	Este comando ejecutará el algoritmo genético con las siguientes configuraciones:

	- ICAflag: Si se añade el flag es true e indica que se debe realizar la eliminación de artefactos.
	- population: Tamaño de la población en cada generación.
	- max_generations: Número máximo de generaciones para ejecutar el algoritmo.
	- direct_Younger: Ruta al directorio que contiene los datos de los sujetos más jóvenes.
	- direct_Older: Ruta al directorio que contiene los datos de los sujetos más mayores.
	- n_y_subject: Número de sujetos más jóvenes.
	- n_o_subject: Número de sujetos más mayores.
	- mutpb: Probabilidad de mutación de los hiperparámetros en cada individuo.

	Ajusta los valores de estos parámetros según tus necesidades.


Se tiene que instalar cuda y cudatoolkit para trabajar con torch:

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt-get install cuda
echo 'export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
nvcc --version

