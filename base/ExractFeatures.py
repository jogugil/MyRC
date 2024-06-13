import pdb
import pywt
import pandas as pd
import numpy as np 

from collections import Counter
from base.Config import Config
from scipy.signal import welch
from scipy.stats import skew
from scipy.stats import kurtosis

from scipy.ndimage import convolve1d
from sklearn.feature_selection import RFE
from concurrent.futures import ThreadPoolExecutor
from statsmodels.tsa.ar_model import AutoReg
 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class ExractFeatures ():
    
    def __init__(self, Config):
        self.config      = Config
 
    def add_features (self, caracteristica):
        if isinstance (caracteristica, pd.DataFrame):
            if self.features_df.empty:
                self.features_df = caracteristica
            else:
                self.features_df = pd.concat ([self.features_df, caracteristica], axis = 1)
        elif isinstance (caracteristica, np.ndarray):
            # En este caso, no se necesita un nombre de columna
            # Agregamos el array como una nueva columna sin nombre
            col_name = f"Unnamed_{len (self.features_df.columns) + 1}"
            self.features_df[col_name] = caracteristica.T

    def add_features (self, nombre_columna, caracteristica):
        self.features_df [nombre_columna] = caracteristica
        
    def get_paper_features (self):
        return self.features_df
    
    # transformada wavelet discreta de superposición máxima (MODWT), que es una transformación invariante por desplazamiento
    # MODWT de profundidad 6 con orden cuatro (db4). se aplica a épocas cada 30 s con una frecuencia de muestreo de 200 Hz. (En nuestro caso 100HZ)
   
    def wavelet_transform (self, data, wavelet, channels, level = 3):
        '''
            Calcula la wavelet que se pasa por parámetro (wavelet). Esta trasnformada puede ser la swdt de la
            libreria pywavelet o modwdt que hemos creado nosotros.
        '''
        coeffs         = wavelet (data, 'db4', level = level, norm = True)
        coeffs_scale   = [level_coeffs [0] for level_coeffs in coeffs]
        coeffs_detalle = [level_coeffs [1] for level_coeffs in coeffs]

        energies_escala         = np.sum (np.square (coeffs_scale), axis = 2)
        energia_total_por_canal = np.sum (energies_escala, axis = 0)

        energies_detalle          = np.sum (np.square (coeffs_detalle), axis = 2)
        energia_total_por_detalle = np.sum (energies_detalle, axis = 0)

        porcentaje_energia_scale   = (energies_escala / energia_total_por_canal) * 100
        porcentaje_energia_detalle = (energies_detalle / energia_total_por_detalle) * 100
    
        means_scale          = np.mean (coeffs_scale, axis = 2)
        std_deviations_scale = np.std (coeffs_scale, axis = 2)

        means_detalle          = np.mean (coeffs_detalle, axis = 2)
        std_deviations_detalle = np.std (coeffs_detalle, axis = 2)

        eeg_resam_shape = data.shape[1]
        
        energies_escala            = energies_escala [:, :, :eeg_resam_shape]
        energies_detalle           = energies_detalle [:, :, :eeg_resam_shape]
        porcentaje_energia_scale   = porcentaje_energia_scale [:, :, :eeg_resam_shape]
        porcentaje_energia_detalle = porcentaje_energia_detalle [:, :, :eeg_resam_shape]
        means_scale                = means_scale [:, :, :eeg_resam_shape]
        std_deviations_scale       = std_deviations_scale [:, :, :eeg_resam_shape]
        means_detalle              = means_detalle [:, :, :eeg_resam_shape]
        std_deviations_detalle     = std_deviations_detalle [:, :, :eeg_resam_shape]

        resultados = {
            'energies_escala': energies_escala,
            'energia_total_por_canal': energia_total_por_canal,
            'energies_detalle': energies_detalle,
            'energia_total_por_detalle': energia_total_por_detalle,
            'porcentaje_energia_scale': porcentaje_energia_scale,
            'porcentaje_energia_detalle': porcentaje_energia_detalle,
            'means_scale': means_scale,
            'std_deviations_scale': std_deviations_scale,
            'means_detalle': means_detalle,
            'std_deviations_detalle': std_deviations_detalle
        }

        return self.wavelets_to_df (resultados, channels, level)
    
    def wavelets_to_df (self, res_swt, channels, level = 3):

        niveles = list(range(1, level + 1))

        variables = {
            'Energy': {'s': res_swt ['energies_escala'], 'd': res_swt ['energies_detalle']},
            'Percentage_energy': {'s': res_swt ['porcentaje_energia_scale'], 'd': res_swt ['porcentaje_energia_detalle']},
            'Mean': {'s': res_swt ['means_scale'], 'd': res_swt ['means_detalle']},
            'Std_deviation': {'s': res_swt ['std_deviations_scale'], 'd': res_swt ['std_deviations_detalle']}
        }

        col_name = []
        data_fe  = []
        for nivel in niveles:
            for canal,ixcn in  zip (channels,range(0, len(channels))):
                for variable, data in variables.items ():

                    col1 = variable + '_s_' + 'nivel-'+ str (nivel) + '_Canal-' + canal
                    col2 = variable + '_d_' + 'nivel-'+ str (nivel) + '_Canal-' + canal

                    col_name.append (col1)
                    col_name.append (col2)
                    data_fe.append (variables [variable]['s'][nivel - 1,:, ixcn])
                    data_fe.append (variables [variable]['d'][nivel - 1,:, ixcn])

        featur      = np.array (data_fe).T
        df_features = pd.DataFrame (featur, columns = col_name)
        
        return df_features
    
     # Amplitud pico a pico: se calcula la amplitud pico a pico P(X) por P (X) = max (X) - min (X)  donde X = {x1, x2 . . . , xN} el conjunto de amplitudes de la señal.
    def PeaktoPeak (self, psg_signal, channels):
        # Calcula la amplitud P-P para cada ventana y canal
        return self.PeaktoPeak_to_df (np.max (psg_signal, axis = 2) - np.min (psg_signal, axis = 2), channels)
    
    def PeaktoPeak_to_df (self, res_PeaktoPeak, channels):
        # Genera la lista de nombres de canales
        nombres_canales = ["P_P_Canal_" + str(i) for i in channels]

        return pd.DataFrame (res_PeaktoPeak, columns = nombres_canales)
    
    def Entropy (self, data, channels):
        # Calcula entropías utilizando operaciones matriciales
        # CUIDADO!!!!!! --123--el número de elementos que devuelve np.histogram para una ventana en particular no coincide con la longitud 
        # de las filas de data_ventanas. Esto se debe a que el número de bins generados automáticamente con bins='auto' puede variar para 
        # cada ventana, lo que resulta en diferentes longitudes de histograma.
        # Define una función para calcular histogramas ----> bins='auto'

        def calculate_histogram (x):
            hist, _ = np.histogram (x, bins = 80) # Como hemos comentado no ponemos bind = auto por osibles problemas de dimensiones
            #print (hist.shape)
            return hist

        # Calcula histogramas para cada ventana y canal
        hist = np.apply_along_axis(calculate_histogram, axis = 2, arr = data)

        p       = hist / np.sum (hist, axis = 2)[:, :, None]
        epsilon = 1e-10  # Pequeña constante para evitar el logaritmo de cero

        q = 2
        shannon_results = -np.sum (p * np.log (p + epsilon), axis = 2)
        renyi_results   = np.log (np.sum (p**q + epsilon, axis = 2)) / (1 - q)  # Reemplaza el valor q deseado
        tsallis_results = (np.sum (p**q + epsilon, axis = 2) - 1) / (q - 1)     # Reemplaza el valor q deseado
        
        res_entropy =shannon_results, renyi_results, tsallis_results
        
        return self.Entropy_to_df (res_entropy, channels) 

    def Entropy_to_df (self, res_entropy, channels):
        
        shannon_results, renyi_results, tsallis_results = res_entropy

        nombres_canales = ["shannon_Canal_" + str(i) for i in channels]
        df_features = pd.DataFrame (shannon_results, columns = nombres_canales)
          
        nombres_canales = ["renyi_Canal_" + str(i) for i in channels]
        df_features     = pd.concat ([df_features, pd.DataFrame (renyi_results, columns = nombres_canales)], axis = 1)


        # Generar la lista de nombres de canales
        nombres_canales = ["tsalli_Canal_" + str(i) for i in channels]
        df_features     = pd.concat ([df_features, pd.DataFrame (tsallis_results, columns = nombres_canales)], axis = 1)
        
        return df_features
        
    # Relative Spectral Power (RSP): Para cada subbanda de frecuencia, se calcula el RSP. Este parámetro viene dado por la relación entre
    # la potencia espectral de subbanda (BSP) y la potencia espectral total, es decir, la suma de las cinco subbandas BSP
    # as bandas espectrales Delta, Theta y Alfa se puede resaltar sobre bandas de onda lenta mediante lento índices de onda definidos por las siguientes relaciones
    # TSI, ASI (Agarwal & Gotman, 2001) y DSI representan thetaslow- índice de onda, índice de onda lenta alfa e índice de onda lenta delta, respectivamente.
    def RSP (self, psg_signal, channels):
        # Especificamos las bandas de frecuencia (en Hz)
        freq_bands = {
            'D1': (25, 50),
            'D2': (12.5, 25),
            'D3': (6.25, 12.5),
            'D4': (3.125, 6.25),
            'D5': (0, 3.125),
        }


        sf = 100
        # Calculamos la densidad espectral de potencia (PSD)
        frequencies, psd = welch (psg_signal, fs = sf, nperseg = sf * 2)

        # Inicializamos diccionarios para almacenar las potencias de las bandas de frecuencia
        band_powers = {}
        total_power = np.trapz (psd, frequencies, axis = 2)  # Potencia total

        for band, (low, high) in freq_bands.items ():
            band_indices      = (frequencies >= low) & (frequencies <= high)
            band_power        = np.trapz (psd[:, :, band_indices], frequencies [band_indices], axis = 2)
            band_powers[band] = band_power

        # Calculamos el RSP
        RSP = {band: band_power / total_power for band, band_power in band_powers.items()}
        
      
        
        # Calcula los índices DSI, TSI y ASI
        DSI = band_powers['D5'] / (band_powers['D4'] + band_powers['D3'])
        TSI = band_powers['D4'] / (band_powers['D1'] + band_powers['D3'])
        ASI = band_powers['D3'] / (band_powers['D1'] + band_powers['D4'])
        
        res_rsp = RSP, DSI, TSI, ASI
        
        return self.RSP_to_df (res_rsp, channels) 
    
    def RSP_to_df (self,res_rsp, channels):
        
        RSP, DSI, TSI, ASI = res_rsp
        
      
        
        col_name  = []
        data_fe   = []
        RSP_array = np.array (list(RSP.values()))
    
        for nivel in range (0, 5):
            for canal,ixcn in zip(channels, range (0, len(channels))):
                    col1 =  'RSP_D'+str (nivel) +  '_Canal-' + str (canal)

                    col_name.append (col1)
                    data_fe.append (RSP_array [nivel,:,ixcn])


        featur      = np.array (data_fe).T
        df_features = pd.DataFrame (featur, columns = col_name) 

        # Generamos la lista de nombres de canales
        nombres_canales = ["DSI_Canal_" + str(i) for i in  channels]
        df_features = pd.concat([df_features, pd.DataFrame(np.array(DSI), columns = nombres_canales)], axis = 1)
        
        # Generamos la lista de nombres de canales
        nombres_canales = ["TSI_Canal_" + str(i) for i in channels]
        df_features = pd.concat([df_features, pd.DataFrame(np.array(TSI), columns = nombres_canales)], axis = 1)
        
        # Generamos la lista de nombres de canales
        nombres_canales = ["ASI_Canal_" + str(i) for i in channels]
        df_features = pd.concat([df_features,  pd.DataFrame(np.array(ASI), columns = nombres_canales)], axis = 1)
        
        return df_features

    # Hjorth Parameters proporcionan información de la dinámica temporal de las señales del PSG. La Actividad, Movilidad y los parámetros de complejidad se calculan a partir de la varianza X,
    # Activity = (var(X)) y las derivadas primera (Mobility) y segunda (Complexity)
    
    def Hjorth_Parameters (self, psg_signal, channels):
        # Número de ventanas, canales y muestras por ventana
        num_ventanas, num_canales, muestras_por_ventana = psg_signal.shape

        # Aplicamos operaciones matriciales para el cálculo de estadísticos
        psg_signal_diff1 = np.diff (psg_signal, axis = 2, n = 1)
        psg_signal_diff2 = np.diff (psg_signal_diff1, axis = 2, n = 1)

        # Calculamos Activity (varianza de la señal)
        activity = np.var (psg_signal, axis =  2)

        # Calculamos Mobility (raíz cuadrada de la varianza de la primera derivada dividida por la varianza)
        mobility = np.sqrt (np.var (psg_signal_diff1, axis = 2) / activity)

        # Calculamos Complexity (raíz cuadrada de la varianza de la segunda derivada dividida por la varianza de la primera derivada)
        complexity = np.sqrt (np.var (psg_signal_diff2, axis = 2) / np.var (psg_signal_diff1, axis = 2))

        # Calculamos Skewness
        skewness = skew (psg_signal, axis = 2)

        # Calculamos Kurtosis
        kurtosis_v  = kurtosis (psg_signal, axis = 2)
 
        res_HJorth = activity, mobility, complexity, skewness, kurtosis_v 
    
        return self.Hjorth_Parameters_to_df (res_HJorth, channels)
    
    def Hjorth_Parameters_to_df (self, res_HJorth, channels):
        
        activity, mobility, complexity, skewness, kurtosis_v  = res_HJorth
        
        def to_df (data, str_comd, channels):
            nombres_canales = [str_comd + str (i) for i in channels]
            return pd.DataFrame (data, columns = nombres_canales)
            
        df_features = to_df (activity,"Activity_Canal_",channels)
        df_features = pd.concat ([df_features, to_df (mobility,"Mobility_Canal_",channels)], axis = 1)
        df_features = pd.concat ([df_features, to_df (complexity,"Complexity_Canal_",channels)], axis = 1)
        df_features = pd.concat ([df_features, to_df (skewness,"Skewness_Canal_",channels)], axis = 1)
        df_features = pd.concat ([df_features, to_df (kurtosis_v,"Kurtosis_Canal_",channels)], axis = 1)
        
        return df_features

    # Harmonic Parameters: incluyen tres parámetros: la frecuencia central (fc), el ancho de banda (fr), y el valor espectral en la frecuencia central (Sfc)
    def Harmonic (self, psg_signal, channels):
        sub_bands = {
            "Delta 1": (0.5, 2.0),
            "Delta 2": (2.0, 4.0),
            "Theta 1": (4.0, 6.0),
            "Theta 2": (6.0, 8.0),
            "Alpha 1": (8.0, 10.0),
            "Alpha 2": (10.0, 12.0),
            "Sigma 1": (12.0, 14.0),
            "Sigma 2": (14.0, 16.0),
            "Beta 1": (16.0, 25.0),
            "Beta 2": (25.0, 35.0)
        }


        def signal_power_spectrum (window):
            fs = 100  # Frecuencia de muestreo de las señales
            f, Pxx = welch (window, fs = fs, nperseg = fs*2)
            return f, Pxx

        def calculate_harmonic_parameters (window, sub_bands):
            f, Pxx = signal_power_spectrum(window)
            params = np.empty((len(sub_bands), 3))

            for i, (subband_name, (fL, fH)) in enumerate (sub_bands.items()):
                subband_indices = np.where ((f >= fL) & (f <= fH))[0]
                subband_Pxx     = Pxx [subband_indices]
                subband_f       = f [subband_indices]
                fc  = np.sum (subband_f * subband_Pxx) / np.sum(subband_Pxx)
                fr  = np.sqrt (np.sum(((subband_f - fc) ** 2) * subband_Pxx) / np.sum(subband_Pxx))
                Sfc = Pxx[np.argmin (np.abs(subband_f - fc))]

                params[i] = [fc, fr, Sfc]

            return params

        # Inicializamos las matrices para almacenar los parámetros armónicos
        num_ventanas, num_canales, _ = psg_signal.shape
        params_fc  = np.zeros((num_ventanas, num_canales, len(sub_bands)))
        params_fr  = np.zeros((num_ventanas, num_canales, len(sub_bands)))
        params_Sfc = np.zeros((num_ventanas, num_canales, len(sub_bands)))

        # Aplicamos la función a lo largo del eje de las ventanas
        result = np.apply_along_axis (calculate_harmonic_parameters, axis = 2, arr = psg_signal, sub_bands = sub_bands)

        params_fc, params_fr, params_Sfc = result [:, :, :, 0], result [:, :, :, 1], result [:, :, :, 2]
        
        res_harmonic = params_fc, params_fr, params_Sfc
        
        return self.Harmonic_to_df (res_harmonic, channels)

 
    def Harmonic_to_df (self, res_harmonic, channels):
        params_fc, params_fr, params_Sfc = res_harmonic
        
        sub_bands = {
            "Delta 1": (0.5, 2.0),
            "Delta 2": (2.0, 4.0),
            "Theta 1": (4.0, 6.0),
            "Theta 2": (6.0, 8.0),
            "Alpha 1": (8.0, 10.0),
            "Alpha 2": (10.0, 12.0),
            "Sigma 1": (12.0, 14.0),
            "Sigma 2": (14.0, 16.0),
            "Beta 1": (16.0, 25.0),
            "Beta 2": (25.0, 35.0)
        }
        variables = {
            'fc' : {'fc':params_fc},
            'fr' : {'fr':params_fr},
            'sfc': {'sfc':params_Sfc}
        }

        # --123-- falta ver para optimizar el código con apply y compresión
        col_name = []
        data_fe  = []
        for sb,isb in  zip (sub_bands, range(10)):
            for canal, ixcn in zip (channels,range (0, len(channels))):
                for variable, data in variables.items ():
                    col1 = variable + '_'+ sb +  '_Canal-' + canal
                    col_name.append (col1)
                    data_fe.append (variables [variable][variable][:,ixcn,isb])


        featur      = np.array (data_fe).T
        df_features = pd.DataFrame (featur, columns = col_name)
 
        
        return df_features 
    # Autoregressive Coefficients: Un modelo Ar se define tal que la variable de salida depende linealmente de sus propios valores anteriores. 
    def CoefAR (self, psg_signal, channels):
        # Número de ventanas, canales y orden del modelo AR
        n_ventanas, n_canales, n_muestras = psg_signal.shape
        order = 3  # Orden del modelo AR

        # Función para calcular coeficientes AR para una ventana y canal específicos
        def coefs_ar (args):
            ventana_actual, canal_actual = args
            ventana_canal = psg_signal [ventana_actual, canal_actual, :]

            calculated_AR = AutoReg (ventana_canal, lags = 3)
            calculated_ar_params = calculated_AR.fit (cov_type = "nonrobust").params
            
            return calculated_ar_params

        # Utiliza ThreadPoolExecutor para calcular los coeficientes AR en paralelo
        with ThreadPoolExecutor () as executor:
            args_list  = [(ventana, canal) for ventana in range (n_ventanas) for canal in range (n_canales)]
            resultados = np.array (list (executor.map (coefs_ar, args_list)))

        # Reorganiza los resultados en un arreglo tridimensional (ventanas, canales, coeficientes)
        ar_coefs = resultados.reshape (n_ventanas, n_canales, order + 1)

        # Calculamos la media y desviación estándar de los 4 coeficientes para cada ventana de la señal de cada canal

        media_ar_coefs = np.mean (ar_coefs, axis = 2)
        std_ar_coefs   = np.std  (ar_coefs, axis = 2)
        
        res_coefAR = ar_coefs, media_ar_coefs, std_ar_coefs
        
        return self.CoefAR_to_df (res_coefAR, channels)
    
    def CoefAR_to_df (self, res_coefAR, channels):
    
        ar_coefs, media_ar_coefs, std_ar_coefs = res_coefAR
        
        data = {
            f"AR_Coef_{coef}_Canal_{canal}": ar_coefs [:, ixcn, coef]
            for coef in range (4)
            for canal, ixcn in zip (channels, range (0, len(channels)))
        }

        df_features = pd.DataFrame (data)

        for canal, ixcn in zip (channels,range (0, len(channels))):
            col_name               = f"Media_AR_Canal_{canal}"
            df_features [col_name] = media_ar_coefs [:, ixcn]

        for canal, ixcn in zip (channels,range (0, len(channels))):
            col_name               = f"Std_AR_Canal_{canal}"
            df_features [col_name] = std_ar_coefs [:, ixcn]

        return df_features

    
   # Percentil 25, 50, 75: El análisis percentil proporciona algunos información sobre la amplitud de la señal y podría ser útil para discernir ciertas etapas del sueño       
     
    def Percentil (self, psg_signal, channels):
        # Definimos los percentiles que deseas calcular
        percentiles = [25, 50, 75]
 
        # Calculamos los percentiles para cada ventana de cada canal
        window_percentiles = np.percentile (psg_signal, percentiles, axis = 2)
    
        return self.Percentil_to_df (window_percentiles, channels) 
 
    def Percentil_to_df (self, res_percent, channels):
    
        def to_df (data, str_comd, channels):
            nombres_canales = [str_comd + str (i) for i in channels]
            return pd.DataFrame (data, columns = nombres_canales)
            
        df_features = to_df (res_percent [0,:,:], "Percentil25_",channels)
        df_features = pd.concat ([df_features, to_df (res_percent [1,:,:],"Percentil50_",channels)], axis = 1)
        df_features = pd.concat ([df_features, to_df (res_percent [2,:,:],"Percentil75_",channels)], axis = 1)
 
 
        return df_features

    
    def extract_paper_features (self, eeg_resam, eeg_names, eeg_eog_resam, eeg_eog_names, level = 3):
        df_features  = self.wavelet_transform (eeg_resam, pywt.swt, eeg_names, level) 
        df_features = pd.concat ([df_features,  self.PeaktoPeak (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.Entropy (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.RSP (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.Harmonic (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.Hjorth_Parameters (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.CoefAR (eeg_eog_resam,eeg_eog_names)], axis = 1)
        df_features = pd.concat ([df_features,  self.Percentil (eeg_eog_resam,eeg_eog_names)], axis = 1)
        return df_features
  
            # NORMALIZACIÓN
    
   # Las características extraídas se transforman y normalizan para para reducir la influencia de los valores extremos. X = arcsin (sqrt(Y)) __> N(dubjects)x M (features)
   # Para evitar que las características en rangos numéricos mayores dominen a las de rangos numéricos más pequeños, así como dificultades numéricas durante
   # clasificación; cada característica de la matriz transformada X es independientemente normalizado al rango [0,1]  Xij = Xij/(max (Xj)-min (Xj))
    def normalice_matrix_features (self, df_features):
        #  Trasnformamos el dataFrame df_features a una matriz de características Y
        Y = df_features.values

        # Limita los valores de Y al rango [-1, 1]
        Y = np.clip (Y, -1, 1)

        # Aplica un valor umbral mínimo a Y antes de calcular la raíz cuadrada. 
        # Esto se hace para evitar que valores muy muy pequelos provoquen un error 
        #en las operaciones, sobretodo en las trigonométricas y raiz cuadrada
        # Además evitamos que en la normalización par aestos datos den cero
        umbral_minimo = 1e-12   
        Y_con_umbral  = np.maximum (Y, umbral_minimo)

        # Aplica la transformación: X = arcsin(sqrt(Y_con_umbral)). Esta trasnformación de los datos
        # permiten lograr una distribución más cercana a una distribución normal o para estabilizar la varianza. 
        # Además permite el escalamiento a un Rango [-1,1], reducir el impacto de valores atípicos en los datos
        # y mejorar la interpretación de los coeficientes en modelos, lo que facilita la comprensión de las relaciones entre variables.
        X = np.arcsin (np.sqrt (Y_con_umbral))

        # Escala los valores resultantes para limitarlos a un rango más pequeño
        X = X / np.pi  # Escala al intervalo [-1/2, 1/2], puedes ajustar este valor si es necesario

        # Calcula los valores máximos y mínimos por característica en X
        max_values = np.max (X, axis = 0)
        min_values = np.min (X, axis = 0)
 
        # Comprueba si las diferencias entre valores máximos y mínimos son cero
        non_zero_diff = max_values - min_values != 0
 
        # Realiza la normalización solo si las diferencias no son cero
        normalized_X = np.zeros_like (X)
        normalized_X [:, non_zero_diff] = (X[:, non_zero_diff] - min_values[non_zero_diff]) / (max_values[non_zero_diff] - min_values[non_zero_diff])
 
        # El resultado es la matriz normalized_X con las características transformadas y normalizadas
        features_norm = normalized_X
        
        return pd.DataFrame (features_norm, columns = df_features.columns) 
    def remove_constant_features(self, X, threshold = 0.001):
        """
        Elimina características constantes o con muy baja varianza.

        Parámetros:
        X (numpy.ndarray): La matriz de características de forma (n_muestras, n_características).
        threshold (float): Umbral mínimo de varianza. Las características con varianza por debajo de este umbral se eliminarán.

        Retorna:
        X_filtrado (numpy.ndarray): Matriz de características con las características constantes o de baja varianza eliminadas.
        """
        varianza_caracteristicas = np.var(X, axis=0)
        mascara_caracteristicas_constantes = varianza_caracteristicas <= threshold
        indices_eliminados = np.where(mascara_caracteristicas_constantes)[0]
        X_filtrado = X[:, ~mascara_caracteristicas_constantes]
        
        return X_filtrado, indices_eliminados
   
    # Seleccion de características basadas en la prueba F de ANOVA o IM y la validación cruzada iterativa.
    # subconjunto de características óptimo que maximiza el rendimiento del modelo de clasificación en
    # función de la prueba F de ANOVA y la validación cruzada. El algoritmo detiene la eliminación de 
    # características cuando se encuentra una configuración que mejora el rendimiento o cuando ya no 
    # se pueden eliminar más características sin empeorar el rendimiento.
    def select_features (self, features, labels, func_select = mutual_info_classif):
        X = features  # características
        y = labels    # etiquetas

        # Eliminamos las caraterístias constantes o de baja varianza

        X_filtered, idx_elim = self.remove_constant_features (X)
        selected_features    = None
         
        selected_features = np.arange (X.shape [1])  # Todas las características se seleccionan inicialmente
        
        # Crea una máscara booleana que indica qué elementos deben mantenerse
        mascara = ~np.isin(selected_features, idx_elim)
        
        # Aplica la máscara para obtener el nuevo vector sin los elementos a eliminar
        selected_features = selected_features [mascara]
          
        try:
            # Step 1: Selecciona características basadas en la prueba F de ANOVA (f_classif) o Información mutua (mutual_info_classif)
            selector          = SelectKBest (score_func = func_select, k = 'all')  # Selecciona todas las características inicialmente
            X_new             = selector.fit_transform (X_filtered, y)
            
            # Realiza una sola vez la validación cruzada con todas las características para obtener un rendimiento de referencia
            clf    = RandomForestClassifier ()  
            scores = cross_val_score (clf, X_filtered, y, cv = 5, scoring = 'accuracy')
             
            max_performance = np.mean (scores)
        
            # Lista para almacenar índices eliminados en cada iteraciom. No corresponde con los originales
            eliminated_idx = []

            # Step 2: Iterativamente refina la selección de características
            for i in range (X.shape [1]):
                if i in idx_elim: 
                    continue
                # Encuentra la característica que contribuye menos al rendimiento y elimínala
                min_feature_idx = np.argmin (selector.scores_)
                
                eliminated_idx.append (min_feature_idx)
                 
                X_new             = np.delete (X_new, min_feature_idx, axis = 1)
                selected_features = np.delete (selected_features, min_feature_idx)
                
 
                # Actualiza el selector con las características restantes
                selector = SelectKBest(score_func=mutual_info_classif, k='all')
                X_new = selector.fit_transform(X_new, y)    

                # Calcula el rendimiento después de eliminar la característica
                clf             = RandomForestClassifier ()  # O el clasificador de tu elección
                scores          = cross_val_score (clf, X_new, y, cv = 5, scoring = 'accuracy')
                avg_performance = np.mean (scores)

                # Si el rendimiento empeora, detén el proceso
                if avg_performance < max_performance:
                    break
                else:
                    max_performance = avg_performance
  
        except ValueError as e:
            print(e) 
        return selected_features 
       
    def jaccard_similarity (self, set1, set2):
  
        # Convierte las matrices numpy en conjuntos
        set1 = set(set1)
        set2 = set(set2)
     
        # Calcula el coeficiente de Jaccard
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
 
    def dice_similarity (self, set1, set2):
        # Calcula el coeficiente de Dice
        intersection = len(set1.intersection(set2))
        total = len(set1) + len(set2)
        return 2 * intersection / total
    
    def consensus_metric(self, subject_selections):
        # Calcula la métrica de consenso promediando el coeficiente de Dice
        n = len(subject_selections)
        consensus = 0.0
 
        for i in range(n):
            for j in range(i + 1, n):
                consensus += self.dice_similarity(subject_selections[i], subject_selections[j])
                # consensus += self.jaccard_similarity(subject_selections[i], subject_selections[j])
       
        consensus /= n * (n - 1) / 2  # Promedio de todas las combinaciones de pares
        return consensus
 
    def select_features_by_consensus (self, features_list, threshold = 14):
        '''
           Obtiene el indice de las características seleccionadas por consenso. A la función
           se le pasa el número mínimo de sujetos que deben haber seleccionado dicha caractersitica
           para ser consensuada. Entendemos que esta métrica pierde menos información que la frecuencia 
           de aparición. en donde se perderían aquellas características seleccionadas por 5 sujetos, 4, etcc..
           
           Otra opción es utilizar el indice dice o jaccard de conjuntos que hemos implementado. Para ello
           se debe modificar esta función. Se le pasaria la lista selections a la función consensus_metric y
           se buscaria el promedio de todas las combinaciones de pares para indie dice o jaccard...lo hemos 
           probado y es mucho más lento y menos interpretable. Así que hemos optado por esta
           estrategia más interpretable y rápida.
        '''
        # Paso 1: Obtiene las selecciones de características para todos los sujetos
        selections = [self.select_features(features.values, labels) for features, labels in features_list]
        
        # Paso 2: Calcula la métrica del consenso
        # Calcular la frecuencia de aparición de cada característica por sujeto y sumarlas
        consensus_metric = [(feature, freq) for feature, freq in Counter(feature for sublist in selections for feature in sublist).items()]
         
        # Paso 3: Aplica el criterio de consenso (umbral)
        selection_matrix = np.array([1 if freq >= threshold else 0 for feature, freq in consensus_metric], dtype=int)
        
        # Paso 4: Obtiene los índices de las características seleccionadas por consenso
        selected_indices = np.where(selection_matrix == 1)[0].tolist()
 
        return selected_indices
  
    def select_features_with_rfe(self,X, y, num_features=10):
        '''
           Esta opción se ha probado e incluso es muchisimo más lenta que
           las métricas de indice de dice o jaccard. A los 20 minutos se corto
           y se descarto su uso.
        '''
        # Crea un modelo de clasificación (por ejemplo, Random Forest)
        model = RandomForestClassifier()
 
        # Crea un objeto RFE para seleccionar características
        rfe = RFE(model, n_features_to_select=num_features)
 
        # Ajusta el modelo RFE a los datos
        rfe.fit(X, y)
 
        # Obtiene las características seleccionadas
        selected_features = rfe.support_
 
        # Filtra las características en tus datos
        X_selected = X[:, selected_features]
 
        return X_selected
 
