import os
import mne
import pywt
import pickle 
import yasa
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from operator import length_hint

from io import StringIO
from pathlib import Path
from nilearn import plotting
from contextlib import redirect_stdout

from mne import io
from mne.io import read_raw_edf

from mne import events_from_annotations
from mne import Epochs, pick_types, find_events

from mne.viz import plot_alignment
from mne.datasets import refmeg_noise

from mne.time_frequency import psd_array_multitaper

from mne.preprocessing import ICA
from mne.preprocessing import regress_artifact
from mne.preprocessing import create_eog_epochs, annotate_muscle_zscore

from scipy.signal import welch
from scipy.signal import lfilter

from scipy.stats import pearsonr
from scipy.signal import resample
from scipy.signal import coherence
from scipy.stats import norm, kstest,chisquare
from scipy.spatial.distance import pdist, squareform
 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, v_measure_score
    
 
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report)
                             
from base.ExractFeatures import ExractFeatures                     
#########MONTAJE Y ASIGNACION DE CANALES ###########

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


###################
## clase para la lectura de los EEG e cada fichero y su preprocesamiento
###################
class EEG_Data:
    _id_subject        = None   # Identificador del sujeto. Número entero que identifica el orden de carga del sujeto
    _class_subject     = None   # Identidicará si el sujeto es Young/Old. Valido para los procesos de clasificación
    _name_subject      = None   # Nombre de los sujectos leidos, es igual al nombre del fichero que tiene lso datos del sujeto
    _raw               = None   # Almacenamos el conjunto de EEG channels sin procesar
    _fraw              = None   # Almacenamos el conjunto de EEG channels preprocedados (por defecto [0,2-100]KHz y mantenemos frezfreq gamma)
    _info              = None   # Almacenamos el objeto info asociado a los canales del subject. Mantiene los nombres de los canales y picks
    _all_channels      = None   # Lista de todos los canales presentes
    _eeg_channels      = None   # canales estandares de los electrodos eeg
    _eog_channels      = None   # canales estandares de los electrodos eog
    _exg_channels      = None   # canales estandares de los electrodos exg
    _sr                = None   # Frecuencia de muestreo de los datos originales
    _epochs            = None   # Enventanado de las señales de cada canal
    _filepath          = None   # Directorio donde se serializa/deserializa en disco las señales del EEG en crudo
    _DEBUG             = True   # Parametrizamos las salidas de debug
    _DEBUG_GR          = True   # Parametrizamos las trazas de gráficos
    _save_path         = './result'
######
    def __init__ (self, name_subject, id_subject, class_subject, data, 
                     DEBUG = False, DEBUG_GR = False, sr = 512, filepath = './eeg_path'):
                         
        self._id_subject       = id_subject
        self._class_subject    = class_subject
        self._name_subject     = name_subject
        self._all_channels     = data.ch_names
        self.channels_assigned = False
        self._sr               = sr
        self._filepath         = filepath
        self.set_data (data.copy().pick (self._all_channels))
        self._DEBUG            = DEBUG
        self._DEBUG_GR         = DEBUG_GR
        if self._DEBUG:
            mne.set_log_level('DEBUG')
        else:
            mne.set_log_level('WARNING')
######
    def set_DEBUG (self, _DEBUG):
        self._DEBUG  = _DEBUG
    def get_DEBUG (self):
        return self._DEBUG
    def set_DEBUG_GR (self, _DEBUG_GR):
        self._DEBUG_GR = _DEBUG_GR
    def get_DEBUG (self):
        return self._DEBUG_GR
    def set_filepath (self, filepath):
        self._filepath  = filepath
    def get_filepath (self):
        return self._filepath
    def clear_ram (self):
        if os.path.exists(self._filepath):
            os.remove(self._filepath)
        self._raw  = None
        self._fraw = None
        
    def set_data (self, data):
        print(data.info['dig'])
        self.set_raw (mne.io.RawArray(data.get_data (),data.info, verbose = self._DEBUG))

    def get_data (self):
        if self.get_fraw() is not None:
          return self.get_fraw ()
        else:
          return self.get_raw ()
    
    def set_raw(self, raw):
        self._raw = raw
        
        # Construir el nombre del archivo
        file_name = os.path.join(self._filepath, f"{self._name_subject}.pkl")
        
        # Crear el directorio si no existe
        os.makedirs(self._filepath, exist_ok = True)
        
        # Serializar a disco
        with open(file_name, 'wb') as f:
          pickle.dump(self.get_raw (), f)
          
        # Eliminar datos de la RAM
        self._raw = None
    
    def get_raw(self):
        if self._raw is None:
          # Construir el nombre del archivo
          file_name = os.path.join(self._filepath, f"{self._name_subject}.pkl")
          # Deserializar desde disco
          if os.path.exists(file_name):
              with open(file_name, 'rb') as f:
                  self._raw= pickle.load (f)
          else:
              raise FileNotFoundError(f"No se encontró el archivo {file_name}")
              
        return self._raw
        
    def set_fraw (self, fraw):
        self._fraw = fraw
    def get_fraw (self):
        return self._fraw
    
    def set_id_subject (self, id_subject):
        self._id_subject = id_subject
    def get_id_subject (self):
        return self._id_subject
    def set_class_subject (self, class_subject):
        self._class_subject = class_subject
    def get_class_subject (self):
        return self._class_subject
    def set_name_subject (self, name_subject):
        self._name_subject = name_subject
    def get_name_subject (self):
        return self._name_subject
    def set_info (self, info):
        self._info = info
    def get_info (self):
        return self._info
    
    def set_all_channels (self, all_channels):
        self._all_channels = all_channels
    def get_all_channels (self):
        return self._all_channels
    def set_eeg_channels (self, eeg_channels):
        self._eeg_channels = eeg_channels
    def get_eeg_channels (self):
        return self._eeg_channels
    def set_eeg_channels (self, eeg_channels):
        self._eeg_channels = eeg_channels
    def get_eeg_channels (self):
        return self._eeg_channels
    def set_eog_channels (self, eog_channels):
        self._eog_channels = eog_channels
    def get_eog_channels (self):
        return self._eog_channels
    def set_exg_channels (self, exg_channels):
        self._exg_channels = exg_channels
    def get_exg_channels (self):
        return self._exg_channels
    def set_exg_channels (self, exg_channels):
        self._exg_channels = exg_channels
    def get_exg_channels (self):
        return self._exg_channels

    def set_sr(self, sr):
        self._sr = sr
    def get_sr (self):
        return self._sr
    def set_epochs (self, epochs):
        self._epochs = epochs
    def get_epochs (self):
        return self._epochs
    def get_eeg_channel_indices (self):
        return [self.all_channels.index (ch) for ch in self._eeg_channels]
    def get_eog_channel_indices(self):
        return [self.all_channels.index (ch) for ch in self._eog_channels]
    def get_exg_channel_indices(self):
        return [self.all_channels.index (ch) for ch in self._exg_channels]
    
    def _ensure_dir (self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
######
    def _verify_selected_channels(self, available_electrodes = None, requested_electrodes = None):
        av_electrodes  = (set(self._eeg_channels) if available_electrodes is None else set (available_electrodes))
        req_electrodes = (set(self._selected_channels) if requested_electrodes is None else set (requested_electrodes))
        missing_electrodes = req_electrodes - av_electrodes

        if missing_electrodes:
            raise ValueError(f"Los electrodos seleccionados no están disponibles en los canales EEG de los datos: {missing_electrodes}")
    def filter_channels(self, selected_channels):
        """
            Filtra el objeto Raw de MNE para quedarse solo con los canales seleccionados.
            
            Parámetros:
            -----------
            raw : mne.io.Raw
                Objeto Raw de MNE que contiene los datos EEG.
            selected_channels : list of str
                Lista de nombres de los canales seleccionados para el análisis.
                
            Retorna:
            --------
            raw_filtered : mne.io.Raw
                Objeto Raw de MNE con solo los canales seleccionados.
        """
        # Verificar si los canales seleccionados están en los canales disponibles
        if self.get_fraw () is not none:
            raw = self.get_fraw ()
        else:
            raw = self.get_raw ()
            
        available_channels = raw.ch_names
        selected_channels  = [ch for ch in selected_channels if ch in available_channels]
    
        if not selected_channels:
            raise ValueError("Ninguno de los canales seleccionados está disponible en los datos EEG.")
    
        # Filtrar los datos para quedarse solo con los canales seleccionados
        raw_filtered = raw.copy().pick_channels(selected_channels)
    
        return raw_filtered
        
    def generate_dataframe (self, electrode_names):
        """
            Genera un DataFrame con los datos de los electrodos especificados.
        
            Args:
                electrode_names (list): Lista de nombres de los electrodos a incluir en el DataFrame.
        
            Returns:
                pd.DataFrame: DataFrame con los datos de los electrodos especificados.
        """
     
        # Filtrar los datos crudos para incluir solo los electrodos especificados
        raw = self.get_raw ()
        if self.get_fraw () is not None:
           raw = self.get_fraw()
           
        filtered_data = raw.copy().pick_channels (electrode_names)
    
        # Crear el DataFrame con los datos filtrados
        data        = filtered_data.get_data().T            # Transponer para tener canales como columnas
        time_points = np.arange(data.shape [0]) / self._sr  # Crear puntos de tiempo basados en la frecuencia de muestreo
    
        df         = pd.DataFrame (data, columns = electrode_names)
        df['time'] = time_points
    
        return data
 
    def _assign_channel (self, all_channels, eeg_channels, eog_channels, exg_channels, exg_type = "ecg", montage = 'biosemi64'):
        """
            Asigna y configura los canales EEG, EOG y EXG, verifica la selección de canales,
            y crea un objeto Raw de MNE con la información de los canales y el montaje especificado.
        
            Parámetros:
            -----------
                all_channels : list of str
                    Lista de todos los nombres de canales disponibles en el estudio.
                eeg_channels : list of str
                    Lista de nombres de canales EEG.
                eog_channels : list of str
                    Lista de nombres de canales EOG.
                exg_channels : list of str
                    Lista de nombres de canales EXG.
                exg_type : str, opcional
                    Tipo de canal EXG (por defecto "ecg").
                montage : str, opcional
                    Nombre del montaje estándar a utilizar (por defecto "biosemi64").
        
            Funcionalidad:
            --------------
                1. Asigna las listas de canales a los atributos internos de la clase.
                2. Verifica que los canales seleccionados estén disponibles en los canales EEG.
                3. Define los nombres y tipos de los canales.
                4. Crea un montaje estándar utilizando MNE.
                5. Imprime los nombres de los electrodos del montaje si está activado el modo de depuración.
                6. Crea un objeto `Info` de MNE con los nombres y tipos de canales.
                7. Asigna el montaje al objeto `Info`.
                8. Crea un objeto `Raw` de MNE con los datos brutos y la información de los canales.
                9. Filtra los datos para quedarse solo con los canales seleccionados y auxiliares.
                10. Guarda el objeto `Raw` resultante en un atributo interno.
                11. Si existen datos filtrados adicionales, crea y guarda un objeto `Raw` filtrado.
        
            Raises:
            -------
                ValueError
                    Si los canales seleccionados no están disponibles en los canales EEG.
        """                             
        self._all_channels      = all_channels
        self._eeg_channels      = eeg_channels
        self._eog_channels      = eog_channels
        self._exg_channels      = exg_channels

        # Definir los nombres de los canales EEG, EOG y EXG
        ch_names = all_channels
        ch_types = ['eeg'] * len (eeg_channels) + ['eog'] * len (eog_channels) + [exg_type] * len (exg_channels) + ['misc']
        
        # Crear un montaje personalizado
        # añado las posiciones de un montaje estandar
        # Crear el montaje "biosemi64"
        montage = mne.channels.make_standard_montage ('biosemi64')
        
        # Obtener los nombres de los electrodos del montaje
        electrodos_disponibles = montage.ch_names
        
        # Imprimir los nombres de los electrodos
        if self._DEBUG: print(f"Electrodos disponibles en el montaje  {montage}: {electrodos_disponibles}")
        
        raw_m = self.get_raw ()
        
        # Crear un objeto Info
        info = mne.create_info (
            ch_names = ch_names,  # Lista de nombres de canales
            ch_types = ch_types,  # Lista de tipos de canales ('eeg', 'eog', 'ecg', 'emg',)
            sfreq    = 512.0
        )
        # Asignar el montaje al objeto Info
        info.set_montage (montage)
        
        # Obtener los índices de los canales de interés
        # picks = mne.pick_channels (info ['ch_names'], include =  eeg_channels)
        
        # Imprimir los índices de los canales seleccionados
        # if self._DEBUG: print("Índices de los canales seleccionados:", picks)
          
        if self._DEBUG: print (f'raw_m.get_montage:{raw_m.get_montage()}')
        
        info.set_meas_date (raw_m.info ['meas_date']) # Registro el momento de creación de los datos mne
        
        self._info = info
        
        # Crear un objeto Raw con tus datos brutos
        raw = mne.io.RawArray (raw_m.get_data (), info) # Creo el objeto conlso datos EEG en crudo y asocio tipo canales
         
        
        if self._DEBUG: print (raw.info)
        
        self.set_raw (raw) # Me guardo los datos en crudo como objeto raw mne y con sus canales asociados
        if self.get_fraw () is not None:
            raw_f       = mne.io.RawArray (self.get_fraw ().get_data (), info) 
            filter_data = raw_f.get_data (picks = filter_channels)
            raw_f       = mne.io.RawArray (filter_data, info_filter)
            self.set_fraw (raw_f) # Me guardo los datos filtrados como objeto raw mne y con sus canales asociados
        

        return self.get_raw ().info ['ch_names']
      
    def assign_channel_names (self, all_channels, eeg_channels, eog_channels, exg_channels, exg_type = "ecg"):
    
        """
            Asigna nombres descriptivos a los canales EEG, EOG y EXG según su ubicación y tipo,
            y los almacena en un diccionario. Utiliza una clasificación basada en los prefijos de los
            nombres de los canales.
        
            Parámetros:
            -----------
            all_channels : list of str
                Lista de todos los nombres de canales disponibles en el estudio.
            eeg_channels : list of str
                Lista de nombres de canales EEG.
            eog_channels : list of str
                Lista de nombres de canales EOG.
            exg_channels : list of str
                Lista de nombres de canales EXG.
            exg_type : str, opcional
                Tipo de canal EXG (por defecto "ecg").
        
            Retorna:
            --------
            dict
                Un diccionario donde las claves son los nombres de los canales y los valores son los
                nombres descriptivos asignados.
        
            Funcionalidad:
            --------------
            1. Llama a `_assign_channel` para configurar y verificar los canales.
            2. Recorre todos los canales en `all_channels`.
            3. Asigna nombres descriptivos a cada canal basado en su prefijo o nombre específico.
            4. Retorna un diccionario con los nombres de los canales y sus nombres descriptivos.
        """
        

        assig_channels = self._assign_channel (all_channels, eeg_channels, eog_channels, exg_channels, exg_type)
        names = {}
        for channel in assig_channels:
            if channel.startswith(('Fp', 'AF')):
                if (channel == 'FP1'):
                    names[channel] = 'H.I-Frontopolar'
                elif (channel == 'FP2'):
                    names[channel] = 'H.D-Frontopolar'
                else:
                    names[channel] = 'Frontopolar'
            elif channel.startswith(('F', 'FC', 'FT')):
                if (channel == 'F3'):
                    names[channel] = 'H.I-Frontal'
                elif (channel == 'F2'):
                    names[channel] = 'L.M-Frontal'
                elif (channel == 'FZ'):
                    names[channel] = 'H.D-Frontal'
                elif (channel == 'F7'):
                    names[channel] = 'H.I-Fronto-temporal'
                elif (channel == 'F8'):
                    names[channel] = 'H.D-Fronto-temporal'
                else:
                    names[channel] = 'Frontal'
            elif channel.startswith(('C', 'CP', 'TP')):
                if (channel == 'C3'):
                    names[channel] = 'H.I-Fronto-temporal'
                elif (channel == 'C4'):
                    names[channel] = 'H.D-Fronto-temporal'
                elif (channel == 'CZ'):
                    names[channel] = 'L.M-Fronto-temporal'
                else:
                    names[channel] = 'Central'
            elif channel.startswith(('P', 'PO')):
                if (channel == 'P3'):
                    names[channel] = 'H.I-Temporal_medio_parietal'
                elif (channel == 'P4'):
                    names[channel] = 'H.D-Temporal_medio_parietal'
                elif (channel == 'PZ'):
                    names[channel] = 'L.M-Temporal_medio_parietal'
                else:
                    names[channel] = 'Parietal-occipital'
            elif channel.startswith(('T')):
                if (channel == 'T3'):
                    names[channel] = 'H.I-Temporal_medio_parietal'
                elif (channel == 'T4'):
                    names[channel] = 'H.D-Temporal_medio_parietal'
                elif (channel == 'T5'):
                    names[channel] = 'H.I-Temporal_posterior_occipital'
                elif (channel == 'T6'):
                    names[channel] = 'H.D-Temporal_posterior_occipital'
                else:
                    names[channel] = 'Temporal-Parietal-Occipital'
            elif channel.startswith(('O', 'Iz', 'Oz', 'POz')):
                if (channel == 'O1'):
                    names[channel] = '-Temporal_posterior_occipital'
                elif (channel == 'O2'):
                    names[channel] = 'H.D-Temporal_medio_parietal'
                else:
                    names[channel] = 'Occipital'
            elif channel in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                names[channel] = 'Motion control'
            elif channel.startswith('EXG'):
                names[channel] = 'Electrooculogram'
            elif channel == 'Status':
                names[channel] = 'Status'
            else:
                names[channel] = 'Unknown'
        return names

###### Preprocesamiento señales temporales de los canales del EEG
    def channel_filtered(self, cut_low=100, cut_hi=0.2):
        """
            Aplica una serie de filtros a los datos EEG para eliminar el ruido y ajustar las frecuencias de interés.
        
            Parámetros:
            -----------
                cut_low : float, opcional
                    Frecuencia de corte baja para el filtro paso bajo (por defecto 100 Hz).
                    Si la frecuencia de corte de paso bajo es mayor a 51 Hz,
                    se aplica un filtro paso banda 0.2 hasta 49 y un paso banda de 51 a la frecuencia de corte.
                    Evitamos realizar un filtro notch porque nos altera la ganancia de la señal y nos genera problemas en la detección de artefactos.
                    Si la frecuencia de corte del paso bajo es menor a 50:
                    se aplica un paso banda con esa frecuencia y con 0.2 que es por defecto.
                cut_hi : float, opcional
                    Frecuencia de corte alta para el filtro paso alto (por defecto 0.2 Hz).
        
            Retorna:
            --------
                mne.io.Raw
                    Objeto Raw de MNE con los datos filtrados.
        
            Funcionalidad:
            --------------
                1. Configura la frecuencia de muestreo y los datos brutos de EEG.
                2. Opción de visualizar el espectro de potencia antes de filtrar, si está en modo de depuración.
                3. Aplica un filtro paso bajo o una combinación de filtros para evitar un filtro notch y paso banda dependiendo del valor de `cut_low`.
                4. Aplica un filtro paso alto a los datos filtrados.
                5. Opción de visualizar el espectro de potencia después de cada paso de filtrado, si está en modo de depuración.
                6. Guarda la información de los datos filtrados y retorna el objeto Raw filtrado.
        """
        sr = self._sr
        raw = self.get_raw()
        raw.info = self._info
    
        if self._DEBUG_GR:
            psd_fig = raw.compute_psd(fmax=sr/2).plot(average=True, picks="data", exclude="bads", amplitude=False)
            plt.show ()
            plt.close(psd_fig)
    
        raw_filter = None
        if cut_low > 51:
            # Aplicar filtro paso banda 0.2 hasta 49 Hz
            raw_lowpass = raw.copy().filter(l_freq=cut_hi, h_freq=49, fir_design='firwin', method='fir',
                                            l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                            fir_window='hamming', verbose=self._DEBUG)
            
            # Aplicar filtro paso banda de 52 a cut_low Hz
            raw_bandpass = raw.copy().filter(l_freq=52, h_freq=cut_low, fir_design='firwin', method='fir',
                                             l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                             fir_window='hamming', verbose=self._DEBUG)
    
            # Concatenar los resultados de los filtros
            raw_filter = mne.concatenate_raws([raw_lowpass, raw_bandpass])
        else:
            # Aplicar filtro paso bajo a cut_low Hz
            raw_filter = raw.copy().filter(l_freq=cut_hi, h_freq=cut_low, fir_design='firwin', method='fir',
                                           l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                           fir_window='hamming', verbose=self._DEBUG)
    
        if self._DEBUG_GR:
            psd_fig = raw_filter.compute_psd(fmax=sr/2).plot(average=True, picks="data", exclude="bads", amplitude=False)
            plt.show()
            plt.close(psd_fig)
    
        # Actualizar la información de la clase con los datos filtrados
        self.set_fraw(raw_filter.copy())
        self.get_fraw().info = raw_filter.info
    
        return self.get_fraw()

    def channel_filtered_notch (self, cut_low = 100, cut_hi = 0.2, freq_notch = [50]):
        '''
            Filtramos las señales para mantener las ondas de estudio e intentar eliminar
            el ruiodo persistente a ondas altas y bajas. Y con ello eliminar ciertos artefactos.
            
            Ondas Delta (δ): 0.5 - 4 Hz Ondas Theta (θ): 4 - 8 Hz
            Ondas Alfa (α) : 8 - 13 Hz Ondas Beta (β): 13 - 30 Hz
            Ondas Gamma (γ): 30 - 100 Hz
            Se realiza los siguientes filtros:
              Filtro Notch:
                  Se aplica un filtro notch a 50 Hz para eliminar el artefacto de la corriente eléctrica.
                  Esto se hace antes de cualquier filtrado paso bajo o paso alto para asegurar que las frecuencias
                  de interés (especialmente las ondas gamma) no se vean afectadas.
            
              Filtro Paso Bajo (Low-Pass):
                  Se aplica un filtro paso bajo si cut_low es menor que la frecuencia de Nyquist (la mitad de la frecuencia
                  de muestreo). Esto preserva las frecuencias hasta cut_low Hz.
                  Si cut_low es mayor o igual a la frecuencia de Nyquist, no se aplica el filtro paso bajo.
            
              Filtro Paso Alto (High-Pass):
                  Se aplica un filtro paso alto a cut_hi Hz para eliminar las frecuencias por debajo de este umbral.
            
              Visualización:
                  Se visualizan los espectros de potencia (PSD) después de aplicar cada filtro para verificar que los filtros
                  están funcionando como se espera.
        '''
        sr       = self.get_sr ()
        raw      = self.get_raw ()
        raw.info = self.get_info ()
        
        # Visualizar el PSD original
        if self._DEBUG: raw.compute_psd (fmax=sr/2).plot(average = True, picks = "data", exclude = "bads", amplitude = False)
        
        # Aplicar filtro notch a 50 Hz para eliminar el artefacto de la corriente eléctrica (Europa)
        
        notch_widths   = 0.25        # Ancho del notch más estrecho
        freqs_to_notch = freq_notch  # Se puede añadir más frecuencias si es necesario, como [50, 100, 150] para armonicos
        raw_notched    = raw.copy().notch_filter (freqs = freqs_to_notch, fir_design = 'firwin', method = 'fir',
                                                    notch_widths = notch_widths, verbose = self._DEBUG)
        
        # Visualizar el PSD después del filtro notch
        if self._DEBUG_GR: raw_notched.compute_psd (fmax = sr/2).plot (average = True, picks = "data", exclude = "bads", amplitude = False)
        
        raw_filter = None
        # Aplicar filtro paso banda (low-pass) a cut_low Hz si cut_low es menor que la frecuencia de Nyquist
        if cut_low < sr / 2:
          raw_filter = raw_notched.copy ().filter (l_freq = cut_hi, h_freq = cut_low, fir_design = 'firwin', method = 'fir',
                                                         l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto',
                                                          fir_window = 'hamming', verbose = self._DEBUG)
        else:
          raw_filter = raw_notched
        

        
        # Visualizar el PSD después de todos los filtros
        if self._DEBUG_GR: raw_filter.compute_psd (fmax = sr/2).plot (average = True, picks = "data", exclude = "bads", amplitude = False)

        # Actualizar la información de la clase con los datos filtrados
        self.set_fraw (raw_filter.copy ())
        self.get_fraw ().info = raw_filter.info
        
        return self.get_fraw ()

######
    def resample_mne (self, n_decim = 2):
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        # Remuestrear los datos
        resampled_data = mne.io.RawArray (raw_p.get_data (), raw_p.info)
        resampled_data.resample (sfreq = raw_p.info['sfreq'] / n_decim)
        # Actualizar los datos y la frecuencia de muestreo
        self.set_fraw (resampled_data)
        self.set_sr (resampled_data.info ['sfreq'])
    
    def resample_scipy (self, n_decim = 2):
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        data      = raw_p.get_data()
        n_samples = data.shape[1] // n_decim
        # Remuestrear los datos
        resampled_data = resample (data, n_samples, axis = 1)
        # Crear nueva instancia de RawArray con los datos remuestreados
        resampled_raw = mne.io.RawArray (resampled_data, raw_p.info)
        resampled_raw.info ['sfreq'] = raw_p.info ['sfreq'] / n_decim
        # Actualizar los datos y la frecuencia de muestreo
        self.set_fraw (resampled_raw)
        self.set_sr (resampled_raw.info['sfreq'])
    
    def decimate(self, n = 2):
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        data = raw_p.get_data()
        
        # Tomar una muestra por cada `n` muestras
        decimated_data = data [:, ::n]
        # Actualizar la información de la frecuencia de muestreo
        decimated_info = raw_p.info.copy ()
        decimated_info ['sfreq'] /= n
        # Crear nueva instancia de RawArray con los datos decimados
        decimated_raw = mne.io.RawArray (decimated_data, decimated_info)
        # Actualizar los datos y la frecuencia de muestreo
        self.set_fraw (decimated_raw)
        self.get_fraw ().info = decimated_raw.info
        self.set_sr (decimated_info ['sfreq'])
######
    def _plot_ica_components (self, times, sources, idx, max_components_per_fig = 10):
        """
            Grafica los componentes ICA de manera organizada en múltiples figuras según el número máximo
            de componentes por figura especificado.

            Parámetros:
            -----------
              times : array-like
                  Vector de tiempo asociado a los datos de cada componente ICA.
              sources : array-like
                  Matriz de fuentes ICA donde cada fila representa un componente ICA y cada columna representa
                  los datos de amplitud de ese componente.
              idx : int
                  Identificador del sujeto o sesión correspondiente a los datos.
              max_components_per_fig : int, opcional
                  Número máximo de componentes a mostrar por figura. Por defecto es 10.

            Returns:
            --------
              None
                  La función muestra múltiples figuras, cada una con un máximo de `max_components_per_fig` componentes
                  ICA representados en gráficos separados por subplots. Cada gráfico muestra la señal de amplitud
                  del componente ICA en función del tiempo.
        """
        n_comp = sources.shape[0]
        n_figs = (n_comp + max_components_per_fig - 1) // max_components_per_fig

        for fig_idx in range(n_figs):
            start_idx = fig_idx * max_components_per_fig
            end_idx = min(start_idx + max_components_per_fig, n_comp)

            fig, axes = plt.subplots(end_idx - start_idx, figsize=(12, 8), sharex=True)
            fig.suptitle(f'Componentes ICA - Sujeto {idx} (Fig. {fig_idx + 1}/{n_figs})', fontsize=16)

            for i, comp_idx in enumerate(range(start_idx, end_idx)):
                if end_idx - start_idx == 1:
                    ax = axes
                else:
                    ax = axes[i]
                ax.plot(times, sources[comp_idx, :])
                ax.set_ylabel(f'Componente {comp_idx + 1}')
                if comp_idx == end_idx - 1:
                    ax.set_xlabel('Tiempo (s)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def _visualize_signals_and_amplitudes (self, times, sources, mean_abs_amplitude, max_peaks, debug = True):
        """
            Visualiza las señales de amplitud de los componentes ICA junto con las líneas que representan
            la media de la amplitud absoluta y los picos máximos.

            Parámetros:
            -----------
              times : array-like
                  Vector de tiempo asociado a los datos de cada componente ICA.
              sources : array-like
                  Matriz de fuentes ICA donde cada fila representa un componente ICA y cada columna representa
                  los datos de amplitud de ese componente.
              mean_abs_amplitude : array-like
                  Lista de valores que representan la media de la amplitud absoluta para cada componente ICA.
              max_peaks : array-like
                  Lista de valores que representan los picos máximos de amplitud para cada componente ICA.
              debug : bool, opcional
                  Si es True, muestra mensajes de depuración en la consola. Por defecto es True.

            Returns:
            --------
              None
                  La función muestra gráficos individuales para cada componente ICA que visualizan la señal de amplitud,
                  la línea de la media de la amplitud absoluta y la línea de los picos máximos.
        """
        if debug: print("*  visualize_signals_and_amplitudes(times, sources, mean_abs_amplitude, max_peaks):")
        num_components = sources.shape[0] - 1
        for i in range(num_components):
            plt.figure(figsize=(15, 3))
            plt.plot(times, sources[i], label=f'Component {i}')
            plt.axhline(mean_abs_amplitude[i], color='r', linestyle='--', label='Mean Abs Amplitude')
            plt.axhline(max_peaks[i], color='g', linestyle='--', label='Max Peak')
            plt.axhline((-1) * max_peaks[i], color='g', linestyle='--', label='Max Peak')
            plt.title(f'Component {i}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _plot_ica_components_distribution(self, times, sources, title='Distribución de Componentes ICA', bins=20):
        """
          Genera un gráfico que muestra la distribución de los componentes ICA junto con estadísticas de resumen
          como percentiles y la distribución normal ajustada.

          Parámetros:
          -----------
            times : array-like
                Vector de tiempo asociado a los datos de cada componente ICA (no utilizado en la función, pero necesario para la consistencia del API).
            sources : array-like
                Matriz de fuentes ICA donde cada fila representa un componente ICA y cada columna representa
                los datos de amplitud de ese componente.
            title : str, opcional
                Título del gráfico. Por defecto es 'Distribución de Componentes ICA'.
            bins : int, opcional
                Número de divisiones para el histograma de amplitudes. Por defecto es 20.

          Returns:
          --------
            artifact_components : list of int
                Índices de los componentes ICA identificados como artefactuales según la función _identify_artifact_hist_components.
        """

        artifact_components = self._identify_artifact_hist_components(sources)

        num_components = sources.shape[0]
        fig, axs = plt.subplots(num_components, 1, figsize=(12, 2 * num_components), sharex=True)

        if num_components == 1:
            axs = [axs]

        for i in range(num_components):
            data = sources[i, :]

            percentile_25 = np.percentile(data, 25)
            percentile_50 = np.percentile(data, 50)
            percentile_75 = np.percentile(data, 75)
            percentile_90 = np.percentile(data, 90)
            percentile_95 = np.percentile(data, 95)
            percentile_99 = np.percentile(data, 99)
            mu, std = norm.fit(data)
            xmin, xmax = min(data), max(data)

            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            axs[i].hist(data, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')
            axs[i].plot(x, p, 'k', linewidth=2)
            axs[i].axvline(percentile_25, color='b', linestyle='--', label=f'Percentil 25: {percentile_25:.2f}')
            axs[i].axvline(percentile_50, color='r', linestyle='--', label=f'Percentil 50 (Mediana): {percentile_50:.2f}')
            axs[i].axvline(percentile_75, color='y', linestyle='--', label=f'Percentil 75: {percentile_75:.2f}')
            axs[i].axvline(percentile_90, color='m', linestyle='--', label=f'Percentil 90: {percentile_90:.2f}')
            axs[i].axvline(percentile_95, color='m', linestyle='--', label=f'Percentil 95: {percentile_95:.2f}')
            axs[i].axvline(percentile_99, color='m', linestyle='--', label=f'Percentil 99: {percentile_99:.2f}')
            if i in artifact_components:
                axs[i].set_title(f'Componente {i + 1}. Ha eliminar ')
            else:
                axs[i].set_title(f'Componente {i + 1}. Distribución correcta')

            axs[i].legend()
            axs[i].grid(True)

        plt.xlabel('Amplitud')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        return artifact_components

    def _identify_artifact_hist_components (self, sources, similarity_threshold = 0.97,
                                                  alpha = 0.05, bins = 20, debug = True):
    
        """
            Identifica componentes ICA artefactuales basados en la comparación de histogramas de amplitudes
            con una distribución normal ajustada y la prueba de chi-cuadrado.

            Parámetros:
            -----------
            sources : array-like
                Matriz de fuentes ICA donde cada fila representa un componente ICA y cada columna representa
                los datos de amplitud de ese componente.
            similarity_threshold : float, opcional
                Umbral de similitud entre el histograma de amplitudes y la distribución normal ajustada.
                Los componentes con similitud por debajo de este umbral se consideran artefactuales. Por defecto es 0.97.
            alpha : float, opcional
                Nivel de significancia para la prueba de chi-cuadrado. Por defecto es 0.05.
            bins : int, opcional
                Número de divisiones para el histograma de amplitudes. Por defecto es 20.
            debug : bool, opcional
                Si es True, imprime información de depuración. Por defecto es True.

            Returns:
            --------
            artifact_components : list of int
                Índices de los componentes ICA identificados como artefactuales basados en la comparación
                del histograma de amplitudes con la distribución normal ajustada y la prueba de chi-cuadrado.
        """
        artifact_components = []
        histogram_similarities = []
        chi2_p_values = []

        for i in range(sources.shape[0]):
            data = sources[i, :]

            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            mu, std = norm.fit(data)
            p = norm.pdf(bin_centers, mu, std)

            hist /= np.sum(hist)
            p /= np.sum(p)

            similarity = np.corrcoef(hist, p)[0, 1]
            histogram_similarities.append(similarity)

            chi2_stat, chi2_p_value = chisquare(hist, f_exp=p)
            chi2_p_values.append(chi2_p_value)

            if similarity < similarity_threshold or chi2_p_value < alpha:
                artifact_components.append(i)

            if debug:
                print(f'Componente {i}:')
                print(f'  Similitud de la densidad del histograma: {similarity:.4f}')
                print(f'  Mu: {mu:.4f}, Std: {std:.4f}')
                print(f'  P-valor chi-cuadrado: {chi2_p_value:.4f}')
                print()

        return artifact_components

    def _detect_and_exclude_high_variance_ica (self, raw, ica, idx, var_thresh, amp_thresh, peak_thresh, debug = True):
        """
            Detectar y excluir componentes ICA con alta varianza, alta amplitud y picos altos. Además se aplica test de similitud
            a una normal de los histogramas de las componentes ICA.

            Parámetros:
            -----------
            raw : instancia de mne.io.Raw
                Los datos EEG sin procesar.
            ica : instancia de mne.preprocessing.ICA
                El objeto ICA ya ajustado a los datos sin procesar.
            idx : int
                El índice o identificador del sujeto.
            var_thresh : float
                El umbral para detectar componentes con alta varianza basado en z-scores.
            amp_thresh : float
                El umbral para detectar componentes con alta amplitud.
            peak_thresh : float
                El umbral para detectar componentes con picos altos.
            debug : bool, opcional
                Si es True, imprime información de depuración y genera gráficos. El valor predeterminado es True.

            Retorna:
            --------
            high_indices : lista de int
              Índices de los componentes ICA identificados como artefactos. Si no se encuentran artefactos, retorna None.
        """
        # Detección artefactos por z-score
        sources          = ica.get_sources (raw).get_data ()
        component_vars   = np.var (sources, axis = 1)
        z_scores         = (component_vars - np.mean(component_vars)) / np.std(component_vars)
        high_var_indices = np.where (z_scores > var_thresh)[0]

        if debug:
            self._plot_ica_components (raw.times, sources, idx, max_components_per_fig = 10)
        # Detección artefactos por mean_abs_amplitude
        mean_abs_amplitude       = np.mean (np.abs(sources), axis = 1)
        global_mean_amplitude    = np.mean (mean_abs_amplitude)
        global_std_dev_amplitude = np.std (mean_abs_amplitude)

        thresh           = global_mean_amplitude + amp_thresh * global_std_dev_amplitude
        high_amp_indices = np.where (mean_abs_amplitude > thresh)[0]

        # Detección de artefactos por max_peaks
        max_peaks            = np.max (np.abs(sources), axis = 1)
        global_mean_peaks    = np.mean (max_peaks)
        global_std_dev_peaks = np.std (max_peaks)

        peak_thresh_value = global_mean_peaks + amp_thresh * global_std_dev_peaks
        high_peak_indices = np.where (max_peaks > peak_thresh_value)[0]

        # Detección de artefactos por test norm of density hist component
        high_norm_indices = self._identify_artifact_hist_components(sources)

        if debug:
            print (f"High variance indices: {high_var_indices.tolist()}")
            print (f"High amplitude indices: {high_amp_indices.tolist()}")
            print (f"High peak indices: {high_peak_indices.tolist()}")
            print (f"High normal indices: {high_norm_indices}")

        if debug:
            self._visualize_signals_and_amplitudes (raw.times, sources, mean_abs_amplitude, max_peaks)

        high_indices = np.unique (np.concatenate ((high_var_indices, high_amp_indices, high_peak_indices, high_norm_indices)))

        if debug:
            print(f"Combined high indices: {high_indices.tolist()}")
            self._plot_ica_components_distribution (raw.times, sources, title = 'Distribución de Componentes ICA', bins = 20)

        if len(high_indices) > 0:
            return high_indices
        else:
            print("No artifacs found, skipping artifact detection.")
            return None

    def _apply_ica_and_reject_segments (self, raw, ica, debug = True):
        """
        Aplicar ICA (Análisis de Componentes Independientes) a los datos EEG sin procesar para identificar y rechazar
        segmentos con artefactos causados por actividades de EOG (Electrooculografía), ECG (Electrocardiografía) y EMG (Electromiografía).

        Parámetros:
        -----------
        raw : instancia de mne.io.Raw
            Los datos EEG sin procesar a ser procesados.
        ica : instancia de mne.preprocessing.ICA
            El objeto ICA que se aplicará a los datos sin procesar.
        debug : bool, opcional
            Si es True, imprime información de depuración. El valor predeterminado es True.

        Retorna:
        --------
        mov_indices : lista de int
            Índices de los componentes ICA identificados como artefactos (EOG, ECG, EMG).
            Si no se encuentran artefactos, retorna None.
        """
        mov_indices = []

        raw_c = raw.copy ()
        raw_c.filter (l_freq = 1.0, h_freq = None)  # Aplicar filtro pasa-altos con frecuencia de corte baja de 1.0 H

        channel_names = raw_c.ch_names
        channel_types = [raw_c.info['chs'][i]['kind'] for i in range (len (channel_names))]

        if debug:
          print("Channel names after ICA:", channel_names)
          print("Channel types after ICA:", channel_types)

        # Detección de canal EOG (movimientos oculares y parpadeos)
        eog_picks = mne.pick_types (raw_c.info, eog = True)
        if len (eog_picks) > 0:
            try:
                eog_indices, _ = ica.find_bads_eog (raw_c)
                if debug:
                    print(f'eog_indices: {eog_indices}')
                mov_indices.extend (eog_indices)
            except Exception as e:
                print (f"Error detecting EOG artifacts: {e}")
        else:
            print ("No EOG channel found, skipping EOG artifact detection.")

        # Detección de canal ECG
        ecg_picks = mne.pick_types (raw_c.info, ecg = True)
        if len (ecg_picks) > 0:
            if debug:
                print(f'ecg_picks: {ecg_picks}')
            ecg_ch_names = [raw_c.ch_names [idx] for idx in ecg_picks]
            if debug:
                print (f'ecg_ch_names: {ecg_ch_names}')
            ecg_indices = []
            try:
                for ecg_ch_name in ecg_ch_names:
                    ecg_idx, _ = ica.find_bads_ecg (raw_c, ch_name = ecg_ch_name, method = 'correlation')
                    ecg_indices.extend (ecg_idx)
                if debug:
                    print(f'ecg_indices: {ecg_indices}')
                mov_indices.extend (ecg_indices)
            except Exception as e:
                print (f"Error detecting ECG artifacts: {e}")
        else:
            print ("No ECG channel found, skipping ECG artifact detection.")

        # Detección de canal EMG (actividad muscular)
        emg_picks = mne.pick_types (raw_c.info, emg = True)
        if len (emg_picks) > 0:
            if debug:
                print (f'emg_picks: {emg_picks}')
            emg_ch_names = [raw_c.ch_names [idx] for idx in emg_picks]
            emg_indices  = []
            try:
                for emg_ch_name in emg_ch_names:
                    emg_idx, _ = ica.find_bads_ecg (raw_c, ch_name = emg_ch_name, method = 'correlation')
                    emg_indices.extend (emg_idx)
                if debug:
                    print("emg_indices:", emg_indices)
                mov_indices.extend (emg_indices)
            except Exception as e:
                print (f"Error detecting EMG artifacts: {e}")
        else:
            print ("No EMG channel found, skipping EMG artifact detection.")

        # Aplicar la exclusión de los componentes ICA identificados
        if len (mov_indices) > 0:
            return mov_indices
        else:
            print ("No EOG, ECG y EMG artifacts found, skipping artifact detection.")
            return None
    def calculate_similarity_histogram (self, data):
        """
          Calcular la similitud entre el histograma de amplitudes de los datos y una distribución normal ajustada
          usando la divergencia de Jensen-Shannon.

          Parámetros:
          -----------
          data : array-like
              Los datos para los cuales se calculará la similitud.

          Retorna:
          --------
          similarity : float
              La medida de similitud entre el histograma de los datos y la distribución normal ajustada,
              derivada de la divergencia de Jensen-Shannon.
        """
        # Calcular histograma de amplitudes y su densidad
        hist, bin_edges = np.histogram (data, bins='auto', density = True)
        bin_centers     = (bin_edges [:-1] + bin_edges [1:]) / 2

        # Ajustar distribución normal a la densidad del histograma
        mu, std = norm.fit (data)
        p       = norm.pdf (bin_centers, mu, std)

        # Normalizar las frecuencias observadas y esperadas
        hist /= np.sum (hist)
        p    /= np.sum (p)

        # Calcular similitud de Jensen-Shannon (manualmente)
        m = 0.5 * (hist + p)
        js_divergence = 0.5 * (entropy(hist, m) + entropy(p, m))
        similarity    = 1 - js_divergence

        return similarity

    def calculate_kl_divergence (self, data):
        """
          Calculate the Kullback-Leibler (KL) divergence between the amplitude histogram of the data
          and a fitted normal distribution.

          Parameters:
          -----------
          data : array-like
              The data for which to calculate the KL divergence.

          Returns:
          --------
          kl_divergence : float
              The KL divergence between the histogram of the data and the fitted normal distribution.
        """
        # Calcular histograma de amplitudes y su densidad
        hist, bin_edges = np.histogram (data, bins = 'auto', density = True)
        bin_centers     = (bin_edges [:-1] + bin_edges[1:]) / 2

        # Ajustar distribución normal a la densidad del histograma
        mu, std = norm.fit (data)
        p       = norm.pdf (bin_centers, mu, std)

        # Normalizar las frecuencias observadas y esperadas
        hist /= np.sum (hist)
        p    /= np.sum (p)

        # Evitar divisiones por cero y calcular KL divergence
        eps = np.finfo (float).eps  # Pequeño valor epsilon para evitar divisiones por cero
        kl_divergence = np.sum (hist * np.log ((hist + eps) / (p + eps)))

        return kl_divergence
        
    def remove_ica_components_artifact (self, var_thresh = 2.0, amp_thresh = 2.0,
                                         ecg_thresh = 0.25, peak_thresh = 0.85):
        """
            Remove ICA components identified as artifacts from EEG data.
        
            Parameters:
            -----------
                var_thresh : float, optional
                    Threshold for variance used to detect ICA components to exclude (default is 2.0).
                amp_thresh : float, optional
                    Threshold for amplitude used to detect ICA components to exclude (default is 2.0).
                ecg_thresh : float, optional
                    Threshold for correlation with ECG signal used to detect ICA components to exclude (default is 0.25).
                peak_thresh : float, optional
                    Threshold for peak-to-peak amplitude used to detect ICA components to exclude (default is 0.85).
        
            Returns:
            --------
                raw_c : instance of Raw
                    The raw data after removal of ICA components identified as artifacts.
        """
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()

        _info = raw_p.info
        raw_c = raw_p

        if self._DEBUG:
            plt.figure (figsize = (15, 10))
            raw_c.plot (duration = 10, n_channels = 30, remove_dc = True, title = 'EEG Original')
            plt.tight_layout ()
            plt.show ()

        ica = mne.preprocessing.ICA (random_state = 0) #max_iter = "auto",
        ica.fit (raw_c)
        ica.exclude = []
        # Mostrar el número de componentes seleccionados automáticamente
        print(f"Número de componentes seleccionados: {ica.n_components_}")
        idx_mov = self._apply_ica_and_reject_segments (raw_p, ica, self._DEBUG)
        idx_var = self._detect_and_exclude_high_variance_ica (raw_p, ica, self._id_subject,
                                                            var_thresh = var_thresh, amp_thresh = amp_thresh,
                                                            peak_thresh = peak_thresh, debug = self._DEBUG)

        if idx_mov is not None:
            ica.exclude.extend (idx_mov)
        if idx_var is not None:
            ica.exclude.extend (idx_var)

        if len (ica.exclude) > 0:
            ica.exclude = sorted (list (set (ica.exclude)))
            raw_c = ica.apply (raw_c, exclude = ica.exclude)
        else:
            raw_c = raw_p

        self.set_fraw (raw_c)
        self.get_fraw ().info = _info

        return raw_c
######
    def set_epochs (self, duration = 30, overlap = 0):
        """
            En fase de pruebas
        """
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        # Definir la duración de cada epoch (por ejemplo, 1 segundo)
        epoch_duration = duration  # Duración en segundos
        
        # Definir el desplazamiento entre epochs (opcional)
        epoch_overlap = overlap  # Superposición de 0.5 segundos entre epochs
        
        # Generar epochs
        self._epochs = mne.make_fixed_length_epochs (raw, duration = epoch_duration, overlap = epoch_overlap)
        return self._epochs

    def apply_epoch_baseline (self, a = None, b = None):
        """
            En fase de pruebas
        """
        # Aplicar la línea base a los datos
        _epochs      = self._epochs
        self._epochs = _epochs.copy ().apply_baseline (baseline = (a, b))
        return self._epochs

    def _eliminate_baseline (self, eeg_data, baseline_interval = [0, 500]):
        """
            En fase de pruebas
        """
        baseline_mean      = np.mean (eeg_data[:, baseline_interval[0]:baseline_interval[1]], axis = 1, keepdims = True)
        eeg_data_corrected = eeg_data - baseline_mean
        return eeg_data_corrected
        
######
    def plot_signal_seg (self, sr = None, seg = 60):
        '''
          Muestra los seg de las señales presentes en cada uno de los canales del PSG
        '''
        self._ensure_dir (os.path.dirname(self._save_path))
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        df   = raw_p.get_data ()
        sr   = self._sr if sr is None else sr
        time = np.arange (df.shape [1]) / sr
        
        fig, axes = plt.subplots (df.shape [0], 1, sharex = True, sharey = False, figsize = (20, df.shape [0]*2))
        for (i,ax) in enumerate (axes):
            ax.plot (time [0:(seg*sr)], df [i][0:(seg*sr)])
            ax.set_title (self._all_channels [i])
            ax.grid (True)
            
        if self._DEBUG_GR: 
            plt.show ()
        else:
            plt.savefig (self._save_path)
            plt.close ()
            print (f"Gráfico guardado en: {self._save_path}")
        
    def plot_epochs (self):
        if self._DEBUG_GR:
            epochs = self._epochs
            if epochs is not None:
                # we'll try to keep a consistent ylim across figures
                plot_kwargs = dict (picks = "all", ylim = dict(eeg = (-10, 10), eog = (-5, 15)))
                # plot the evoked for the EEG and the EOG sensors
                fig = epochs.average ("all").plot (**plot_kwargs)
                fig.set_size_inches (6, 6)
            else:
                print ('No se ha relizado ningún tipo de eventanado en las señales del EEG.')
        else:
            print ('No se ha activado las trazas del entorno grafico.')
    
    def plot_signals (self, channels = None):
        """
            Plot the signals of the specified channels from the Raw object.
            
            Parameters:
                channels (list): List of channel names.
                None
        """
        self._ensure_dir (os.path.dirname(self._save_path))
        raw_p = None
        if self.get_fraw () is None:
            raw_p = self.get_raw ().copy ()
        else:
            raw_p = self.get_fraw ().copy ()
        
        if channels is None:
            channels = raw.ch_names  # Obtener todos los nombres de los canales disponibles
        
        # Filter data to include only the specified channels
        picks = raw_p.pick_channels (channels)
        
        # Get data for the selected channels
        data, _ = raw_p.get_data (return_times = True)
        
        # Plot the signals of the specified channels
        for i, ch_name in enumerate (channels):
          if ch_name != 'Status':
              ch_data = data [i]  # Get data for the current channel
              # Normalize the data to range [0, 1]
              normalized_data = (ch_data - np.min(ch_data)) / (np.max(ch_data) - np.min(ch_data))
              plt.plot (normalized_data + i, label=ch_name)  # Plot the normalized data
        
        # Add legend to the plot
        num_columns = min (10, len (channels))  # Maximum 10 entries per column
        plt.legend (loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = num_columns)
        plt.grid (True)
        
        # Show the plot
        if self._DEBUG_GR: 
            plt.show ()
        else:
            plt.savefig (self._save_path)
            plt.close ()
            print (f"Gráfico guardado en: {self._save_path}")
    
    def plot_spectrum (self, duration = 5):
        if self._DEBUG_GR:
            raw = self.get_raw ( )
            raw.compute_psd (fmax = self._sr/2).plot (average = True,  picks = "data", exclude = "bads", amplitude = False)
            plt.show ()
        else:
            print ('No se ha activado las trazas del entorno grafico.')

    def plot_artifact_eog_ecg (self):
        if self._DEBUG_GR:
            raw_p = None
            if self.get_fraw () is None:
              raw_p = self.get_raw ().copy ()
            else:
              raw_p = self.get_fraw ().copy ()
            
            eog_epochs_y     = mne.preprocessing.create_eog_epochs (raw_p, baseline = (-0.5, -0.2))
            eog_epochs_y.plot_image (combine = "mean")
            avg_eog_epochs_y = eog_epochs_y.average ().apply_baseline ((-0.5, -0.2))
            
            avg_eog_epochs_y.plot_topomap (times = np.linspace (-0.05, 0.05, 11))
            avg_eog_epochs_y.plot_joint (times = [-0.25, -0.025, 0, 0.025, 0.25])
            
            ecg_epochs_y     = mne.preprocessing.create_ecg_epochs (raw_p)
            ecg_epochs_y.plot_image (combine = "mean")
            avg_ecg_epochs_y = ecg_epochs_y.average ().apply_baseline ((-0.5, -0.2))
            
            avg_ecg_epochs_y.plot_topomap (times = np.linspace (-0.05, 0.05, 11))
            avg_ecg_epochs_y.plot_joint (times = [-0.25, -0.025, 0, 0.025, 0.25])
        else:
            print ('No se ha activado las trazas del entorno grafico.')
    def miscellaneous (self):
        raw  = self.get_raw ( )
        if self.get_fraw () is not None:
            raw = self.get_fraw ()
        
        time_secs    = raw.times
        n_time_samps = len (time_secs)
        ch_names     = raw.ch_names
        n_chan       = len (ch_names)  # note: there is no raw.n_channels attribute
        print (
            "the (cropped) sample data object now has time samples {} and byte-wise information about {} channels."
            "".format (n_time_samps, n_chan)
        )
        #print ("The last time sample is at {} seconds.".format (time_secs[-1]))
        print ("The first few channel names are {}.".format (", ".join(ch_names[:3])))
        print ()  # insert a blank line in the output
        
        # some examples of raw.info:
        print ("bad channels:", raw.info ["bads"])  # chs marked "bad" during acquisition
        print (raw.info ["sfreq"], "Hz")  # sampling frequency
        print (raw.info ["description"], "\n")  # miscellaneous acquisition info
        
        print (raw.info)
        
###### Funciones auxiliares para la lectura y el Preprocesamiento señales temporales de los canales del EEG
def visualize_eeg_data_mne (eeg_instance):
    """
        Visualizes EEG data from a dataset instance in a plot for each channel.
    
        :param eeg_instance: Instance of the dataset containing EEG data and the file name.
    """
    file_name     = eeg_instance ['File']
    eeg_data      = eeg_instance ['EEG']
    channel_names = eeg_data.ch_names
    
    ch_names = channel_names
     
    # Create an MNE Raw object with the EEG data
    info = mne.create_info (ch_names = ch_names, sfreq = 250, ch_types = 'eeg')  # Assuming a sampling frequency of 250 Hz

    eeg_data.plot_signal_seg ()
    eeg_data.plot_signals ()
    # Visualize the EEG channels
    #raw.plot(n_channels=num_channels, scalings='auto', title=f'EEG Data from {file_name}', show=True)

def read_eeg_data (directory, entity_subject, count_sub, verbose = False, verbose_gr = False):
    """
        Lee los datos de EEG de todos los archivos BDF en un directorio y los almacena en un DataFrame.
    
        :param directory: Ruta al directorio que contiene los archivos BDF.
        :return: DataFrame con los datos de EEG de todos los archivos en el directorio.
    """
    dt_eeg = [] # Sólo cargaremos 6 ficheros, no podemso por falta de memoria
    count = 0 #
    id_subject = 0
    for file in os.listdir (directory):
        if file.endswith ('.bdf'):
            if verbose: print (f'* file:{file}')
            r_file = os.path.join (directory, file)
            with redirect_stdout (StringIO ()):  # Esto evita que se muestren mensajes en la pantalla
                raw = mne.io.read_raw_bdf (r_file, preload = True)

            dt_channels_eeg = raw
            #  def __init__(self, name_subject, id_subject, class_subject, data, DEBUG = True, sr = 512, filepath = './eeg_path'):
            dt_file = {'File': file, 'EEG': EEG_Data (name_subject = file, id_subject = id_subject,
                                                      class_subject = entity_subject, 
                                                      data = dt_channels_eeg, DEBUG = verbose, 
                                                      DEBUG_GR = verbose_gr, sr = 512)
                    }
            dt_eeg.append (dt_file)
            count += 1
            id_subject += 1

        if count == count_sub:
          break

    return pd.DataFrame (dt_eeg)
    
def read_random_eeg_files(directory, entity_subject, count_subj, verbose = False, verbose_gr = False):
    """
        Reads EEG data from a random selection of BDF files in a directory and stores them in a DataFrame.
    
        :param directory: Path to the directory containing BDF files.
        :param entity_subject: Identifier for the subject class.
        :param count_subj: Number of random files to read.
        :param verbose: If True, prints the progress.
        :return: DataFrame with EEG data from the randomly selected files in the directory.
    """
    # Get list of all BDF files in the directory
    all_files = [file for file in os.listdir(directory) if file.endswith('.bdf')]
    
    # Select a random subset of files
    selected_files = random.sample(all_files, min(count_subj, len(all_files)))

    eeg_data_list = []
    id_subject = 0

    for file in selected_files:
        if verbose:
            print(f'* file: {file}')
        file_path = os.path.join(directory, file)
        
        with redirect_stdout(StringIO()):  # This prevents messages from being printed to the screen
            raw = mne.io.read_raw_bdf(file_path, preload=True)
        
        eeg_data = {
            'File': file,
            'EEG': EEG_Data (name_subject = file, id_subject = id_subject,
                            class_subject = entity_subject, data = raw, 
                            DEBUG = verbose, DEBUG_GR = verbose_gr, sr = 512)
        }
        eeg_data_list.append(eeg_data)
        id_subject += 1

    return pd.DataFrame(eeg_data_list)
def create_dataset (directory, label, count_subj, rand_load = True, verbose = False, verbose_gr = False):
    """
        Crea un dataset con los datos de EEG de todos los archivos DBF en un directorio y agrega una columna con la etiqueta.
    
        :param directory : Ruta al directorio que contiene los archivos DBF.
        :param label     : Etiqueta que se agregará como columna al dataset.
        :param rand_load : Indica si la lectura de ficheros se ecoge de forma aleaoria o secuencial
        :return: DataFrame con los datos de EEG y la etiqueta.
    """
    eeg = None
    if rand_load:
        eeg = read_random_eeg_files (directory, label, count_subj, verbose = verbose, verbose_gr = verbose_gr) 
    else:
        eeg = read_eeg_data (directory, label, count_subj, verbose = verbose, verbose_gr = verbose_gr)
        
    eeg ['label'] = label
    return eeg
def flatten_representations(internal_representations):
    """
        Aplana las representaciones internas en una sola dimensión por sujeto.
    
        Parámetros:
        internal_representations (ndarray): Tensor de representaciones internas con forma (num_subjects, num_samples, num_neurons).
    
        Retorna:
        flattened_representations (ndarray): Array 2D con las representaciones aplanadas, de forma (num_subjects, num_samples * num_neurons).
    
        Descripción:
        - Convierte las representaciones 3D (sujetos, muestras, neuronas) en una forma 2D más manejable.
        - Cada sujeto tendrá todas sus muestras concatenadas en una única dimensión.
    """
    num_subjects, num_samples, num_neurons = internal_representations.shape
    flattened_representations = internal_representations.reshape (num_subjects, num_samples * num_neurons)
    return flattened_representations

def extract_features(internal_representations):
    """
        Extrae la media y la varianza de las activaciones neuronales para cada sujeto.
    
        Parámetros:
        internal_representations (ndarray): Tensor de representaciones internas con forma (num_subjects, num_samples, num_neurons).
    
        Retorna:
        features (ndarray): Array 2D con las características extraídas (media y varianza) para cada sujeto, de forma (num_subjects, num_neurons * 2).
    
        Descripción:
        - Calcula la media y la varianza de las activaciones de cada neurona a lo largo de las muestras.
        - Almacena las características en un array donde la primera mitad son las medias y la segunda mitad las varianzas.
    """
    num_subjects, num_samples, num_neurons = internal_representations.shape
    features = np.zeros((num_subjects, num_neurons * 2))  # Media y varianza para cada neurona

    for i in range(num_subjects):
        subject_data = internal_representations[i]
        mean_features = subject_data.mean(axis=0)
        var_features = subject_data.var(axis=0)
        features[i, :num_neurons] = mean_features
        features[i, num_neurons:] = var_features

    return features
 
def load_data (direct_Younger = "./Younger",direct_Older = './Older', n_y_subject = 2, n_o_subject = 2, rand_load = True, verbose = False, verbose_gr = False):
    """
        Carga los datos de EEG de sujetos jóvenes y mayores desde los directorios especificados.
    
        Parámetros:
            direct_Younger (str): Directorio que contiene los datos de los sujetos jóvenes.
            direct_Older (str): Directorio que contiene los datos de los sujetos mayores.
            n_y_subject (int): Número de sujetos jóvenes a cargar.
            n_o_subject (int): Número de sujetos mayores a cargar.
            rand_load (bool): Indica si la lectura de ficheros se realiza de forma aleatoria = True o secuencial = False.
            verbose (bool): Si es True, imprime mensajes de progreso.
            verbose_gr (bool): Si es True, se muestran gráficas de progreso
        Retorna:
            dataset_Younger (DataFrame): Conjunto de datos de los sujetos jóvenes.
            dataset_Older (DataFrame): Conjunto de datos de los sujetos mayores.
    
        Descripción:
            - Utiliza la función `create_dataset` para cargar los datos.
            - Imprime mensajes de progreso si `verbose` es True.
    """
    if verbose:
        print ("Cargamos dataset Younger")
    dataset_Younger = create_dataset (direct_Younger, label = 'Younger', count_subj = n_y_subject, rand_load = rand_load, verbose = verbose, verbose_gr = verbose_gr)
    if verbose:
        print ("Cargamos dataset Older")
    dataset_Older   = create_dataset (direct_Older, label = 'Older', count_subj = n_o_subject, rand_load = rand_load,verbose = verbose, verbose_gr = verbose_gr)

    return dataset_Younger, dataset_Older
 
def assing_channels_dt (dt_subject, all_channels, eeg_channels, eog_channels, exg_channels,  exg_type = 'ecg'):
    """
        Asigna los nombres de los canales a los datos EEG de cada sujeto en el DataFrame.
    
        Parámetros:
            dt_subject (DataFrame): Conjunto de datos de los sujetos.
            all_channels (list)   : Lista completa de nombres de canales.
            eeg_channels (list)   : Lista de nombres de canales EEG.
            eog_channels (list)   : Lista de nombres de canales EOG.
            exg_channels (list)   : Lista de nombres de canales EXG.
            exg_type (str)        : Tipo de canal EXG, por defecto 'emg'.
    
        Retorna:
            None
    
        Descripción:
            - Recorre cada sujeto en el DataFrame y asigna los nombres de los canales especificados.
            - Utiliza el método `assign_channel_names` de los datos EEG.
    """
    n_subjt = dt_subject.shape [0]

    for i in range (n_subjt):
        dt_subject.iloc [i] ['EEG'].assign_channel_names (all_channels, eeg_channels, eog_channels, exg_channels, exg_type = exg_type)

def filtering_dt (dt_subject, cut_low = 30, n_decim = 2):
    """
        Filtra y remuestrea los datos EEG de cada sujeto en el DataFrame.
    
        Parámetros:
        dt_subject (DataFrame): Conjunto de datos de los sujetos.
        cut_low (int): Frecuencia de corte baja para el filtro.
        n_decim (int): Factor de decimación para remuestrear los datos.
    
        Retorna:
        None
    
        Descripción:
        - Aplica un filtro de frecuencia a los datos EEG de cada sujeto.
        - Remuestrea los datos utilizando un factor de decimación.
    """
    n_subjt = dt_subject.shape [0]

    for i in range (n_subjt):
        dt_subject.iloc [i] ['EEG'].channel_filtered (cut_low = cut_low)
        dt_subject.iloc [i] ['EEG'].resample_mne (n_decim = n_decim)

def process_eeg_data_with_ica(dataset, electrode_names, var_thresh = 2.0,
                                                amp_thresh = 2.0, ecg_thresh = 0.25,
                                                peak_thresh = 0.85, debug = True):
    """
        Procesa los datos EEG utilizando Análisis de Componentes Independientes (ICA) para eliminar artefactos.
    
        Parámetros:
        -----------
            dataset : dict
                Un diccionario que contiene los datos EEG de los sujetos, incluyendo objetos Raw de MNE.
            electrode_names : list
                Lista de nombres de electrodos a considerar para el análisis.
            var_thresh : float, opcional
                Umbral para la detección de componentes ICA con alta varianza. Por defecto es 2.0.
            amp_thresh : float, opcional
                Umbral para la detección de componentes ICA con alta amplitud. Por defecto es 2.0.
            ecg_thresh : float, opcional
                Umbral específico para la detección de artefactos de ECG. Por defecto es 0.25.
            peak_thresh : float, opcional
                Umbral para la detección de picos máximos en los datos de EEG. Por defecto es 0.85.
            debug : bool, opcional
                Si es True, muestra información de depuración durante el proceso. Por defecto es True.
    
        Retorna:
        --------
            data_3d : array
                Un arreglo tridimensional (número de sujetos x número de canales x número de muestras)
                que contiene los datos EEG limpios después de aplicar ICA para cada sujeto.
    
        Descripción:
        ------------
            - Itera sobre cada sujeto en el dataset y aplica el método `remove_ica_components_artifact`.
            - Grafica los datos EEG antes y después de la limpieza con ICA para cada sujeto.
            - Devuelve un arreglo tridimensional con los datos EEG limpios de todos los sujetos.
    """
    n_subjects = len(dataset['EEG'])
    data_list = []
    min_n_samples = min([len(eeg_data.get_data()) for eeg_data in dataset['EEG']])

    for idx in range (n_subjects):
        try:
            eeg_data = dataset ['EEG'][idx]
            raw      = eeg_data.get_fraw ()

            # Prueba de las nuevas funciones
            raw_c = eeg_data.remove_ica_components_artifact (var_thresh = var_thresh, amp_thresh = amp_thresh, peak_thresh = peak_thresh)

            # Obtener los datos limpios
            filtered_data_old = raw.get_data (picks = electrode_names)
            filtered_data     = raw_c.get_data (picks = electrode_names)
            n_samples         = filtered_data.shape [1]

            if n_samples < min_n_samples:
                min_n_samples = n_samples

            # Recortar los datos al tamaño mínimo de muestras
            cropped_data     = filtered_data [:, :min_n_samples]
            cropped_data_old = filtered_data_old [:, :min_n_samples]

            # Graficar los datos del sujeto después de ICA
            data_list.append (cropped_data.T)
            if debug:
                # Graficar los datos del sujeto después de ICA
                fig, ax = plt.subplots(nrows=len(electrode_names), sharex=True, figsize=(12, 8))
                fig.suptitle(f'Señales del Sujeto {idx} Después de ICA', fontsize=16)
                for i, channel in enumerate(electrode_names):
                    ax[i].plot(raw_c.times, filtered_data[i])
                    ax[i].set_ylabel(f'Amplitud ({channel})')
                    if i == len(electrode_names) - 1:
                        ax[i].set_xlabel('Tiempo (s)')
                plt.show()
    
                # Crear una figura y un solo subplot
                fig, ax = plt.subplots()
    
                # Graficar los datos del sujeto
                ax.plot(cropped_data_old.T)
    
                # Etiquetas y título
                ax.set_xlabel('Tiempo')
                ax.set_ylabel('Amplitud')
                ax.set_title(f'Señales del Sujeto {idx} - ANTES cropped_data_old.T')
    
                # Mostrar la figura
                plt.show()
    
                # Crear una figura y un solo subplot
                fig, ax = plt.subplots()
    
                # Graficar los datos del sujeto
                ax.plot(cropped_data.T)
    
                # Etiquetas y título
                ax.set_xlabel('Tiempo')
                ax.set_ylabel('Amplitud')
                ax.set_title(f'Señales del Sujeto {idx} - DESPUES cropped_data.T')
    
                # Mostrar la figura
                plt.show()
        except ValueError as e:
            print(f"Error para el sujeto {dataset['File'][idx]}: {e}")

    # Convertir la lista de datos en una matriz tridimensional
    data_3d = np.stack(data_list)

    return data_3d
    
def process_eeg_data_without_ica (dataset, electrode_names, debug = True):
    n_subjects = len(dataset['EEG'])
    data_list = []
    min_n_samples = min([len(eeg_data.get_data()) for eeg_data in dataset['EEG']])

    for idx in range (n_subjects):
        try:
            eeg_data = dataset ['EEG'][idx]
            raw      = eeg_data.get_fraw ()

            # Obtener los datos limpios
            filtered_data = raw.get_data (picks = electrode_names)
            n_samples     = filtered_data.shape [1]

            if n_samples < min_n_samples:
                min_n_samples = n_samples

            # Recortar los datos al tamaño mínimo de muestras
            cropped_data     = filtered_data [:, :min_n_samples]

            # Graficar los datos del sujeto después de ICA
            data_list.append (cropped_data.T)
            if debug:
                # Graficar los datos del sujeto después de ICA
                fig, ax = plt.subplots(nrows = len(electrode_names), sharex=True, figsize=(12, 8))
                fig.suptitle(f'Señales del Sujeto {idx}   - Después de ICA', fontsize=16)
                for i, channel in enumerate(electrode_names):
                    ax[i].plot(raw.times, filtered_data[i])
                    ax[i].set_ylabel(f'Amplitud ({channel})')
                    if i == len(electrode_names) - 1:
                        ax[i].set_xlabel('Tiempo (s)')
                plt.show()
        except ValueError as e:
            print(f"Error para el sujeto {dataset['File'][idx]}: {e}")

    # Convertir la lista de datos en una matriz tridimensional
    data_3d = np.stack (data_list)

    return data_3d
    
def ica_artifact (dt_subject, type_subject, var_thresh = 2.0,
                                                amp_thresh = 2.0, ecg_thresh = 0.25, peak_thresh = 0.85, debug = True):
    """
        Aplica la eliminación de artefactos en los datos EEG usando Análisis de Componentes Independientes (ICA).
        
        Parámetros:
        -----------
            dt_subject : DataFrame
                Conjunto de datos de los sujetos que contiene los datos EEG a procesar.
            type_subject : str
                Tipo de sujeto o cualquier otra información relevante para el procesamiento.
            var_thresh : float, opcional
                Umbral para la detección de componentes ICA con alta varianza. Por defecto es 2.0.
            amp_thresh : float, opcional
                Umbral para la detección de componentes ICA con alta amplitud. Por defecto es 2.0.
            ecg_thresh : float, opcional
                Umbral específico para la detección de artefactos de ECG. Por defecto es 0.25.
            peak_thresh : float, opcional
                Umbral para la detección de picos máximos en los datos de EEG. Por defecto es 0.85.
            debug : bool, opcional
                Si es True, muestra información de depuración durante el proceso. Por defecto es True.
        
        Retorna:
        --------
        None
        
        Descripción:
        ------------
            - Utiliza ICA para identificar y eliminar componentes de artefactos en los datos EEG de cada sujeto en el DataFrame.
            - Itera sobre cada fila del DataFrame `dt_subject` y aplica el método `remove_ica_components_artifact` a cada registro
              bajo las configuraciones especificadas.
    """
    n_subjt_y = dt_subject.shape [0] # 4
    for i in range (n_subjt_y):
        dt_subject.iloc [i] ['EEG'].remove_ica_components_artifact (var_thresh = 2.0, amp_thresh = 2.0, 
                                                                          ecg_thresh = 0.25, peak_thresh = 0.85)


def create_3d_matrix (dataset, num_subjects,  
                           exclude_bads = ["EXG5", "EXG6", "EXG7", "EXG8", "UP", "DOWN", "LEFT","RIGHT","Status"]):
    """
      Crea una matriz tridimensional con los datos EEG de los sujetos en el dataset.

      Args:
      - dataset (list): El dataset que contiene los datos EEG de los sujetos.
      - num_subjects (int): El número de sujetos que se incluirán en la matriz.
      - exclude_bads (list): Lista del nombre de los canales que se desean eliminar del estudio

      Returns:
      - data_matrix (numpy.ndarray): La matriz tridimensional con los datos EEG de los sujetos.
    """

    # Obtener el número de sujetos a incluir
    num_subjects = min (len(dataset), num_subjects)

    # Obtener el tamaño mínimo de las muestras de los primeros num_subjects elementos del dataset
    min_sample_size = dataset ['EEG'].iloc[:num_subjects].apply (lambda x: x.get_data ().get_data ().shape[1]).min()

    # Obtener el número de canales y el tamaño de las muestras del primer sujeto
    dt_rc        = (dataset.iloc[0]) ['EEG'].get_data ().get_data ()
    num_channels = dt_rc.shape [0]
    sample_size  = dt_rc.shape [1]
    min_dt       = min (min_sample_size,sample_size)

    eeg_data = []
    # Eliminar canales excluidos
    if exclude_bads is not None:
      num_channels = num_channels - len (exclude_bads)
      n_subjct = dataset.shape [0]
      for i in range (n_subjct):
        raw_data = (dataset.iloc [i])['EEG'].get_data(). copy ().drop_channels(exclude_bads)
        eeg_data.append(raw_data)

    # Crear la matriz tridimensional para almacenar los datos
    data_matrix = np.zeros ((num_subjects, num_channels, min_dt))

    # Llenar la matriz con los datos de EEG de cada sujeto
    for i in range(num_subjects):
      data_matrix [i, :, :] = eeg_data [i].get_data () [:,:min_dt]

    return data_matrix
    
def generate_ar_process(coeffs, n_samples):
    """
    Generate an autoregressive (AR) process.
    :param coeffs: Coefficients of the AR process.
    :param n_samples: Number of samples to generate.
    :return: Generated AR process.
    """
    noise = np.random.normal(size=n_samples)
    return lfilter([1], np.concatenate(([1], [-c for c in coeffs])), noise)

def generate_gaussian_process(mean, cov, n_samples):
    """
    Generate a Gaussian process.
    :param mean: Mean of the process.
    :param cov: Covariance matrix of the process.
    :param n_samples: Number of samples to generate.
    :return: Generated Gaussian process.
    """
    return np.random.multivariate_normal(mean, cov, n_samples).T

def generate_synthetic_eeg_data(n_subjects_per_group, n_samples_per_subject, n_channels, sr):
    """
        Genera datos sintéticos de EEG para dos grupos distintos de sujetos (jóvenes y mayores), incluyendo 
        diversas bandas de frecuencia y características específicas para cada grupo.
    
        Parámetros:
        n_subjects_per_group (int): Número de sujetos en cada grupo (jóvenes y mayores).
        n_samples_per_subject (int): Número de muestras de EEG por sujeto.
        n_channels (int): Número de canales de EEG.
        sr (int): Frecuencia de muestreo (Hz).
    
        Retorna:
        data (ndarray): Matriz con datos sintéticos de EEG de tamaño 
                        (2 * n_subjects_per_group, n_samples_per_subject, n_channels).
        labels (ndarray): Matriz con etiquetas one-hot de tamaño (2 * n_subjects_per_group, 2).
    
        Descripción:
        - Cada grupo tiene un conjunto específico de características de señal EEG.
        - Se simulan cinco bandas de frecuencia: Delta, Theta, Alpha, Beta, y Gamma.
        - Los sujetos jóvenes tienen un pico de amplitud en la banda Beta.
        - Los sujetos mayores tienen señales de menor amplitud y más ruido.
        - Se añade un proceso autoregresivo y un proceso gaussiano para mayor realismo.
        - Se añade ruido gaussiano a las señales generadas.
    """
    bandas_frecuencia = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }

    data = np.zeros((2 * n_subjects_per_group, n_samples_per_subject, n_channels))
    labels = np.zeros((2 * n_subjects_per_group, 2))  # Etiquetas one-hot

    for group in range(2):
        np.random.seed(group)  # Establecer una semilla única para cada grupo

        for subj in range(n_subjects_per_group):
            for channel in range(n_channels):
                signal = np.zeros(n_samples_per_subject)
                for band_name, (low_freq, high_freq) in bandas_frecuencia.items():
                    if group == 0:  # Sujetos jóvenes
                        if band_name == 'Beta':
                            # Agregar pico de amplitud en la banda Beta para el grupo 0
                            amplitude = np.random.uniform (5, 7)
                            frequency = np.random.uniform (20, 25)
                            phase = np.random.uniform (0, 2*np.pi)
                            peak_signal = amplitude * np.sin (2 * np.pi * frequency * np.arange (n_samples_per_subject) / sr + phase)
                            signal += peak_signal
                        else:
                            frequency = np.random.uniform (low_freq, high_freq)
                            amplitude = np.random.uniform (2, 5)
                            phase = np.random.uniform (0, 2*np.pi)
                            band_signal = amplitude * np.sin(2 * np.pi * frequency * np.arange(n_samples_per_subject) + phase)
                            signal += band_signal
                    else:  # Sujetos mayores
                        amplitude = np.random.uniform (0.5, 1.5)
                        frequency = np.random.uniform (low_freq, high_freq)
                        phase = np.random.uniform (0, 2*np.pi)
                        band_signal = amplitude * np.sin (2 * np.pi * frequency * np.arange (n_samples_per_subject) / sr + phase)
                        band_signal += np.random.normal (loc = 0, scale = 0.8, size = n_samples_per_subject)
                        signal += band_signal

                # Adición de un proceso autoregresivo
                ar_coeffs  = [0.75, -0.25]
                ar_process = generate_ar_process (ar_coeffs, n_samples_per_subject)
                signal += ar_process

                # Adición de un proceso de Gauss
                mean = np.zeros (n_samples_per_subject)
                cov  = np.eye(n_samples_per_subject) * 0.5
                gaussian_process = generate_gaussian_process (mean, cov, n_samples_per_subject)
                signal += gaussian_process [channel % n_samples_per_subject]

                # Agregar ruido gaussiano
                noise = np.random.normal (loc = 0, scale = 0.5, size = n_samples_per_subject)
                data [group * n_subjects_per_group + subj, :, channel] = signal + noise
                labels [group * n_subjects_per_group + subj, group] = 1  # Asignar etiqueta one-hot de grupo

    return data, labels

def generate_groups_synthetic_data_with_bands  (n_subjects_per_group, n_samples_per_subject, n_channels, sr):
    """
        Genera datos sintéticos de EEG para dos grupos distintos de sujetos (jóvenes y mayores), incluyendo 
        diversas bandas de frecuencia y características específicas para cada grupo.
    
        Parámetros:
        n_subjects_per_group (int): Número de sujetos en cada grupo (jóvenes y mayores).
        n_samples_per_subject (int): Número de muestras de EEG por sujeto.
        n_channels (int): Número de canales de EEG.
        sr (int): Frecuencia de muestreo (Hz).
    
        Retorna:
        data (ndarray): Matriz con datos sintéticos de EEG de tamaño 
                        (2 * n_subjects_per_group, n_samples_per_subject, n_channels).
        labels (ndarray): Matriz con etiquetas one-hot de tamaño (2 * n_subjects_per_group, 2).
    
        Descripción:
        - Cada grupo tiene un conjunto específico de características de señal EEG.
        - Se simulan cinco bandas de frecuencia: Delta, Theta, Alpha, Beta, y Gamma.
        - Los sujetos jóvenes tienen un pico de amplitud en la banda Beta.
        - Los sujetos mayores tienen señales de menor amplitud y más ruido.
        - Se añade ruido gaussiano a las señales generadas.
    """
    bandas_frecuencia = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }

    # Generar señales sinusoidales con ruido para EEG
    data = np.zeros((2 * n_subjects_per_group, n_samples_per_subject, n_channels))
    labels = np.zeros((2 * n_subjects_per_group, 2))  # Etiquetas one-hot

    for group in range(2):
        for subj in range(n_subjects_per_group):
            for channel in range(n_channels):
                signal = np.zeros(n_samples_per_subject)
                for band_name, (low_freq, high_freq) in bandas_frecuencia.items():
                    # Generar señal sinusoidal para la banda de frecuencia actual
                    if group == 0:  # Sujetos jóvenes
                        if band_name == 'Beta':
                            # Agregar pico de amplitud en la banda Beta para el grupo 0
                            amplitude = np.random.uniform(5, 7)  # Amplitud del pico
                            frequency = np.random.uniform(20, 25)  # Frecuencia del pico en Beta
                            phase = np.random.uniform(0, 2*np.pi)  # Fase aleatoria
                            peak_signal = amplitude * np.sin(2 * np.pi * frequency * np.arange(n_samples_per_subject) / sr + phase)
                            signal += peak_signal
                        else:
                            # Aumentar similitud entre las señales de los sujetos del grupo 0
                            amplitude = np.random.uniform(2, 3)  # Amplitud aleatoria mayor
                            frequency = np.random.uniform(low_freq, high_freq)  # Frecuencia aleatoria dentro de la banda
                            phase = np.random.uniform(0, 2*np.pi)  # Fase aleatoria
                            # Usar la misma semilla para todos los sujetos del grupo 0
                            np.random.seed(0)
                            band_signal = amplitude * np.sin(2 * np.pi * frequency * np.arange(n_samples_per_subject) / sr + phase)
                            signal += band_signal
                    else:  # Sujetos mayores
                        # Introducir patrones de señales distintivos para el grupo 1
                        amplitude = np.random.uniform(0.5, 1.5)  # Amplitud aleatoria menor
                        frequency = np.random.uniform(low_freq, high_freq)  # Frecuencia aleatoria dentro de la banda
                        phase = np.random.uniform(0, 2*np.pi)  # Fase aleatoria
                        band_signal = amplitude * np.sin(2 * np.pi * frequency * np.arange(n_samples_per_subject) / sr + phase)
                        signal += band_signal
                # Agregar ruido gaussiano
                noise = np.random.normal(loc=0, scale=0.4, size=n_samples_per_subject)
                data[group * n_subjects_per_group + subj, :, channel] = signal + noise
                labels[group * n_subjects_per_group + subj, group] = 1  # Asignar etiqueta one-hot de grupo

    return data, labels

def plot_eeg_signals (subject, data_matrix):
    """
    Función para dibujar las señales EEG de un sujeto específico.

    Argumentos:
    subject : int
        Número del sujeto del cual se dibujarán las señales EEG.
    data_matrix : numpy.ndarray
        Matriz tridimensional de datos EEG con dimensiones [N, V, T],
        donde N es el número de sujetos, V es el número de muestras y T es el número de canales.
    """
    # Obtener las señales EEG del sujeto especificado
    eeg_signals = data_matrix[subject]

    # Número de canales (T)
    num_channels = eeg_signals.shape[1]

    # Crear una figura con subgráficos para cada canal
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 5*num_channels), sharex=True)

    # Iterar sobre cada canal y dibujar la señal correspondiente
    for i in range(num_channels):
        axes[i].plot(eeg_signals[:60, i])
        axes[i].set_ylabel(f'Canal {i+1}')

    # Establecer etiqueta en el eje x para el último subgráfico
    axes[num_channels-1].set_xlabel('Muestras')

    # Título de la figura
    fig.suptitle(f'Señales EEG del Sujeto {subject}', fontsize=16)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()

    # Mostrar la figura
    plt.show()


def create_coherence_original_vs_reconstructed(eeg_o, eeg_r, fs=512, nperseg=256, banda=None):
    """
        Calcula y grafica la coherencia entre las señales EEG originales y las reconstruidas canal por canal.
    
        Args:
        - eeg_o (numpy array): EEG original con dimensiones (n_channels, n_samples).
        - eeg_r (numpy array): EEG reconstruido con dimensiones (n_channels, n_samples).
        - fs (int, optional): Frecuencia de muestreo de las señales EEG. Por defecto es 512 Hz.
        - nperseg (int, optional): Longitud de cada segmento de señal para el cálculo de la coherencia. Por defecto es 256.
        - banda (tuple, optional): Banda de frecuencia específica para filtrar la coherencia. Debe ser un tuple de dos valores (frecuencia mínima, frecuencia máxima). Por defecto es None.
    
        Returns:
        - cohe (list): Lista con los valores promedio de coherencia por canal.
        - cohe_band (list): Lista con los valores promedio de coherencia por banda y canal.
    """
    n_channels = eeg_o.shape[0]
    print(f'eeg_o: {eeg_o.shape}')
    print(f'eeg_r: {eeg_r.shape}')
    cohe_band = []
    cohe      = []
    for i in range(n_channels):
        f, Cxy = coherence(eeg_o[i, :], eeg_r[i, :], fs=fs, nperseg=nperseg)
        if banda:
            # Filtrar coherencia en la banda de frecuencia específica
            idx_band = np.where((f >= banda[0]) & (f <= banda[1]))
            Cxy_band = Cxy[idx_band]
            cohe_band.append(np.mean(Cxy_band))
            cohe.append(np.mean(Cxy)) 
        else:
            cohe.append(np.mean(Cxy))  # Promedio de coherencia en todas las frecuencias

        print(f"Cxy average for channel {i}: {np.mean(Cxy)}")
        print(f"Cxy shape: {Cxy.shape}")
        print(f"f shape: {f.shape}")

        # Graficar la coherencia
        plt.figure()
        plt.semilogy(f, Cxy)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Coherence')
        plt.title(f'Coherence Channel {i} - {np.mean(Cxy)}')
        plt.show()

    return cohe, cohe_band  
    
def create_coherence_matrix(eeg_o, eeg_r, fs=512, nperseg=256):
    """
        Calcula la matriz de coherencia entre las señales EEG originales y reconstruidas canal por canal.
    
        Args:
        - eeg_o (numpy array): EEG original con dimensiones (n_channels, n_samples).
        - eeg_r (numpy array): EEG reconstruido con dimensiones (n_channels, n_samples).
        - fs (int, optional): Frecuencia de muestreo de las señales EEG. Por defecto es 512 Hz.
        - nperseg (int, optional): Longitud de cada segmento de señal para el cálculo de la coherencia. Por defecto es 256.
    
        Returns:
        - coherence_matrix (numpy array): Matriz de coherencia con dimensiones (n_channels, n_channels).
          Cada elemento [i, j] representa la coherencia promedio entre el canal i de eeg_o y el canal j de eeg_r.
    """
    n_channels = eeg_o.shape[0]
    coherence_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            f, Cxy = coherence(eeg_o[i, :], eeg_r[j, :], fs=fs, nperseg=nperseg)
            # Aquí se podría promediar en una banda de frecuencias específica
            coherence_matrix[i, j] = np.mean(Cxy)  # Promedio sobre todas las frecuencias
    return coherence_matrix
    


def evaluate_clustering(labels, labels_pred):
    """
    Calcula y muestra varias métricas de evaluación para el clustering, incluyendo la matriz de confusión,
    frecuencia, precisión, recall, F1 score, ROC AUC score y la curva ROC. También muestra el reporte de clasificación.

    Args:
    labels (array-like): Etiquetas reales.
    labels_pred (array-like): Etiquetas predichas por el modelo de clustering.
    """

    # Calcular la matriz de confusión
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

    # Calcular frecuencia y precisión
    fq = np.sum(cm, axis=0) / np.sum(cm)
    accuracy = np.trace(cm) / np.sum(cm)
    print("Frequency:", fq)
    print("Accuracy:", accuracy)

    # Calcular otras métricas
    precision = precision_score(labels, labels_pred, average='weighted')
    recall = recall_score(labels, labels_pred, average='weighted')
    f1 = f1_score(labels, labels_pred, average='weighted')
    roc_auc = roc_auc_score(labels, labels_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    # Calcular la curva ROC
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
    
    ################################################################# 
    ## Extracción de características
    #################################################################
def homogenize_raw_size(dataset):
    """
        Ajusta el tamaño de los objetos Raw de cada sujeto en el dataset al tamaño mínimo encontrado.
    
        Args:
            - dataset (DataFrame): El DataFrame que contiene los datos EEG de los sujetos.
            - num_subjects (int): El número de sujetos que se incluirán en el ajuste.
    
        Returns:
            - dataset (DataFrame): El dataset con los objetos Raw de cada sujeto ajustados al tamaño mínimo.
    """

    # Obtener el número de sujetos a incluir
    num_subjects = dataset.shape [0]

    # Obtener el tamaño mínimo de las muestras de los primeros num_subjects elementos del dataset
    if dataset ['EEG'].iloc[0].get_fraw () is not None:
        min_dt = min ([dataset ['EEG'].iloc[i].get_fraw ().get_data ().shape[1] for i in range (num_subjects)])
    else:
        min_dt = min ([dataset ['EEG'].iloc[i].get_raw ().get_data ().shape[1] for i in range (num_subjects)])
    print (f'min_dt:{min_dt}')
    # Recortar todos los objetos Raw al tamaño mínimo encontrado
    for i in range(num_subjects):
        print (f'i_subject:{i}')
        subj_data = dataset.iloc [i]
        print (f"1,- subj_data[EEG].get_fraw().copy().get_data ()  :{subj_data['EEG'].get_fraw().copy().get_data ().shape }")
        data = subj_data['EEG'].get_fraw().copy().get_data () [:,:min_dt]  # Obtener una copia del objeto Raw del sujeto actual
       
        subj_data['EEG'].set_data (mne.io.RawArray (data,  subj_data['EEG'].get_fraw().info))  # Actualizar el objeto Raw en el dataset
        subj_data['EEG'].set_fraw (mne.io.RawArray (data,  subj_data['EEG'].get_fraw().info))  # Actualizar el objeto fRaw en el dataset

        print (f"2,- subj_data[EEG].get_fraw().copy().get_data ()  :{subj_data['EEG'].get_fraw().copy().get_data ().shape }")
    return dataset


def segment_data (raw_dt,  decim = 1, sw = 10):
    # segmentamos la señal en epochs con tamaño de ventaana sw
    sr = raw_dt.info['sfreq']
    times, data_win = yasa.sliding_window (raw_dt.get_data (), sf = sr, window = sw)
    print (f'data_win:{data_win.shape}')

    return times, data_win

def extract_features (ext_features, eeg_resam, eeg_names, eeg_eog_resam, eeg_eog_names, level = 1):
    print (eeg_resam.shape)
    df_features  = ext_features.wavelet_transform (eeg_resam, pywt.swt, eeg_names, level)
    df_features = pd.concat ([df_features,  ext_features.PeaktoPeak (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.Entropy (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.RSP (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.Harmonic (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.Hjorth_Parameters (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.CoefAR (eeg_eog_resam,eeg_eog_names)], axis = 1)
    df_features = pd.concat ([df_features,  ext_features.Percentil (eeg_eog_resam,eeg_eog_names)], axis = 1)
    return df_features

def extract_features_eeg_dataset ( dataset, selected_electrodes, var_thresh = 1.5, amp_thresh = 1.5, peak_thresh =  0.85,
                                                ecg_thresh = 0.25, exclude_bads=["EXG5", "EXG6", "EXG7", "EXG8", "UP", "DOWN", "LEFT", "RIGHT","Status"], exg_type = 'emg', icaFlag = True, debug = True):
    """
    Crea una matriz tridimensional con las características de los datos EEG de todos los sujetos en el dataset.

    Args:
    - dataset (DataFrame): El DataFrame que contiene los datos EEG de los sujetos.
    - num_subjects (int): El número de sujetos que se incluirán en la matriz.
    - exclude_bads (list): Lista del nombre de los canales que se desean eliminar del estudio.

    Returns:
    - features_matrix (numpy.ndarray): La matriz tridimensional con las características de los datos EEG de todos los sujetos.

    time, eeg_resam =  segment_data (raw_young)
    print (eeg_resam.shape)
    eeg_resam = eeg_resam [:,:,:-30] #eliminamos las 30 últimas muestrs de cada canal porque lo indica el artículo de referencia

    n_eeg = len (eeg_channels)
    n_eog = len (eog_channels)
    ext_features =  ExractFeatures (None)
    mtx_fetures  = extract_paper_features (ext_features, eeg_resam [:,0:n_eeg,:], eeg_channels, eeg_resam [:,n_eeg+1:(n_eeg+1+n_eog),:], eog_channels)
    """
    dataset = homogenize_raw_size (dataset)

    print (f'dataset.shape: {dataset.shape}') 
    # Obtener el número de sujetos a incluir
    num_subjects = dataset.shape [0]
    print (f'num_subjects: {num_subjects}')

    # Crear una lista para almacenar las características de todos los sujetos
    all_features = []

    n_eeg = len (eeg_channels)
    n_eog = len (eog_channels)


    # Iterar sobre cada sujeto en el dataset
    for subj_idx in range(num_subjects):
        print (f'subj_idx:{subj_idx}')

        # Obtener los datos EEG del sujeto actual
        raw_     = dataset.iloc [subj_idx]['EEG'].get_fraw ().copy ()
        eeg_data = dataset.iloc [subj_idx]['EEG']

        # Prueba de las nuevas funciones
        if icaFlag:
            raw_c = eeg_data.remove_ica_components_artifact (var_thresh = var_thresh, amp_thresh = amp_thresh, peak_thresh = peak_thresh)
        else:
            raw_c = raw_

        # Seleccionar sólo los canales que se pasan como parametro y los extra
        electrode_select  = selected_electrodes + exclude_bads
        filtered_data     = raw_c.get_data (picks = electrode_select)
        # Crear un objeto Info
        info_filter = mne.create_info (
            ch_names = electrode_select,  # Lista de nombres de canales
            ch_types =  ['eeg'] * len (selected_electrodes) + ['eog'] * len (eog_channels) + [exg_type] * len (exg_channels) + ['misc'],  # Lista de tipos de canales ('eeg', 'eog', 'ecg', 'emg',)
            sfreq    = raw_c.info ['sfreq']
        )
        print (f'filtered_data:{filtered_data.shape}')
        raw_clean = mne.io.RawArray (filtered_data, info_filter)

        # Extracción características de lso canales sleccionados
        time, eeg_resam =  segment_data (raw_clean)
        print (f'eeg_resam:{eeg_resam.shape}')
        eeg_resam = eeg_resam [:,:,:-30] #eliminamos las 30 últimas muestras de cada canal porque lo indica el artículo de referencia

        n_eeg = len (selected_electrodes)
        n_eog = len (eog_channels)
        ext_features =  ExractFeatures ()
        mtx_fetures  = extract_features (ext_features, eeg_resam [:,0:n_eeg,:], selected_electrodes, eeg_resam [:,n_eeg+1:(n_eeg+1+n_eog),:], eog_channels)
        print (f'mtx_fetures:{mtx_fetures.shape}')
        # Agregar las características del sujeto actual a la lista de todas las características
        all_features.append(mtx_fetures)

    features_matrix = np.array (all_features)


    return features_matrix