import os
import mne
import pywt
import pickle 
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from io import StringIO
from pathlib import Path
from nilearn import plotting
from contextlib import redirect_stdout

from mne import io
from mne.preprocessing import ICA
from scipy.signal import coherence
from mne.viz import plot_alignment
from mne.datasets import refmeg_noise
from mne.preprocessing import regress_artifact

from scipy.signal import welch
from scipy.stats import pearsonr
from scipy.signal import lfilter
from scipy.signal import resample
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
  '''
  '''
  _id_subject    = None   # Identificador del sujeto. Número entero que identifica el orden de carga del sujeto
  _class_subject = None   # Identidicará si el sujeto es Young/Old. Valido para los procesos de clasificación
  _name_subject  = None   # Nombre de los sujectos leidos, es igual al nombre del fichero que tiene lso datos del sujeto
  _raw_data      = None   # Almacenamos el conjunto de EEG channels sin procesar
  _fraw_data     = None   # Almacenamos el conjunto de EEG channels preprocedados (100HZ-mantenemos frezfreq gamma) y un paso alto 0.1
  _info          = None   # Almacenamos el objeto info asociado a los canales del subject. Mantiene los nombres de los canales y picks
  _all_channels  = None   # Lista de todos los canales presentes
  _eeg_channels  = None   # canales estandares de los electrodos eeg
  _eog_channels  = None   # canales estandares de los electrodos eog
  _exg_channels  = None   # canales estandares de los electrodos exg
  _sr            = None   # Frecuencia de muestreo de los datos originales
  _epochs        = None   # Enventanado de las señales de cada canal
  _filepath      = None   # Directorio donde se serializa/deserializa en disco las señales del EEG en crudo
  _DEBUG         = True   # Parametrizamos las salidas de debug
######
  def __init__(self, name_subject, id_subject, class_subject, data, DEBUG = False, sr = 512, filepath = './eeg_path'):
        self._id_subject    = id_subject
        self._class_subject = class_subject
        self._name_subject  = name_subject
        self._all_channels  = data.ch_names
        self._sr            = sr
        self._filepath      = filepath
        self.set_data (data.copy().pick (self._all_channels))
        self._DEBUG         = DEBUG
        if self._DEBUG:
            mne.set_log_level('DEBUG')
        else:
            mne.set_log_level('WARNING')
######
  def set_DEBUG (self, _DEBUG):
    self._DEBUG  = _DEBUG
  def get_DEBUG (self):
    return self._DEBUG
  def set_filepath (self, filepath):
    self._filepath  = filepath
  def get_filepath (self):
    return self._filepath
  def clear_disk(self):
      if os.path.exists(self._filepath):
          os.remove(self._filepath)
      self._raw_data = None

  def set_data (self, data):
    if self._info is not None:
      raw = mne.io.RawArray (data, self._info)
      self.set_raw (raw)
    else:
      self.set_raw (data)

  def get_data (self):
    if self._fraw_data is not None:
      return self.get_fraw ()
    else:
      return self.get_raw ()

  def set_raw(self, raw_data):
      self._raw_data = raw_data
      # Construir el nombre del archivo
      file_name = os.path.join(self._filepath, f"{self._name_subject}.pkl")
      # Crear el directorio si no existe
      os.makedirs(self._filepath, exist_ok=True)
      # Serializar a disco
      with open(file_name, 'wb') as f:
          pickle.dump(self._raw_data, f)
      # Eliminar datos de la RAM
      self._raw_data = None

  def get_raw(self):
      if self._raw_data is None:
          # Construir el nombre del archivo
          file_name = os.path.join(self._filepath, f"{self._name_subject}.pkl")
          # Deserializar desde disco
          if os.path.exists(file_name):
              with open(file_name, 'rb') as f:
                  self._raw_data = pickle.load(f)
          else:
              raise FileNotFoundError(f"No se encontró el archivo {file_name}")
      return self._raw_data
  def set_fraw (self, fraw_data):
    self._fraw_data  = fraw_data
  def get_fraw (self):
    return self._fraw_data

  def set_id_subject (self, id_subject):
    self._id_subject  = id_subject
  def get_id_subject (self):
    return self._id_subject
  def set_class_subject (self, class_subject):
    self._class_subject  = class_subject
  def get_class_subject (self):
    return self._class_subject
  def set_name_subject (self, name_subject):
    self._name_subject  = name_subject
  def get_name_subject (self):
    return self._name_subject

  def set_info (self, info):
    self._info  = info
  def get_info (self):
    return self._info

  def set_all_channels (self, all_channels):
    self._all_channels  = all_channels
  def get_all_channels (self):
    return self._all_channels
  def set_eeg_channels (self, eeg_channels):
    self._eeg_channels  = eeg_channels
  def get_eeg_channels (self):
    return self._eeg_channels
  def set_eeg_channels (self, eeg_channels):
    self._eeg_channels  = eeg_channels
  def get_eeg_channels (self):
    return self._eeg_channels
  def set_eog_channels (self, eog_channels):
    self._eog_channels  = eog_channels
  def get_eog_channels (self):
    return self._eog_channels
  def set_exg_channels (self, exg_channels):
    self._exg_channels  = exg_channels
  def get_exg_channels (self):
    return self._exg_channels

  def set_sr(self, sr):
    self._sr  = sr
  def get_sr (self):
    return self._sr

  def set_epochs (self, epochs):
    self._epochs  = epochs
  def get_epochs (self):
    return self._epochs

  def get_eeg_channel_indices(self):
      return [self.all_channels.index(ch) for ch in self.eeg_channels]

  def get_eog_channel_indices(self):
      return [self.all_channels.index(ch) for ch in self.eog_channels]

  def get_exg_channel_indices(self):
      return [self.all_channels.index(ch) for ch in self.exg_channels]

######
  
  
        
  def generate_dataframe(self, electrode_names):
        """
        Genera un DataFrame con los datos de los electrodos especificados.

        Args:
            electrode_names (list): Lista de nombres de los electrodos a incluir en el DataFrame.

        Returns:
            pd.DataFrame: DataFrame con los datos de los electrodos especificados.
        """
        # Verificar si los electrodos especificados están en los canales disponibles
        available_electrodes = set(self._all_channels)
        requested_electrodes = set(electrode_names)
        missing_electrodes = requested_electrodes - available_electrodes

        if missing_electrodes:
            raise ValueError(f"Los siguientes electrodos no están disponibles en los datos: {missing_electrodes}")

        # Filtrar los datos crudos para incluir solo los electrodos especificados
        raw = self.get_raw ()
        if self.get_fraw () is not None:
           raw = self.get_fraw()
        filtered_data = raw.copy().pick_channels(electrode_names)

        # Crear el DataFrame con los datos filtrados
        data = filtered_data.get_data().T  # Transponer para tener canales como columnas
        time_points = np.arange(data.shape[0]) / self._sr  # Crear puntos de tiempo basados en la frecuencia de muestreo

        df = pd.DataFrame(data, columns=electrode_names)
        df['time'] = time_points

        return data
 
  def _assign_channel (self, all_channels, eeg_channels, eog_channels, exg_channels, exg_type = "ecg", montage = 'biosemi64'):
    self.all_channels = all_channels
    self.eeg_channels = eeg_channels
    self.eog_channels = eog_channels
    self.exg_channels = exg_channels

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
    picks = mne.pick_channels (info ['ch_names'], include =  eeg_channels)

    # Imprimir los índices de los canales seleccionados
    if self._DEBUG: print("Índices de los canales seleccionados:", picks)

    #print (raw_m.info)
    #print (raw_m.info ['highpass'])
    #print (raw_m.info ['lowpass'])
    if self._DEBUG: print (f'raw_m.get_montage:{raw_m.get_montage()}')
    info.set_meas_date (raw_m.info ['meas_date'])

    self._info = info
    # Crear un objeto Raw con tus datos brutos
    raw = mne.io.RawArray (raw_m.get_data (), info)
    if self._DEBUG: print (raw.info)
    self.set_raw (raw)

  def assign_channel_names (self, all_channels, eeg_channels, eog_channels, exg_channels, exg_type = "ecg"):
    """
      Assign descriptive names to EEG channels based on their location and function.

      Parameters:
      channels (list): List of channel names.

      Returns:
      dict: A dictionary mapping channel names to descriptive names.
    """
    #self._assign_channel (all_channels, eeg_channels, eog_channels, exg_channels)
    self._assign_channel_type (all_channels, eeg_channels, eog_channels, exg_channels,exg_type)
    names = {}
    for channel in all_channels:
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

  def plot_electrodes_on_mni_atlas (self, channels):
    """
      Plot electrodes on the Montreal Neurological Institute (MNI) atlas.

      Parameters:
        channels (list): List of channel names.
      Returns:
        None
    """

    # Load the standard 10-20 montage
    easycap_montage = mne.channels.make_standard_montage ("easycap-M1")

    # Create MNE info structure with electrode positions
    info = mne.create_info (ch_names = channels, sfreq = 1, ch_types = 'eeg')

    # Create a new figure
    fig = plt.figure ()

    # Get the subplot axes
    ax = fig.axes [0]

    # Set color for electrodes in the channels list to red
    for ch_name in channels:
        ch_pos = info ['ch_pos'][info ['ch_names'].index (ch_name)]
        ax.scatter3D (ch_pos [0], ch_pos [1], ch_pos [2], color = 'red', s = 50)

    # Show the plot
    plt.show()

  def plot_electrodes_on_brain_atlas (self, channels, coordinates):
    """
      Plot electrodes on a brain atlas.

      Parameters:
        channels (list): List of channel names.
        coordinates (ndarray): Array of shape (n_channels, 3) containing the coordinates of each channel.

      Returns:
        None
    """
    # Create a mapping dictionary for channel names to coordinates
    channel_mapping = dict(zip(channels, coordinates))
     
    # Load a brain atlas (e.g., the MNI atlas)
    atlas = plotting.plot_anat ()

    # Mark electrodes on the atlas
    for channel, coord in channel_mapping.items ():
        plt.text (coord[0], coord[1], channel, color = 'black', fontsize = 8, ha = 'center', va = 'center')

    plt.show()
###### Preprocesamiento señales temporales de los canales del EEG
  def channel_filtered (self, cut_low = 100, cut_hi = 0.2):
    
      sr       = self._sr
      raw      = self.get_raw ()
      raw.info = self._info
    
      if self._DEBUG: raw.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
    
      raw_notched = None
      if cut_low > 51:
            # --123-- tengo que ver si aplicar un filtro notch a 50 ó 60 o hacer esta friqueria
        # Aplicar filtro paso bajo hasta 49 Hz
        raw_lowpass = raw.copy().filter(l_freq=None, h_freq=49, fir_design='firwin', method='fir',
                                        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                        fir_window='hamming', verbose = self._DEBUG)
    
        # Aplicar filtro paso banda de 52 a cutoff Hz
        raw_bandpass = raw.copy().filter(l_freq=52, h_freq=cut_low, fir_design='firwin', method='fir',
                                          l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                          fir_window='hamming', verbose = self._DEBUG)
    
        # Concatenar los resultados de los filtros
        raw_notched = mne.concatenate_raws([raw_lowpass, raw_bandpass])
    
      else:
        # Aplicar filtro paso bajo a 49 Hz
        raw_notched = raw.copy ().filter (l_freq = None, h_freq = cut_low, fir_design = 'firwin', method = 'fir',
                                         l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto',
                                         fir_window = 'hamming', verbose = self._DEBUG)
    
    
      if self._DEBUG: raw_notched.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
      # Aplicar filtro paso alto a 0.2 Hz
      raw_highpass = raw_notched.copy ().filter (l_freq = cut_hi, h_freq = None, fir_design = 'firwin')
      if self._DEBUG: raw_highpass.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
    
    
      # Visualizar los datos filtrados
      if self._DEBUG: raw_highpass.plot (title ="raw_highpass")
      self._info           = raw_highpass.info
      self._fraw_data      = raw_highpass.copy ()
      self._fraw_data.info = raw_highpass.info
    
      return self._fraw_data

  def channel_filtered_notch (self, cut_low = 100, cut_hi = 0.2):
    
    sr       = self._sr
    raw      = self.get_raw ()
    raw.info = self._info
    
    # Visualizar el PSD original
    if self._DEBUG: raw.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
    
    raw_notched = None
    if cut_low > 51:
            # --123-- tengo que ver si aplicar un filtro notch a 50 ó 60 o hacer esta friqueria
        # Aplicar filtro paso bajo hasta 49 Hz
        raw_lowpass = raw.copy().filter(l_freq=None, h_freq=49, fir_design='firwin', method='fir',
                                        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                        fir_window='hamming', verbose=True)
    
        # Aplicar filtro paso banda de 52 a cutoff Hz
        raw_bandpass = raw.copy().filter(l_freq=52, h_freq=cut_low, fir_design='firwin', method='fir',
                                          l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                          fir_window='hamming', verbose=True)
    
        # Concatenar los resultados de los filtros
        raw_notched = mne.concatenate_raws([raw_lowpass, raw_bandpass])
    
    else:
        # Aplicar filtro paso bajo a 49 Hz
        raw_notched = raw.copy ().filter (l_freq = None, h_freq = cut_low, fir_design = 'firwin', method = 'fir',
                                         l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto',
                                         fir_window = 'hamming', verbose = True)
    
    
    # Visualizar el PSD después de todos los filtros
    if self._DEBUG: raw_notched.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
    # Aplicar filtro paso alto a 0.2 Hz
    raw_highpass = raw_notched.copy ().filter (l_freq = cut_hi, h_freq = None, fir_design = 'firwin')
    if self._DEBUG: raw_highpass.compute_psd (fmax = sr/2).plot (average = True,  picks = "data", exclude = "bads" )
    
    
    # Visualizar los datos filtrados
    if self._DEBUG: raw_highpass.plot (title ="raw_highpass")
    
    self._info           = raw_highpass.info
    self._fraw_data      = raw_highpass.copy ()
    self._fraw_data.info = raw_highpass.info
    
    return self._fraw_data
  def channel_filtered_ (self, cut_low = 100, cut_hi = 0.2, freq_notch = [50]):
      '''
        Filtramos las señales para mantener las ondas de estudio e intentar eliminar
        el ruiodo persistente a ondas altas y bajas. Y con ello eliminar ciertos artefactos.

        Ondas Delta (δ): 0.5 - 4 Hz Ondas Theta (θ): 4 - 8 Hz
        Ondas Alfa (α): 8 - 13 Hz Ondas Beta (β): 13 - 30 Hz
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
      sr       = self._sr
      raw      = self.get_raw ()
      raw.info = self._info

      # Visualizar el PSD original
      if self._DEBUG: raw.compute_psd (fmax=sr/2).plot(average = True, picks = "data", exclude = "bads")

      # Aplicar filtro notch a 50 Hz para eliminar el artefacto de la corriente eléctrica

      notch_widths   = 0.25  # Ancho del notch más estrecho
      freqs_to_notch = freq_notch  # Se puede añadir más frecuencias si es necesario, como [50, 100, 150] para armonicos
      raw_notched    = raw.copy().notch_filter (freqs = freqs_to_notch, fir_design = 'firwin', method = 'fir',
                                                    notch_widths = notch_widths, verbose = self._DEBUG)

      # Visualizar el PSD después del filtro notch
      if self._DEBUG: raw_notched.compute_psd (fmax = sr/2).plot (average = True, picks = "data", exclude = "bads")

      # Aplicar filtro paso bajo (low-pass) a cut_low Hz si cut_low es menor que la frecuencia de Nyquist
      if cut_low < sr / 2:
          raw_lowpass = raw_notched.copy ().filter (l_freq = None, h_freq = cut_low, fir_design = 'firwin', method = 'fir',
                                                         l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto',
                                                          fir_window = 'hamming', verbose = self._DEBUG)
      else:
          raw_lowpass = raw_notched

      # Aplicar filtro paso alto (high-pass) a cut_hi Hz
      raw_highpass = raw_lowpass.copy().filter (l_freq = cut_hi, h_freq = None, fir_design = 'firwin', method = 'fir',
                                                             l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto',
                                                             fir_window = 'hamming', verbose = self._DEBUG)

      # Visualizar el PSD después de todos los filtros
      if self._DEBUG: raw_highpass.compute_psd (fmax = sr/2).plot (average = True, picks = "data", exclude = "bads")

      # Visualizar los datos filtrados
      if self._DEBUG: raw_highpass.plot (title = "raw_highpass")

      # Actualizar la información de la clase con los datos filtrados
      self._info           = raw_highpass.info
      self._fraw_data      = raw_highpass.copy ()
      self._fraw_data.info = raw_highpass.info

      return self._fraw_data

######
  def resample_mne (self, n_decim = 2):
      raw_p = None
      if self._fraw_data is None:
        raw_p = self.get_raw ().copy ()
      else:
        raw_p = self.get_fraw ().copy ()

      # Remuestrear los datos
      resampled_data = mne.io.RawArray (raw_p.get_data (), raw_p.info)
      resampled_data.resample (sfreq = raw_p.info['sfreq'] / n_decim)
      # Actualizar los datos y la frecuencia de muestreo
      self.set_fraw (resampled_data)
      self._sr = resampled_data.info ['sfreq']

  def resample_scipy (self, n_decim = 2):
      raw_p = None
      if self._fraw_data is None:
        raw_p = self.get_raw ().copy ()
      else:
        raw_p = self.get_fraw ().copy ()

      data = raw_p.get_data()
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
      if self._fraw_data is None:
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
      self._sr = decimated_info ['sfreq']
######
  def set_epochs (self, duration = 30, overlap = 0):
    raw_p = None
    if self._fraw_data is None:
      raw_p = self.get_raw ().copy ()
    else:
      raw_p = self.get_fraw ().copy ()

    # Definir la duración de cada epoch (por ejemplo, 1 segundo)
    epoch_duration = duration  # Duración en segundos

    # Definir el desplazamiento entre epochs (opcional)
    epoch_overlap = overlap  # Superposición de 0.5 segundos entre epochs

    # Generar epochs
    self._epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=epoch_overlap)
    return self._epochs

  def apply_epoch_baseline (self, a = None, b = None):
    # Aplicar la línea base a los datos
    _epochs = self._epochs
    self._epochs = _epochs.copy ().apply_baseline (baseline = (a, b))
    return self._epochs

  def _eliminate_baseline (self, eeg_data, baseline_interval = [0, 500]):
      baseline_mean = np.mean (eeg_data[:, baseline_interval[0]:baseline_interval[1]], axis = 1, keepdims = True)
      eeg_data_corrected = eeg_data - baseline_mean
      return eeg_data_corrected
######
  def plot_signal_seg (self, sr = None, seg = 60):
      '''
          Muestra los seg de las señales presentes en cada uno de los canales del PSG
      '''
      raw_p = None
      if self._fraw_data is None:
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
      plt.show ()

  def plot_electrodes_on_mni_atlas (self, channels, standard_montage = "easycap-M1"):
    """
      Plot electrodes on the Montreal Neurological Institute (MNI) atlas.

      Parameters:
        channels (list): List of channel names.
      Returns:
        None
    """

    # Load the standard 10-20 montage
    easycap_montage = mne.channels.make_standard_montage ("easycap-M1")

    # Create MNE info structure with electrode positions
    info = mne.create_info (ch_names = channels, sfreq = 1, ch_types = 'eeg')

    # Create a new figure
    fig = plt.figure ()

    # Get the subplot axes
    ax = fig.axes [0]

    # Set color for electrodes in the channels list to red
    for ch_name in channels:
        ch_pos = info ['ch_pos'][info ['ch_names'].index (ch_name)]
        ax.scatter3D (ch_pos [0], ch_pos [1], ch_pos [2], color = 'red', s = 50)

    # Show the plot
    plt.show ()

  def plot_epochs (self):
    epochs = self._epochs
    if epochs is not None:
      # we'll try to keep a consistent ylim across figures
      plot_kwargs = dict (picks = "all", ylim = dict(eeg = (-10, 10), eog = (-5, 15)))

      # plot the evoked for the EEG and the EOG sensors
      fig = epochs.average ("all").plot (**plot_kwargs)
      fig.set_size_inches (6, 6)
    else:
      print ('No se ha relizado ningún tipo de eventanado en las señales del EEG.')

  def plot_signals (self, channels = None):
      """
      Plot the signals of the specified channels from the Raw object.

      Parameters:
      channels (list): List of channel names.
      None
      """
      raw_p = None
      if self._fraw_data is None:
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
      plt.show ()

  def plot_spectrum (self, duration = 5):
      raw = self.get_raw ( )
      raw.compute_psd (fmax = self._sr/2).plot (average = True,  picks = "data", exclude = "bads")
      plt.show ()

#######
  def plot_artifact_eog_ecg (self):
    raw_p = None
    if self._fraw_data is None:
      raw_p = self.get_raw ().copy ()
    else:
      raw_p = self.get_fraw ().copy ()

    eog_epochs_y = mne.preprocessing.create_eog_epochs(raw_p, baseline = (-0.5, -0.2))
    eog_epochs_y.plot_image (combine = "mean")
    avg_eog_epochs_y = eog_epochs_y.average ().apply_baseline ((-0.5, -0.2))

    avg_eog_epochs_y.plot_topomap (times = np.linspace (-0.05, 0.05, 11))
    avg_eog_epochs_y.plot_joint (times = [-0.25, -0.025, 0, 0.025, 0.25])

    ecg_epochs_y = mne.preprocessing.create_ecg_epochs (raw_p)
    ecg_epochs_y.plot_image (combine = "mean")
    avg_ecg_epochs_y = ecg_epochs_y.average ().apply_baseline ((-0.5, -0.2))

    avg_ecg_epochs_y.plot_topomap (times = np.linspace (-0.05, 0.05, 11))
    avg_ecg_epochs_y.plot_joint (times = [-0.25, -0.025, 0, 0.025, 0.25])

  def remove_ica_components_artifact(self):
      """
        Esta función utiliza ICA para identificar y eliminar automáticamente
        varios tipos de artefactos de los datos EEG, y muestra el EEG
        antes y después de la limpieza.
      """
      # 1. Copiar los datos raw para preservar los originales
      raw_p = None
      if self._fraw_data is None:
          raw_p = self.get_raw().copy()
      else:
          raw_p = self.get_fraw().copy()

      _info = raw_p.info
      raw_c = raw_p.copy().filter(l_freq=1, h_freq=None)

      # 2. Mostrar el EEG original antes de eliminar artefactos
      if self._DEBUG:
        plt.figure(figsize=(15, 10))
        raw_c.plot(duration=10, n_channels=30, remove_dc=True, title='EEG Original')
        plt.tight_layout()
        plt.show()

      # 3. Inicializar y ajustar el modelo de ICA a los datos
      n_components = min (len (self.eeg_channels), 0.999)  # 99.9% de varianza explicada
      ica = mne.preprocessing.ICA(n_components=n_components, method="picard",
                              max_iter="auto", random_state=97)
      ica.fit(raw_c)
      ica.exclude = []

      # 4. Identificar automáticamente varios tipos de artefactos
      # EOG (movimientos oculares y parpadeos)
      eog_indices, _ = ica.find_bads_eog (raw_c)
      if self._DEBUG: print (f'eog_indices:{eog_indices}')
      ica.exclude.extend(eog_indices)

      # Detección de canal ECG
      ecg_indices = []
      ecg_picks = mne.pick_types(raw_c.info, ecg=True)
      if len(ecg_picks) > 0:
          if self._DEBUG: print(f'ecg_picks:{ecg_picks}')
          ecg_ch_names = [raw_c.ch_names[idx] for idx in ecg_picks]
          if self._DEBUG: print(f'ecg_ch_names: {ecg_ch_names}')
          
          for ecg_ch_name in ecg_ch_names:
              if self._DEBUG: print(f'ecg_ch_name: {ecg_ch_name}')
              ecg_idx, _ = ica.find_bads_ecg(raw_c, ch_name=ecg_ch_name, method='correlation')
              ecg_indices.extend(ecg_idx)
          if self._DEBUG: print(f'**** ecg_indices: {ecg_indices}')
          ica.exclude.extend(ecg_indices)
      else:
          print("No ECG channel found, skipping ECG artifact detection.")

      # EMG (actividad muscular) - verificar si hay canal EMG
      # Detección de canal EMG
      emg_indices = []
      emg_picks = mne.pick_types(raw_c.info, emg=True)
      if len(emg_picks) > 0:
          if self._DEBUG: print(f'emg_picks:{emg_picks}')
          raw_c_emg = raw_c.copy().filter(l_freq=30, h_freq=None)
          ica_emg = ica.copy()
          ica_emg.fit(raw_c_emg)
          for idx, ic in enumerate(ica_emg.get_sources(raw_c_emg).get_data()):
              if np.median(np.abs(ic)) > 2.5 * np.mean(np.abs(ic)):
                  emg_indices.append(idx)
          if self._DEBUG: print("emg_indices:", emg_indices)
          ica.exclude.extend(emg_indices)
      else:
          print("No EMG channel found, skipping EMG artifact detection.")

      # Identificar componentes con alta varianza
      var_thresh     = 4.0 # Umbral de Z-score para considerar alta varianza
      sources        = ica.get_sources(raw_c).get_data()  # Obtener los datos como array NumPy
      component_vars = np.var(sources, axis=1)  # Calcular la varianza a lo largo del tiempo
      z_scores       = (component_vars - np.mean(component_vars)) / np.std(component_vars)
      high_var_indices = np.where(z_scores > var_thresh)[0]
      print (high_var_indices)
      if self._DEBUG: print("high_var_indices:", high_var_indices.tolist())
      ica.exclude.extend(high_var_indices.tolist())
      
      # Identificar componentes con amplitudes anormalmente altas
      sources = ica.get_sources(raw_c).get_data()  # Obtener los datos como array NumPy
      std_dev = np.std(sources, axis=1)  # Desviación estándar de cada componente
      mean_abs_amplitude = np.mean(np.abs(sources), axis=1)  # Promedio de la amplitud absoluta de cada componente
        
      threshold = 2.5  # Ajustar según la distribución de los datos
      high_amplitude_indices = np.where(mean_abs_amplitude > threshold * std_dev)[0]  # Identificar componentes con amplitudes anormales
      if self._DEBUG:
        print("high_amplitude_indices:", high_amplitude_indices.tolist())
        
      # Excluir los componentes con amplitudes anormalmente altas
      ica.exclude.extend(high_amplitude_indices.tolist())

       
      # Eliminar duplicados y ordenar
      ica.exclude = sorted(list(set(ica.exclude)))

      # 5. Aplicar la limpieza de artefactos a los datos raw
      if ica.exclude:
          raw_y = ica.apply(raw_c)
          if self._DEBUG: print("Aplicando ICA para eliminar artefactos...")
      else:
          raw_y = raw_c.copy()
          if self._DEBUG: print("No se identificaron componentes para excluir.")

      # 6. Mostrar el EEG después de eliminar artefactos
      if self._DEBUG:
        plt.figure(figsize=(15, 10))
        raw_y.plot(duration=10, n_channels=30, remove_dc=True, title='EEG Sin Artefactos')
        plt.tight_layout()
        plt.show()

      # 7. Mostrar los componentes ICA excluidos (si hay alguno)
      if self._DEBUG:
          if ica.exclude:
              plt.figure(figsize=(10, 10))
              ica.plot_components(ica.exclude, title='Componentes ICA Excluidos')
              plt.tight_layout()
              plt.show()
          else:
              print("No hay componentes ICA para mostrar.")

      # 8. Informe sobre la varianza explicada
      if self._DEBUG: 
        print("Componentes ICA excluidos:", ica.exclude)
        explained_var_ratio = ica.get_explained_variance_ratio(raw_c)
        for channel_type, ratio in explained_var_ratio.items():
            print(f"Fracción de varianza {channel_type} explicada por todos los componentes: {ratio}")

      # 9. Comparar la densidad espectral de potencia antes y después
      if self._DEBUG:
          fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
          raw_c.plot_psd(ax=ax1, fmin=1, fmax=50, average=True, xscale='log',
                        spatial_colors=False, show=False)
          ax1.set_title('PSD del EEG Original')
    
          raw_y.plot_psd(ax=ax2, fmin=1, fmax=50, average=True, xscale='log',
                        spatial_colors=False, show=False)
          ax2.set_title('PSD del EEG Sin Artefactos')
          plt.tight_layout()
          plt.show()

      # 10. Almacenamos los datos raw limpios
      self.set_fraw(raw_y)
      self.get_fraw().info = _info

      return raw_y

  def remove_ica_components (self, threshold=3):
    raw_p = None
    if self._fraw_data is None:
      raw_p = self.get_raw ().copy ()
    else:
      raw_p = self.get_fraw ().copy ()

    raw_data = raw_p.copy ()
    _info    = raw_p.info

    # Initialize ICA with desired number of components
    ica = mne.preprocessing.ICA (n_components = len(self.eeg_channels), random_state = 97)

    # Fit ICA to the raw data
    ica.fit (raw_data)

    # Compute z-scores for each component
    ica_scores      = np.abs (ica.get_sources (raw_data).get_data())
    ica_scores_mean = np.mean (ica_scores, axis=0)
    ica_scores_std  = np.std (ica_scores, axis=0)
    z_scores        = (ica_scores_mean - np.mean (ica_scores_mean)) / ica_scores_std

    # Identify components above threshold
    components_to_remove = np.where (z_scores > threshold)[0]

    # Exclude identified components
    ica.exclude = components_to_remove

    # Apply ICA to remove identified components
    cleaned_data = ica.apply  (raw_data)

    # Save cleaned data raw
    self.set_fraw (cleaned_data)
    self.get_fraw ().info = _info
    return cleaned_data, components_to_remove
    
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

def read_eeg_data (directory, entity_subject, count_sub, verbose = True):
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
                                                      data = dt_channels_eeg, DEBUG = verbose, sr = 512)}
            dt_eeg.append (dt_file)
            count += 1
            id_subject += 1

        if count == count_sub:
          break

    return pd.DataFrame (dt_eeg)
    
def read_random_eeg_files(directory, entity_subject, count_subj, verbose=True):
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
            'EEG': EEG_Data(name_subject=file, id_subject=id_subject,
                            class_subject=entity_subject, data=raw, DEBUG=verbose, sr=512)
        }
        eeg_data_list.append(eeg_data)
        id_subject += 1

    return pd.DataFrame(eeg_data_list)
def create_dataset (directory, label, count_subj, verbose = True):
    """
    Crea un dataset con los datos de EEG de todos los archivos DBF en un directorio y agrega una columna con la etiqueta.

    :param directory : Ruta al directorio que contiene los archivos DBF.
    :param label     : Etiqueta que se agregará como columna al dataset.

    :return: DataFrame con los datos de EEG y la etiqueta.
    """
    eeg =  read_random_eeg_files (directory, label, count_subj, verbose = verbose) # read_eeg_data (directory, label, count_subj, verbose = verbose)
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
 
def load_data (direct_Younger = "./Younger",direct_Older = './Older', n_y_subject = 12, n_o_subject = 12, verbose = True):
    """
        Carga los datos de EEG de sujetos jóvenes y mayores desde los directorios especificados.
    
        Parámetros:
        direct_Younger (str): Directorio que contiene los datos de los sujetos jóvenes.
        direct_Older (str): Directorio que contiene los datos de los sujetos mayores.
        n_y_subject (int): Número de sujetos jóvenes a cargar.
        n_o_subject (int): Número de sujetos mayores a cargar.
        verbose (bool): Si es True, imprime mensajes de progreso.
    
        Retorna:
        dataset_Younger (DataFrame): Conjunto de datos de los sujetos jóvenes.
        dataset_Older (DataFrame): Conjunto de datos de los sujetos mayores.
    
        Descripción:
        - Utiliza la función `create_dataset` para cargar los datos.
        - Imprime mensajes de progreso si `verbose` es True.
    """
    if verbose:
        print ("Cargamos dataset Younger")
    dataset_Younger = create_dataset (direct_Younger, label = 'Younger', count_subj = n_y_subject, verbose = verbose)
    if verbose:
        print ("Cargamos dataset Older")
    dataset_Older   = create_dataset (direct_Older, label = 'Older', count_subj = n_o_subject, verbose = verbose)

    return dataset_Younger, dataset_Older
 
def assing_channels_dt (dt_subject, all_channels, eeg_channels, eog_channels, exg_channels,exg_type = 'emg'):
    """
        Asigna los nombres de los canales a los datos EEG de cada sujeto en el DataFrame.
    
        Parámetros:
        dt_subject (DataFrame): Conjunto de datos de los sujetos.
        all_channels (list): Lista completa de nombres de canales.
        eeg_channels (list): Lista de nombres de canales EEG.
        eog_channels (list): Lista de nombres de canales EOG.
        exg_channels (list): Lista de nombres de canales EXG.
        exg_type (str): Tipo de canal EXG, por defecto 'emg'.
    
        Retorna:
        None
    
        Descripción:
        - Recorre cada sujeto en el DataFrame y asigna los nombres de los canales especificados.
        - Utiliza el método `assign_channel_names` de los datos EEG.
    """
    n_subjt = dt_subject.shape [0]

    for i in range (n_subjt):
        dt_subject.iloc [i] ['EEG'].assign_channel_names (all_channels, eeg_channels, eog_channels, exg_channels, exg_type = exg_type)

def filtering_dt (dt_subject,cut_low = 16, n_decim = 2):
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
        dt_subject.iloc [i] ['EEG'].channel_filtered (cut_low = 16)
        dt_subject.iloc [i] ['EEG'].resample_mne (n_decim = n_decim)

def ica_artifact (dt_subject):
    """
        Aplica la eliminación de artefactos en los datos EEG usando Análisis de Componentes Independientes (ICA).
    
        Parámetros:
        dt_subject (DataFrame): Conjunto de datos de los sujetos.
    
        Retorna:
        None
    
        Descripción:
        - Utiliza ICA para identificar y eliminar componentes de artefactos en los datos EEG.
        - Procesa los datos EEG de cada sujeto en el DataFrame.
    """
    n_subjt_y = dt_subject.shape [0] # 4
    for i in range (n_subjt_y):
        dt_subject.iloc [i] ['EEG'].remove_ica_components_artifact ()


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