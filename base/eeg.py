from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from tslearn.metrics import dtw
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, v_measure_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import pywt

import mne
from mne import io
from mne.datasets import refmeg_noise
from mne.viz import plot_alignment
from mne.preprocessing import ICA
from mne.preprocessing import regress_artifact

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from nilearn import plotting
from scipy.signal import resample
from scipy.signal import welch
from sklearn.metrics import homogeneity_score

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
  def _assign_channel_type (self, all_channels, eeg_channels, eog_channels, exg_channels, exg_type = "ecg", montage='biosemi64'):
    self.all_channels = all_channels
    self.eeg_channels = eeg_channels
    self.eog_channels = eog_channels
    self.exg_channels = exg_channels

    # Definir los nombres de los canales EEG, EOG y EXG
    ch_names = all_channels

    # Aquí puedes decidir cómo clasificar los canales EXG (ecg, emg, misc, etc.)
    ch_types  = ['eeg'] * len(eeg_channels) + ['eog'] * len(eog_channels)
    exg_type  = exg_type  # Cambiamos 'emg' o 'misc' según lo necesites
    ch_types += [exg_type] * len(exg_channels) + ['misc']

    # Crear un montaje personalizado
    montage = mne.channels.make_standard_montage(montage)

    # Obtener los nombres de los electrodos del montaje
    electrodos_disponibles = montage.ch_names
    
    if self._DEBUG: print(f"Electrodos disponibles en el montaje {montage}: {electrodos_disponibles}")

    raw_m = self.get_raw()

    # Crear un objeto Info
    info = mne.create_info(
        ch_names=ch_names,  # Lista de nombres de canales
        ch_types=ch_types,  # Lista de tipos de canales ('eeg', 'eog', 'ecg', etc.)
        sfreq=512.0
    )

    # Asignar el montaje al objeto Info
    info.set_montage(montage)

    # Obtener los índices de los canales de interés
    picks = mne.pick_channels(info['ch_names'], include=eeg_channels)
    if self._DEBUG: print("Índices de los canales seleccionados:", picks)
    if self._DEBUG: print(f'raw_m.get_montage:{raw_m.get_montage()}')
    info.set_meas_date(raw_m.info['meas_date'])

    self._info = info

    # Crear un objeto Raw con tus datos brutos
    raw = mne.io.RawArray(raw_m.get_data(), info)
    if self._DEBUG: print(raw.info)
    
    self.set_raw(raw)

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
      Returns:
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
      n_components = min(len(self.eeg_channels), 0.999)  # 99.9% de varianza explicada
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
      var_thresh     = 3.0  # Umbral de Z-score para considerar alta varianza
      sources        = ica.get_sources(raw_c).get_data()  # Obtener los datos como array NumPy
      component_vars = np.var(sources, axis=1)  # Calcular la varianza a lo largo del tiempo
      z_scores       = (component_vars - np.mean(component_vars)) / np.std(component_vars)
      high_var_indices = np.where(z_scores > var_thresh)[0]
      if self._DEBUG: print("high_var_indices:", high_var_indices.tolist())
      ica.exclude.extend(high_var_indices.tolist())

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
    num_subjects, num_samples, num_neurons = internal_representations.shape
    flattened_representations = internal_representations.reshape (num_subjects, num_samples * num_neurons)
    return flattened_representations

def extract_features(internal_representations):
    num_subjects, num_samples, num_neurons = internal_representations.shape
    features = np.zeros((num_subjects, num_neurons * 2))  # Media y varianza para cada neurona

    for i in range(num_subjects):
        subject_data = internal_representations[i]
        mean_features = subject_data.mean(axis=0)
        var_features = subject_data.var(axis=0)
        features[i, :num_neurons] = mean_features
        features[i, num_neurons:] = var_features

    return features
##CARGAMOS LOS DATOS
def load_data (direct_Younger = "./Younger",direct_Older = './Older', n_y_subject = 12, n_o_subject = 12, verbose = True):
    # ## 1. Cargamos los datos
    if verbose:
        print ("Cargamos dataset Younger")
    dataset_Younger = create_dataset (direct_Younger, label = 'Younger', count_subj = n_y_subject, verbose = verbose)
    if verbose:
        print ("Cargamos dataset Older")
    dataset_Older   = create_dataset (direct_Older, label = 'Older', count_subj = n_o_subject, verbose = verbose)

    return dataset_Younger, dataset_Older
## Preprocesado señales EEG:
### Creamos los dataset young/old con su EEG completo
def assing_channels_dt (dt_subject, all_channels, eeg_channels, eog_channels, exg_channels):
  n_subjt = dt_subject.shape [0]

  for i in range (n_subjt):
      dt_subject.iloc [i] ['EEG'].assign_channel_names (all_channels, eeg_channels, eog_channels, exg_channels, exg_type = 'ecg')

def filtering_dt (dt_subject,cut_low = 16, n_decim = 2):
  n_subjt = dt_subject.shape [0]

  for i in range (n_subjt):
      dt_subject.iloc [i] ['EEG'].channel_filtered (cut_low = 16)
      dt_subject.iloc [i] ['EEG'].resample_mne (n_decim = n_decim)

def ica_artifact (dt_subject):
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
