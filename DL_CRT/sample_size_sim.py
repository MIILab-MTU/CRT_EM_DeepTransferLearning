# Seeds
import numpy as np
import os, os.path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

import random as python_random
from imblearn.over_sampling import SMOTE
import time
# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

# TDA
import gtda.mapper 
from sklearn.utils import shuffle

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover,
    OneDimensionalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Sklearn_TDA and Stat_mapper 

from sklearn.cluster import AgglomerativeClustering, KMeans
from itertools import product

# Persistent Homology Machine Learning 
from gtda.homology import CubicalPersistence
import tensorflow as tf
from matplotlib import pyplot as plt
from gtda.images import DensityFiltration
import keras_tuner
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.feature_selection import RFE, f_classif, SelectKBest, VarianceThreshold, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn import metrics
from sklearn import model_selection
from gtda.diagrams import Scaler

def reset_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed) 
    python_random.seed(seed)
    tf.random.set_seed(1)
    
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from keras import callbacks
# Keras model things
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import concatenate, Input, Dropout, BatchNormalization, SpatialDropout2D, Lambda, Conv2DTranspose, Reshape, UpSampling2D
import tensorflow_addons as tfa
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.models import load_model
from keras.losses import binary_crossentropy
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
import scipy.stats as stats

os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
##os.environ['TF_DETERMINISTIC_OPS'] = 'true'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()  ### added
tf.compat.v1.enable_eager_execution() 

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from matplotlib import pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Computer
directory = ''

df_og = pd.read_csv(directory + '')
df_NoImp = pd.read_csv(directory + '')

reset_seeds(123)
def clean_data(dataframe, iaea):
    # filter by  death 
    dataframe = dataframe[np.array(dataframe['Death'] == 0)]
    
    # subset iaea
    
    ids = dataframe['ID']
    if iaea:
        length_checker = np.vectorize(len)
        index = length_checker(ids) >= 4
        dataframe = dataframe[index]

    
    
    response = dataframe['response']
    # Remove for high NA
    # Echo_vars, AF, CKD, Ischemia, Metroprolo
    categorical = ['ACEI_or_ARB','CABG','CAD','Concordance','DM','Gender','HTN','Hospitalization','LBBB', 'MI','NYHA','PCI','Race', 'ID', 'Smoking']

    continuous = [ 'ECG_pre_QRSd', 'SPECT_pre_LVEF', 'SPECT_pre_ESV', 'SPECT_pre_EDV','SPECT_pre_WTper',
              'SPECT_pre_PSD', 'SPECT_pre_PBW', 'Age','SPECT_pre_50scar', 
              'SPECT_pre_gMyoMass', 'SPECT_pre_WTsum', 'SPECT_pre_StrokeVolume', 'SPECT_pre_PhaseKurt',
              'SPECT_pre_Diastolic_PhasePeak', 'SPECT_pre_Diastolic_PhaseSkew','SPECT_pre_Diastolic_PBW' , 'SPECT_pre_Diastolic_PSD',
              'SPECT_pre_EDSI', 'SPECT_pre_EDE', 'SPECT_pre_ESSI', 'SPECT_pre_ESE', 'SPECT_pre_Diastolic_PhaseKurt', 'SPECT_pre_PhasePeak', 'SPECT_pre_SRscore']
    
    # Create continuous resonse 
    dataframe = dataframe.dropna(subset = 'SPECT_post_LVEF')
    response_vector = dataframe['SPECT_post_LVEF'] - dataframe['SPECT_pre_LVEF']
    
    # Remove post values 
    dataframe = pd.concat([dataframe[continuous],dataframe[categorical]], axis = 1)
    
    # Add response 
    dataframe.insert(0, 'resp', response_vector)
    dataframe.insert(0, 'response', response)

    # Drop NA
    dataframe = dataframe.dropna(how = 'any')
    #dataframe = dataframe.astype(int)
    response = dataframe['response']
    #id label for images 
    ids = dataframe['ID']
 
    del dataframe['response']
    del dataframe['ID']
       
    # dummify nyha and race
    nyha_race_dummy = pd.get_dummies(dataframe[['NYHA','Race']].astype(str))
    
    #Combine Dataframe with all features
    
    dataframe = pd.concat([nyha_race_dummy, dataframe], axis = 1)
    super_response = np.where(dataframe['resp'] >= 15, 1, 0)
    return dataframe, response, ids, super_response

df, response, ids, super_response = clean_data(df_og, False)
#color_data  = pd.get_dummies(response, prefix="response") # color discrete 
color_data  = df['resp']

                          
#del df['resp'] # the continuous normal response 

def norm(image_array):
    return image_array / 255

input_shape = 64
input_shape = (input_shape, input_shape)

def load_image(identification):
    perf_image_array = []
    wallthk_image_array = []
    systolic_image_array = []
    
    for idd in identification:
        patient_id = idd

        perf_path = directory + patient_id + '_perfusion.png'
        systolic_path = directory + patient_id + '_systolicPhase.png'
        wallthk_path = directory + patient_id + '_wallthk.png'

        perf_image = tf.keras.preprocessing.image.load_img(perf_path, color_mode = 'grayscale', target_size = input_shape)
        systolic_image = tf.keras.preprocessing.image.load_img(systolic_path, color_mode = 'grayscale', target_size = input_shape)
        wallthk_image = tf.keras.preprocessing.image.load_img(wallthk_path, color_mode = 'grayscale', target_size = input_shape)

        perf_image = tf.keras.preprocessing.image.img_to_array(perf_image, data_format = "channels_last")
        systolic_image = tf.keras.preprocessing.image.img_to_array(systolic_image, data_format = "channels_last")
        wallthk_image = tf.keras.preprocessing.image.img_to_array(wallthk_image, data_format = "channels_last")

        perf_image = norm(perf_image)
        systolic_image = norm(systolic_image)
        wallthk_image = norm(wallthk_image)
        
        perf_image_array.append(perf_image.reshape((1,64,64)))
        systolic_image_array.append(systolic_image.reshape((1,64,64)))
        wallthk_image_array.append(wallthk_image.reshape((1,64,64)))

    perf_conc = np.concatenate(perf_image_array, axis = 0)
    syst_conc = np.concatenate(systolic_image_array, axis = 0)
    wall_conc = np.concatenate(wallthk_image_array, axis = 0)

    conc_img = np.concatenate([perf_image_array, systolic_image_array, wallthk_image_array], axis = 1)
    print('max',np.max(conc_img))
    print('min',np.min(conc_img))
    conc_redo = np.concatenate([np.expand_dims(perf_conc, axis = 0),np.expand_dims(syst_conc, axis = 0), np.expand_dims(wall_conc, axis = 0)], axis = 0)

    return perf_image_array, systolic_image_array, wallthk_image_array, np.transpose(conc_img, [0,2,3,1]), np.transpose(conc_redo, [1,2,3,0])
perf, syst, wall, conc, conc2 = load_image(ids)

from gtda.images import Binarizer, RadialFiltration, ImageToPointCloud, DensityFiltration
from gtda.homology import VietorisRipsPersistence

tpe = ['Gray', 'TGray', 'bin', 'den', 'rips', 'rad']
def filtration(imgs, filterr, bin_threshold):
    
    if filterr == 'Gray':
        output = imgs
        
    elif filterr == 'TGray':
        new_img = []
        for i in range(len(imgs)):
            new_img.append(np.max(imgs[i]) - imgs[i])
        output = new_img 
        
    elif filterr == 'bin':
        binarizer = Binarizer(threshold=bin_threshold)
        output = binarizer.fit_transform(imgs)
        
    elif filterr == 'den':
        binarizer = Binarizer(threshold=bin_threshold)
        bin_img = binarizer.fit_transform(imgs)
        
        den_filter = DensityFiltration()
        output = den_filter.fit_transform(bin_img)
        
    elif filterr == 'rad':
        binarizer = Binarizer(threshold=bin_threshold)
        bin_img = binarizer.fit_transform(imgs)
        
        rad_filter = RadialFiltration(center = np.array([1,31,31]))
        output = rad_filter.fit_transform(bin_img)
    
    
    return output

def view_filt_img(imgs, index):
    imgs = np.array(imgs)
    print('Filtered Image')
    plt.imshow(np.squeeze(imgs[index,:,:,:]))
    plt.colorbar()
    plt.show()
    
    print('persistance Diagram')
    plt.imshow(np.squeeze(imgs[index+1,:,:,:]))
    plt.colorbar()
    plt.show()
          
# filt_img = filtration(perf, 'TGray', 0.7)
# view_filt_img(filt_img,0)

color_map = [
    [0.000000, 0.000000, 0.000000],
    [0.000000, 0.007843, 0.007843],
    [0.000000, 0.015686, 0.015686],
    [0.000000, 0.023529, 0.023529],
    [0.000000, 0.031373, 0.031373],
    [0.000000, 0.039216, 0.039216],
    [0.000000, 0.047059, 0.047059],
    [0.000000, 0.054902, 0.054902],
    [0.000000, 0.062745, 0.062745],
    [0.000000, 0.070588, 0.070588],
    [0.000000, 0.078431, 0.078431],
    [0.000000, 0.086275, 0.086275],
    [0.000000, 0.094118, 0.094118],
    [0.000000, 0.101961, 0.101961],
    [0.000000, 0.109804, 0.109804],
    [0.000000, 0.117647, 0.117647],
    [0.000000, 0.129412, 0.125490],
    [0.000000, 0.137255, 0.133333],
    [0.000000, 0.145098, 0.141176],
    [0.000000, 0.152941, 0.149020],
    [0.000000, 0.160784, 0.156863],
    [0.000000, 0.168627, 0.164706],
    [0.000000, 0.176471, 0.172549],
    [0.000000, 0.184314, 0.180392],
    [0.000000, 0.192157, 0.188235],
    [0.000000, 0.200000, 0.196078],
    [0.000000, 0.207843, 0.203922],
    [0.000000, 0.215686, 0.211765],
    [0.000000, 0.223529, 0.219608],
    [0.000000, 0.231373, 0.227451],
    [0.000000, 0.239216, 0.235294],
    [0.000000, 0.247059, 0.243137],
    [0.000000, 0.254902, 0.250980],
    [0.000000, 0.262745, 0.258824],
    [0.000000, 0.270588, 0.266667],
    [0.000000, 0.278431, 0.274510],
    [0.000000, 0.286275, 0.282353],
    [0.000000, 0.294118, 0.290196],
    [0.000000, 0.301961, 0.298039],
    [0.000000, 0.309804, 0.305882],
    [0.000000, 0.317647, 0.313725],
    [0.000000, 0.325490, 0.321569],
    [0.000000, 0.333333, 0.329412],
    [0.000000, 0.341176, 0.337255],
    [0.000000, 0.349020, 0.345098],
    [0.000000, 0.356863, 0.352941],
    [0.000000, 0.364706, 0.360784],
    [0.000000, 0.372549, 0.368627],
    [0.000000, 0.384314, 0.376471],
    [0.000000, 0.392157, 0.384314],
    [0.000000, 0.400000, 0.392157],
    [0.000000, 0.407843, 0.400000],
    [0.000000, 0.415686, 0.407843],
    [0.000000, 0.423529, 0.415686],
    [0.000000, 0.431373, 0.423529],
    [0.000000, 0.439216, 0.431373],
    [0.000000, 0.447059, 0.439216],
    [0.000000, 0.454902, 0.447059],
    [0.000000, 0.462745, 0.454902],
    [0.000000, 0.470588, 0.462745],
    [0.000000, 0.478431, 0.470588],
    [0.000000, 0.486275, 0.478431],
    [0.000000, 0.494118, 0.486275],
    [0.000000, 0.501961, 0.494118],
    [0.007843, 0.494118, 0.501961],
    [0.015686, 0.486275, 0.505882],
    [0.023529, 0.478431, 0.513725],
    [0.031373, 0.470588, 0.521569],
    [0.039216, 0.462745, 0.529412],
    [0.047059, 0.454902, 0.537255],
    [0.054902, 0.447059, 0.545098],
    [0.062745, 0.439216, 0.552941],
    [0.070588, 0.431373, 0.560784],
    [0.078431, 0.423529, 0.568627],
    [0.086275, 0.415686, 0.576471],
    [0.094118, 0.407843, 0.584314],
    [0.101961, 0.400000, 0.592157],
    [0.109804, 0.392157, 0.600000],
    [0.117647, 0.384314, 0.607843],
    [0.125490, 0.376471, 0.615686],
    [0.133333, 0.372549, 0.623529],
    [0.141176, 0.364706, 0.631373],
    [0.149020, 0.356863, 0.639216],
    [0.156863, 0.349020, 0.647059],
    [0.164706, 0.341176, 0.654902],
    [0.168627, 0.333333, 0.662745],
    [0.176471, 0.325490, 0.670588],
    [0.184314, 0.317647, 0.678431],
    [0.192157, 0.309804, 0.686275],
    [0.200000, 0.301961, 0.694118],
    [0.207843, 0.294118, 0.701961],
    [0.215686, 0.286275, 0.709804],
    [0.223529, 0.278431, 0.717647],
    [0.231373, 0.270588, 0.725490],
    [0.239216, 0.262745, 0.733333],
    [0.247059, 0.254902, 0.741176],
    [0.254902, 0.247059, 0.749020],
    [0.262745, 0.239216, 0.756863],
    [0.270588, 0.231373, 0.764706],
    [0.278431, 0.223529, 0.772549],
    [0.286275, 0.215686, 0.780392],
    [0.294118, 0.207843, 0.788235],
    [0.301961, 0.200000, 0.796078],
    [0.309804, 0.192157, 0.803922],
    [0.317647, 0.184314, 0.811765],
    [0.325490, 0.176471, 0.819608],
    [0.333333, 0.168627, 0.827451],
    [0.341176, 0.160784, 0.835294],
    [0.349020, 0.152941, 0.843137],
    [0.356863, 0.145098, 0.850980],
    [0.364706, 0.137255, 0.858824],
    [0.372549, 0.129412, 0.866667],
    [0.380392, 0.125490, 0.874510],
    [0.388235, 0.117647, 0.882353],
    [0.396078, 0.109804, 0.890196],
    [0.403922, 0.101961, 0.898039],
    [0.411765, 0.094118, 0.905882],
    [0.419608, 0.086275, 0.913725],
    [0.427451, 0.078431, 0.921569],
    [0.435294, 0.070588, 0.929412],
    [0.443137, 0.062745, 0.937255],
    [0.450980, 0.054902, 0.945098],
    [0.458824, 0.047059, 0.952941],
    [0.466667, 0.039216, 0.960784],
    [0.474510, 0.031373, 0.968627],
    [0.482353, 0.023529, 0.976471],
    [0.490196, 0.015686, 0.984314],
    [0.498039, 0.007843, 0.992157],
    [0.501961, 0.000000, 1.000000],
    [0.509804, 0.007843, 0.984314],
    [0.517647, 0.015686, 0.968627],
    [0.525490, 0.023529, 0.952941],
    [0.533333, 0.031373, 0.937255],
    [0.541176, 0.039216, 0.921569],
    [0.549020, 0.047059, 0.905882],
    [0.556863, 0.054902, 0.890196],
    [0.564706, 0.062745, 0.874510],
    [0.572549, 0.070588, 0.858824],
    [0.580392, 0.078431, 0.843137],
    [0.588235, 0.086275, 0.827451],
    [0.596078, 0.094118, 0.811765],
    [0.603922, 0.101961, 0.796078],
    [0.611765, 0.109804, 0.780392],
    [0.619608, 0.117647, 0.764706],
    [0.627451, 0.125490, 0.749020],
    [0.635294, 0.133333, 0.733333],
    [0.643137, 0.141176, 0.717647],
    [0.650980, 0.149020, 0.701961],
    [0.658824, 0.156863, 0.686275],
    [0.666667, 0.164706, 0.670588],
    [0.674510, 0.172549, 0.654902],
    [0.682353, 0.180392, 0.639216],
    [0.690196, 0.188235, 0.623529],
    [0.698039, 0.196078, 0.607843],
    [0.705882, 0.203922, 0.592157],
    [0.713725, 0.211765, 0.576471],
    [0.721569, 0.219608, 0.560784],
    [0.729412, 0.227451, 0.545098],
    [0.737255, 0.235294, 0.529412],
    [0.745098, 0.243137, 0.513725],
    [0.752941, 0.250980, 0.501961],
    [0.760784, 0.258824, 0.486275],
    [0.768627, 0.266667, 0.470588],
    [0.776471, 0.274510, 0.454902],
    [0.784314, 0.282353, 0.439216],
    [0.792157, 0.290196, 0.423529],
    [0.800000, 0.298039, 0.407843],
    [0.807843, 0.305882, 0.392157],
    [0.815686, 0.313725, 0.376471],
    [0.823529, 0.321569, 0.360784],
    [0.831373, 0.329412, 0.345098],
    [0.835294, 0.337255, 0.329412],
    [0.843137, 0.345098, 0.313725],
    [0.850980, 0.352941, 0.298039],
    [0.858824, 0.360784, 0.282353],
    [0.866667, 0.368627, 0.266667],
    [0.874510, 0.376471, 0.250980],
    [0.882353, 0.384314, 0.235294],
    [0.890196, 0.392157, 0.219608],
    [0.898039, 0.400000, 0.203922],
    [0.905882, 0.407843, 0.188235],
    [0.913725, 0.415686, 0.172549],
    [0.921569, 0.423529, 0.156863],
    [0.929412, 0.431373, 0.141176],
    [0.937255, 0.439216, 0.125490],
    [0.945098, 0.447059, 0.109804],
    [0.952941, 0.454902, 0.094118],
    [0.960784, 0.462745, 0.078431],
    [0.968627, 0.470588, 0.062745],
    [0.976471, 0.478431, 0.047059],
    [0.984314, 0.486275, 0.031373],
    [0.992157, 0.494118, 0.015686],
    [1.000000, 0.505882, 0.000000],
    [1.000000, 0.513725, 0.015686],
    [1.000000, 0.521569, 0.031373],
    [1.000000, 0.529412, 0.047059],
    [1.000000, 0.537255, 0.062745],
    [1.000000, 0.545098, 0.078431],
    [1.000000, 0.552941, 0.094118],
    [1.000000, 0.560784, 0.109804],
    [1.000000, 0.568627, 0.125490],
    [1.000000, 0.576471, 0.141176],
    [1.000000, 0.584314, 0.156863],
    [1.000000, 0.592157, 0.176471],
    [1.000000, 0.600000, 0.192157],
    [1.000000, 0.607843, 0.207843],
    [1.000000, 0.615686, 0.223529],
    [1.000000, 0.623529, 0.239216],
    [1.000000, 0.631373, 0.254902],
    [1.000000, 0.639216, 0.270588],
    [1.000000, 0.647059, 0.286275],
    [1.000000, 0.654902, 0.301961],
    [1.000000, 0.662745, 0.317647],
    [1.000000, 0.670588, 0.333333],
    [1.000000, 0.678431, 0.349020],
    [1.000000, 0.686275, 0.364706],
    [1.000000, 0.694118, 0.380392],
    [1.000000, 0.701961, 0.396078],
    [1.000000, 0.709804, 0.411765],
    [1.000000, 0.717647, 0.427451],
    [1.000000, 0.725490, 0.443137],
    [1.000000, 0.733333, 0.458824],
    [1.000000, 0.741176, 0.474510],
    [1.000000, 0.749020, 0.490196],
    [1.000000, 0.756863, 0.509804],
    [1.000000, 0.764706, 0.525490],
    [1.000000, 0.772549, 0.541176],
    [1.000000, 0.780392, 0.556863],
    [1.000000, 0.788235, 0.572549],
    [1.000000, 0.796078, 0.588235],
    [1.000000, 0.803922, 0.603922],
    [1.000000, 0.811765, 0.619608],
    [1.000000, 0.819608, 0.635294],
    [1.000000, 0.827451, 0.650980],
    [1.000000, 0.835294, 0.666667],
    [1.000000, 0.843137, 0.682353],
    [1.000000, 0.850980, 0.698039],
    [1.000000, 0.858824, 0.713725],
    [1.000000, 0.866667, 0.729412],
    [1.000000, 0.874510, 0.745098],
    [1.000000, 0.882353, 0.760784],
    [1.000000, 0.890196, 0.776471],
    [1.000000, 0.898039, 0.792157],
    [1.000000, 0.905882, 0.807843],
    [1.000000, 0.913725, 0.823529],
    [1.000000, 0.921569, 0.843137],
    [1.000000, 0.929412, 0.858824],
    [1.000000, 0.937255, 0.874510],
    [1.000000, 0.945098, 0.890196],
    [1.000000, 0.952941, 0.905882],
    [1.000000, 0.960784, 0.921569],
    [1.000000, 0.968627, 0.937255],
    [1.000000, 0.976471, 0.952941],
    [1.000000, 0.984314, 0.968627],
    [1.000000, 0.992157, 0.984314],
    [1.000000, 1.000000, 1.000000]]

def grayscale_to_rgb(grayscale_image, colormap):
    grayscale_image = np.array(grayscale_image)
    grayscale_image = grayscale_image * 255

    # Check that the grayscale image has values that range from 0 to 255
    # and that the colormap has a length of 256
    # if grayscale_image.min() < 0 or grayscale_image.max() > 255:
    #     raise ValueError('Grayscale values must be in the range 0-255')
    if len(colormap) != 256:
        raise ValueError('Colormap must have a length of 256')

    # Create an empty RGB image
    rgb_image = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], grayscale_image.shape[2], 3), dtype=np.uint8)

    # Iterate over the grayscale image and set the pixel values in the RGB image
    for k in range(grayscale_image.shape[0]):
        for i in range(grayscale_image.shape[1]):
            for j in range(grayscale_image.shape[2]):
                # Lookup the corresponding RGB value in the colormap
                r, g, b = colormap[int(grayscale_image[k,i, j, 0] )] 
                # Set the pixel value in the RGB image
                rgb_image[k,i, j, 0] = r * (int(grayscale_image[k,i, j, 0]))
                rgb_image[k,i, j, 1] = g * (int(grayscale_image[k,i, j, 0]))
                rgb_image[k,i, j, 2] = b * (int(grayscale_image[k, i, j, 0]))

  # Return the RGB image
    return rgb_image

reset_seeds(123)
a = 2 ** np.arange(10)
a = a[1:9].tolist()
from sklearn.linear_model import Perceptron

def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """
    df = pd.DataFrame(df)
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)


    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def weights(label):
    neg, pos = np.bincount(np.squeeze(label))
    total = neg + pos 
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0 : weight_for_0, 1: weight_for_1}
    return class_weight

def sensitivity(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def rfe_fs(clinical, label, num_features, automatic):
    if automatic:
        model = Perceptron(random_state=0)
        rfecv = RFECV(
            estimator=model,
            min_features_to_select=10,
            step=1,
            cv=7)
        fit = rfecv.fit(clinical, np.squeeze(label))

    return np.array(clinical[:,fit.support_.tolist()]), fit.support_.tolist()

        

class pretune_CVTuner(keras_tuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, **kwargs):
        reset_seeds(123)
        
        val_objective = []
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 30, 150, step = 7, default = 100)
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 2, 35, step = 3, default = 15)
        
        val_objective = []
#         #for train_indices, test_indices in StratifiedKFold(n_splits=8, shuffle = True, random_state = 123).split(x[0], y):
        for train_indices, test_indices in StratifiedShuffleSplit(n_splits=4, test_size = 1/10,  random_state = 123).split(x[0], y):
        
            model = self.hypermodel.build(trial.hyperparameters)

            model.fit([x[0][train_indices], x[1][train_indices]], y[train_indices], epochs=kwargs['epochs'], verbose = 0,
                      batch_size = kwargs['batch_size'],
                    shuffle = True,
                    class_weight = class_weight)
            val_objective.append(model.evaluate([x[0][test_indices],x[1][test_indices]], y[test_indices]))

        # AUC 
        self.oracle.update_trial(trial.trial_id, {'val_auc': np.mean(val_objective, axis = 0)[2]})
        
def tune_pre(hp):
################################################################################################### transfer learn
    hp_activ = hp.Choice('activation', values = ['elu', 'leakyRelu'])

    with hp.conditional_scope('activation', ['leakyRelu']):
        hp_activ = tf.keras.layers.LeakyReLU(alpha = 0.01)
    initializer = tf.keras.initializers.HeNormal(seed = 123)

    ######### images 
    img = Input(shape=(128, 128,3), name = 'img_features')
    x = img
    
    base_model = tf.keras.applications.VGG16(input_shape=(128,128,3),
                                                include_top=False,
                                                weights='imagenet',
                                                input_tensor = x,
                                                pooling = 'avg')

    base_model.trainable = False 

    x = Dropout(hp.Float(f"csv_dropout {1}",
                    min_value=0.2,
                    max_value=0.5,
                    default=0.3,
                    step=0.1), seed = 123)(base_model.output)
    
    a = [16,32,64,128,256]    
    
    ########### clinical + topological 
    y1 = Input(shape=(var_len), name = 'csv_features')
    y = y1 

    conv_dense = hp.Int("n_clinical_dense", 0, 1)
    if conv_dense == 1:
        with hp.conditional_scope('n_clinical_dense',1):
            y = Dense(hp.Choice(f"clinical_dense { 1} units", a), activation = hp_activ,
                   kernel_initializer=initializer,
                   kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f"clinical_dense {1} decay", min_value = 0.00001, max_value = .001, sampling = 'LOG', default = 0.001)))(y)
            y = Dropout(hp.Float(
                f"clinical_dense_dropout {1}",
                min_value=0.0,
                max_value=0.5,
                default=0.2,
                step=0.05), seed = 123)(y)      


    j = concatenate([x, y])

    ############################## concatenated 

    j = Dense(hp.Choice(f"csv_dense { 1} units", a), activation = hp_activ, 
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f"csv_dense {1} decay", min_value = 0.00001, max_value = .001, sampling = 'LOG', default = 0.001)))(j)
    j = Dropout(hp.Float(
            f"csv_dropout {1}",
            min_value=0.0,
            max_value=0.5,
            default=0,
            step=0.1), seed = 123)(j)

    outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(j)
    model = tf.keras.Model([img, y1], outputs)

    base_learning_rate = hp.Float('learning_rate',
                        min_value=.0000001,
                        max_value=.001,
                        sampling='LOG',
                        default= .0001)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                     tf.keras.metrics.AUC(name = 'auc', num_thresholds = 2000),
                    tfa.metrics.CohenKappa(num_classes = num_class, name = 'kappa'),
                    sensitivity,
                    specificity])

    return model 

# input dataframe and image array, returns train/test split 
def data_preprocess(dataframe, img, resp):
    clinical_data = dataframe.drop(columns = ['resp', 'Race', 'NYHA', 'Hospitalization'])
    X =  np.array(clinical_data)
    Y = np.where(np.array(resp)  == 'yes', 1, 0)    
   
    X = [img, X]
    
    
    
    for train, test in StratifiedShuffleSplit(n_splits=1 , test_size = 1/11,  random_state = 123).split(X[1], Y):  # equivalent to train/test split 
        x_train = [X[0][train], X[1][train]]
        y_train = Y[train]
        x_test = [X[0][test], X[1][test]]
        y_test = Y[test]
        
        resp_rate = sum(y_train)/len(y_train)
        
    return x_train, y_train, x_test, y_test, resp_rate

x_train, y_train, x_test, y_test, resp_rate = data_preprocess(df, conc, response)

def run_dl(x_train1, y_train1, x_test1, y_test1):
    
    scaler = preprocessing.StandardScaler().fit(x_train1[1])
    xt = scaler.transform(x_train1[1])
   
    xt, fs_column = rfe_fs(xt, y_train1, 80, True)
 
    x_test1[1] = x_test1[1][:,fs_column]
    x_train1[1] = x_train1[1][:,fs_column]
################################################################################################### Preprocess

    global class_weight
    class_weight = weights(y_train1)

    # Cor 
    corr_col = find_correlation(x_train1[1], .8)
    x_train1[1] = np.delete(x_train1[1], corr_col, axis = 1)
    nzv = VarianceThreshold(0.01).fit(x_train1[1])
    x_train1[1] = nzv.transform(x_train1[1])
    
    x_test1[1] = np.delete(x_test1[1], corr_col, axis = 1)
    x_test1[1] = nzv.transform(x_test1[1])
    
    #Standardize 
    scaler = preprocessing.StandardScaler().fit(x_train1[1])
    x_train1[1] = scaler.transform(x_train1[1])
    x_test1[1] = scaler.transform(x_test1[1])

# BUFFER ZEROS ON IMAGE 
# test
    perf_temp = np.expand_dims(x_test1[0][:,:,:,0], axis = -1)
    syst_temp = np.expand_dims(x_test1[0][:,:,:,1], axis = -1)
    wall_temp = np.expand_dims(x_test1[0][:,:,:,2], axis = -1)

    perf50_temp = np.where(perf_temp < 0.5, 0, perf_temp)
    #perf50_temp = perf_temp

    img = np.concatenate((perf_temp,syst_temp), axis = 1)
    img2 = np.concatenate((wall_temp, perf50_temp), axis = 1)
    x_test1[0] = np.concatenate((img, img2), axis = 2)
    x_test1[0] = np.tile(x_test1[0], 3)
    x_test1[0] = x_test1[0] * 255
    
# train
    # add images AND CONVERT TO RGB
    perf_temp = np.expand_dims(x_train1[0][:,:,:,0], axis = -1)
    syst_temp = np.expand_dims(x_train1[0][:,:,:,1], axis = -1)
    wall_temp = np.expand_dims(x_train1[0][:,:,:,2], axis = -1)

    perf50_temp = np.where(perf_temp < 0.5, 0, perf_temp)

    img = np.concatenate((perf_temp,syst_temp), axis = 1)
    img2 = np.concatenate((wall_temp, perf50_temp), axis = 1)
    x_train1[0] = np.concatenate((img, img2), axis = 2)
    x_train1[0] = np.tile(x_train1[0], 3)
    x_train1[0] = x_train1[0] * 255
    # print('train max',np.max(x_train[0]))
    # print('train min', np.min(x_train[0]))

    x_train1[0] = tf.keras.applications.vgg16.preprocess_input(x_train1[0])     
    x_test1[0] = tf.keras.applications.vgg16.preprocess_input(x_test1[0])    
    
    global var_len
    var_len = x_train1[1].shape[1]

# modeling 

    tuner = pretune_CVTuner(
    hypermodel = tune_pre,
    oracle=keras_tuner.oracles.BayesianOptimization(
    objective= keras_tuner.Objective('val_auc', direction = 'max'),
    max_trials=70, #70
    tune_new_entries = True,
    allow_new_entries = True,
    seed = 123),
    overwrite=True,
    directory = "CNN CRT",
    project_name = "keras_tuner_pre")
    
    
    tuner.search(x_train1, y_train1)
    global model
    model = tune_pre(tuner.get_best_hyperparameters(1)[0])
    
    # Final epoch model
    model.fit(x_train1, y_train1, 
        epochs = tuner.get_best_hyperparameters(num_trials=1)[0].get('epochs'),
        batch_size = tuner.get_best_hyperparameters(num_trials=1)[0].get('batch_size'),
        verbose = 0,
        class_weight = class_weight, 
        shuffle = True)
    
    return model, x_train1, y_train1, x_test1, y_test1

path = ''
train_sq = np.arange(48, 208, 10).tolist() #from 48 to 198 by 10's
num_class =2 

def simulation_psuedo_bootstrap(dataframe, images, response_status, save_path, train_seq, n_iter):
    
    # some way to store data across all repitions 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for train_size in train_seq:
        print('Training Sample Size: ', train_size)
        seed = 123
    
        # intermediate data storage for a specific repition
    
        for rep in range(n_iter):
            print('On Repition: ', rep)
    
            reset_seeds(seed + rep + 40)
            train_x, train_y, test_x, test_y, resp_r = data_preprocess(dataframe, images, response_status)


            ##### Perform psuedo boostrap
            index_of_response = np.where(train_y == 1, np.arange(198), -50 ).tolist()
            index_of_response = [x for x in index_of_response if x != -50]       # create list of index's that have response

            index_of_non_response = np.where(train_y == 0, np.arange(198), -50 ).tolist()
            index_of_non_response = [x for x in index_of_non_response if x != -50]  # create list of index's that have NON-response

            resp_size = round(resp_r * train_size)
            non_size = round((1-resp_r) * train_size)

            resp_index = np.random.choice(index_of_response , size = resp_size, replace = True).tolist() # psuedo bootstrap 
            non_resp_index = np.random.choice(index_of_non_response , size = non_size, replace = True).tolist() # psuedo bootstrap 

            index_useme = non_resp_index + resp_index  
                    
            ##### Reorient data for modeling 
            
            x_train_use = [train_x[0][index_useme], train_x[1][index_useme]]
            y_train_use = train_y[index_useme]

            ##### Plug into model 
            print('test_x[1] shape', test_x[1].shape)
            trained_model, x_train_use, y_train_use, x_test_use, y_test_use  = run_dl(x_train_use, y_train_use, test_x, test_y)
            
            train_eval = trained_model.evaluate(x_train_use, y_train_use) # loss, acc, auc, kappa, sens, spec, BSx6
            test_eval = trained_model.evaluate(x_test_use, y_test_use)

            train_eval = pd.DataFrame([train_eval])  # loss, acc, auc, kappa, sens, spec, BSx6
            test_eval = pd.DataFrame([test_eval])

            train_eval['sample_size'] = train_size
            test_eval['sample_size'] =  train_size
            
            train_eval['iteration'] = rep
            test_eval['iteration'] = rep
            
            train_eval.to_csv(save_path + 'train.csv', mode='a', header=not os.path.exists(save_path + 'train.csv') )
            test_eval.to_csv(save_path + 'test.csv', mode='a', header=not os.path.exists(save_path + 'test.csv') )
            
            del x_train_use, y_train_use, x_test_use, y_test_use, trained_model, train_eval, test_eval, train_x, train_y, test_x, test_y, resp_r


simulation_psuedo_bootstrap(df, conc, response, path,train_sq, n_iter = 5)


def plot_bootstrap_performance(data):
    data = data.rename({'0': 'loss', '1': 'Accuracy','2':'AUC','3':'Kappa','4': 'Sensitivity','5':'Specificity'}, axis = 1)

    mean_data = data.groupby('sample_size').mean().reset_index()
    std_data = data.groupby('sample_size').std().reset_index()
    
    # plotting 
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3.0)
    x = mean_data['sample_size']
    
    axs[0, 0].plot(x, mean_data['Accuracy'])
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_xlabel('Sample Size')
    axs[0, 0].fill_between(x, mean_data['Accuracy'] - std_data['Accuracy'], mean_data['Accuracy'] +  std_data['Accuracy'],alpha = 0.5)

    axs[0, 1].plot(x, mean_data['AUC'], 'tab:orange')
    axs[0, 1].set_title('AUC')
    axs[0, 1].set_xlabel('Sample Size')
    axs[0, 1].fill_between(x, mean_data['AUC'] - std_data['AUC'], mean_data['AUC'] +  std_data['AUC'],alpha = 0.5, color = 'orange')

    axs[1, 0].plot(x, mean_data['Sensitivity'], 'tab:green')
    axs[1, 0].set_title('Sensitivity')
    axs[1, 0].set_xlabel('Sample Size')
    axs[1, 0].fill_between(x, mean_data['Sensitivity'] - std_data['Sensitivity'], mean_data['Sensitivity'] +  std_data['Sensitivity'],alpha = 0.5, color = 'green')
    
    axs[1, 1].plot(x, mean_data['Specificity'], 'tab:red')
    axs[1, 1].set_title('Specificity')
    axs[1, 1].set_xlabel('Sample Size')
    axs[1, 1].fill_between(x, mean_data['Specificity'] - std_data['Specificity'], mean_data['Specificity'] +  std_data['Specificity'],alpha = 0.5, color = 'red')
    
    return fig

test_data = pd.read_csv('', usecols= ['0','1','2','3','4','5','sample_size','iteration'])
fig = plot_bootstrap_performance(test_data)