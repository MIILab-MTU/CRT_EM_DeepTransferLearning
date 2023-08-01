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


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Computer
directory = ''

df_og = pd.read_csv(directory + 'CNN_CRT_data.csv')
df_NoImp = pd.read_csv(directory + 'CNN_CRT_data.csv')

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img = np.expand_dims(img_array[0], axis = 0)
    tabular = np.expand_dims(img_array[1], axis = 0)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        #tape.watch(grad_model.get_layer('vgg16').get_layer(last_conv_layer_name).variables)
        inputs = [tf.cast(img,tf.float32), tf.cast(tabular,tf.float32)]
        last_conv_layer_output, preds = grad_model(inputs) 

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(preds, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 64)
y = np.arange(0, 64)
arr = np.zeros((y.size, x.size))

cx = 32.
cy = 32.
r = 32.

# The two lines below could be merged, but I stored the mask
# for code clarity.
mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
arr[mask] = True

big_arr = np.tile(arr, (2,2))
big_array = np.repeat(big_arr[:, :, np.newaxis], 3, axis = 2)
print(big_array.shape)
#plt.imshow(big_arr)

def save_and_display_gradcam(img_path, heatmap, alpha=0.5):
    # Load the original image

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_path.shape[1], img_path.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img1 = jet_heatmap * alpha + img_path * 255
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img1)

    # Display Grad CAM
    scale = 2
    display(superimposed_img.resize(( int(superimposed_img.width * scale), int(superimposed_img.height * scale))))
    # plt.imshow((img_path).astype(np.uint8), cmap='gray')
    # plt.show()

    img_path = keras.preprocessing.image.array_to_img(img_path)
    display(img_path.resize(( int(img_path.width * scale), int(img_path.height * scale))))

    superimposed_masked = np.where(big_array == True, superimposed_img1, 0)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_masked)
    scale = 2
    display(superimposed_img.resize(( int(superimposed_img.width * scale), int(superimposed_img.height * scale))))

    #display(superimposed_img, width = 1000)
    
    reset_seeds(123)
import matplotlib.cm as cm

PATH_TO_MODEL = ''
custom_objects={"sensitivity": sensitivity,
                    'specificity': specificity}
    
model = load_model(PATH_TO_MODEL, custom_objects = custom_objects)

last_conv_layer_name = "block5_conv3"
#model.layers[-1].activation = 'sigmoid'

# Print what the top predicted class is
#preds = model.predict([x_test_data[0][0], x_test_data[1][0]])
model.layers[-1].activation = None



reset_seeds(123)
for train, test in StratifiedShuffleSplit(n_splits=1 , test_size = 1/11,  random_state = 123).split(df, response):
    a = train
    b = test
    conc_test = conc[test]

perf_temp = np.expand_dims(conc_test[:,:,:,0], axis = -1)
syst_temp = np.expand_dims(conc_test[:,:,:,1], axis = -1)
wall_temp = np.expand_dims(conc_test[:,:,:,2], axis = -1)


perf50_temp = np.where(perf_temp < 0.5, 0, perf_temp)


img = np.concatenate((perf_temp,syst_temp), axis = 1)
img2 = np.concatenate((wall_temp, perf50_temp), axis = 1)
bruh = np.concatenate((img, img2), axis = 2)
bruh = np.tile(bruh, 3)

class_1 = (df['SPECT_pre_LVEF'] <= 35)  & (df['ECG_pre_QRSd'] >= 150) & (df['LBBB'] == 1) & (df['NYHA'] >= 2 )
class_2a1 = (df['SPECT_pre_LVEF'] <= 35)  & (df['ECG_pre_QRSd'] <= 149) & (df['ECG_pre_QRSd'] >= 120) & (df['LBBB'] == 1) & (df['NYHA'] >= 2 )
class_2a2 = (df['SPECT_pre_LVEF'] <= 35)  & (df['ECG_pre_QRSd'] >= 150) & (df['LBBB'] == 0) & (df['NYHA'] >= 3 )

overall_class = []
for i in b: 
    if (class_1.iloc[i] == True) & (class_2a1.iloc[i] == False) & (class_2a2.iloc[i] == False):
        overall_class.append('class 1')
    elif (class_1.iloc[i] == False) & (class_2a1.iloc[i] == True) & (class_2a2.iloc[i] == False):
        overall_class.append('class 2A1')
    elif (class_1.iloc[i] == False) & (class_2a1.iloc[i] == False) & (class_2a2.iloc[i] == True):
        overall_class.append('class 2A2')
    else:
        overall_class.append('Other')

for i in range(20):
    sample = i
    preds = model.predict(x_test_data)
    preds = preds[sample]
    if preds > 0:
        status = 'response' 
    else:
        status = 'non-response'

    print("Predicted:", status, "with logit,", preds, ",and true response:",response.iloc[b[sample]], '. with CRT class:', overall_class[sample])

    
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap([x_test_data[0][sample], x_test_data[1][sample]], model, last_conv_layer_name)
    plt.matshow(heatmap)
    plt.show()

    save_and_display_gradcam(bruh[sample], heatmap)