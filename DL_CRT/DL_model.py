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
directory = '/root/DL_CRT_project/CNN_data/'

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

def rfe_fs(clinical, label, num_features, automatic, fold):
    if automatic:
        model = Perceptron(random_state=0)
        rfecv = RFECV(
            estimator=model,
            min_features_to_select=10,
            step=1,
            cv=7)
        fit = rfecv.fit(clinical, np.squeeze(label))

    return np.array(clinical[:,fit.support_.tolist()]), fit.support_.tolist()

        
def rotate_images(images, classes, angle_range):
    reset_seeds(123)
  # Step 1: Find the counts of each class and the corresponding indices in the image array
    class_counts = np.unique(classes, return_counts=True)
    class_indices = {cls: np.where(classes == cls)[0] for cls in class_counts[0]}

    # Step 2: Randomly rotate the images in the image array for each class
    rotated_images = []
    class_status = []
    for cls, indices in class_indices.items():
        for index in indices:
            # Generate a random angle within the given range
            angle = np.random.uniform(angle_range[0], angle_range[1])

            # Rotate the image using the scipy.ndimage.rotate function
            rotated_image = rotate(images[index], angle, reshape = False)

            # Add the rotated image to the list of rotated images
            rotated_images.append(rotated_image)
            class_status.append(cls)
    
    
    # Step 3: Artificially balance the number of images in the smaller class
    # by performing additional rotations on some random images within the class
    if class_counts[1][0] < class_counts[1][1]:
        # Get the indices of the smaller class
        smaller_class_indices = class_indices[class_counts[0][0]]

        # Calculate the difference in size between the two classes
        size_difference = class_counts[1][1] - class_counts[1][0]

        # Randomly select some of the images from the smaller class
        # and perform additional rotations on them
        for i in range(2*size_difference):
            index = np.random.choice(smaller_class_indices)
            angle = np.random.uniform(angle_range[0], angle_range[1])
            rotated_image = rotate(images[index], angle, reshape = False)
            rotated_images.append(rotated_image)
            class_status.append(0)
            
    if class_counts[1][0] > class_counts[1][1]:
        # Get the indices of the smaller class
        smaller_class_indices = class_indices[class_counts[1][1]]

        # Calculate the difference in size between the two classes
        size_difference = class_counts[1][0] - class_counts[1][1]

        # Randomly select some of the images from the smaller class
        # and perform additional rotations on them
        for i in range(2*size_difference):
            index = np.random.choice(smaller_class_indices)
            angle = np.random.uniform(angle_range[0], angle_range[1])
            rotated_image = rotate(images[index], angle, reshape = False)
            rotated_images.append(rotated_image)
            class_status.append(1)
            
    # rotated_images_array= np.zeros(shape = (len(rotated_images), 64, 64,3))
    # for i in range(len(rotated_images)):
    #     rotated_images_array[i,:,:,:] = rotated_images[i]
    
    rotated_images = np.array(rotated_images)
  # Step 4: Concatenate the original images with the rotated images
  # and return a numpy array of the combined images and a corresponding array of class labels
    all_images = np.concatenate([images, rotated_images], axis = 0)
    all_classes = np.concatenate([classes, class_status])
    return all_images, all_classes

def tune_dl(hp):
    reset_seeds(123)
    
    # Initializer
    initializer = tf.keras.initializers.HeNormal(seed = 123)
    #initializer =  tf.keras.initializers.HeUniform(seed = 123)
    opt = hp.Choice('optimizer', ['adam', 'sgd'])
    if opt == 'adam':
        with hp.conditional_scope('optimizer', 'adam'):
            lr_decayed_fn  = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = hp.Float('learning_rate',
                            min_value=.000001,
                            max_value=.001,
                            sampling='LOG',
                            default= .00001),  
                            decay_steps = 100)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_decayed_fn,
                            amsgrad = True)
    elif opt == 'sgd':
        with hp.conditional_scope('optimizer', 'sgd'):
            lr_decayed_fn  = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = hp.Float('learning_rate',
                            min_value=.000001,
                            max_value=.001,
                            sampling='LOG',
                            default= .00001),  
                            decay_steps = 50)
            
            optimizer = tf.keras.optimizers.SGD(
            learning_rate = lr_decayed_fn,
            momentum=hp.Choice('momentum', [0.0, 0.05, 0.1, 1.0]))


    hp_activ = hp.Choice('activation', values = ['elu', 'leakyRelu'])

    with hp.conditional_scope('activation', ['leakyRelu']):
        hp_activ = tf.keras.layers.LeakyReLU(alpha = 0.01)
        
        
    y1 = Input(shape=(var_len), name = 'csv_features')
    merge = y1

    ####################################################################################
    # Conditional units
    num_csv_layers = hp.Int('n_csv_layers', 1 , 3)
    units = []
    with hp.conditional_scope('n_csv_layers', list(range(1, 3 + 1))):
        first_lay_units = hp.Choice(f"csv_neuron {1} units",a[num_csv_layers-1:])
        units.append(first_lay_units)
    for i in range(num_csv_layers-1):
        with hp.conditional_scope('n_csv_layers', list(range(i + 2, 3 + 1))):
            b = a.index(hp.get(f"csv_neuron {i+1} units"))
            units.append(hp.Choice(f"csv_neuron {i + 2} units",a[0:b+1]))

    for i in range(num_csv_layers):
        with hp.conditional_scope('n_csv_layers', list(range(i + 1, 3 + 1))):

            merge = Dense(units[i], activation = hp_activ,
                       kernel_initializer=initializer,
                         kernel_regularizer = tf.keras.regularizers.l2(hp.Float(f"csv {i +1} decay", min_value = 0.00001, max_value = .001, sampling = 'LOG', default = 0.001)))(merge)
            merge = Dropout(hp.Float(
                    f"csv_dropout {i + 1}",
                    min_value=0.0,
                    max_value=0.5,
                    default=0.05,
                    step=0.1), seed = 123)(merge)

    
    output = Dense(1, activation = 'sigmoid')(merge)

    #model

    model = Model(inputs = [y1],
              outputs = output)

    model.compile(optimizer = optimizer,
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                         tf.keras.metrics.AUC(name = 'auc'),
                        tfa.metrics.CohenKappa(num_classes = num_class, name = 'kappa'),
                        sensitivity,
                        specificity])
    return model



class dl_CVTuner(keras_tuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, **kwargs):
        reset_seeds(123)
        
        val_objective = []
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 10, 30, step = 3, default = 15)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 30, 50, step = 5, default = 40)
        
        #for train_indices, test_indices in StratifiedKFold(n_splits=8, shuffle = True, random_state = 123).split(x[0], y):
        for train_indices, test_indices in StratifiedShuffleSplit(n_splits=6, test_size = 1/10,  random_state = 123).split(x, y):


            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x[train_indices], y[train_indices], batch_size= kwargs['batch_size'], epochs=kwargs['epochs'], verbose = 0,
                    shuffle = True,
                    #validation_split = .055,
                    class_weight = class_weight
                     # ,callbacks =[earlystopping]
                     )
            
            val_objective.append(model.evaluate(x[test_indices], y[test_indices]))

        # AUC 
        self.oracle.update_trial(trial.trial_id, {'val_auc': np.mean(val_objective, axis = 0)[2]})


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


def tune_fine_tune(hp):
    if pre_tuned == 'vgg16':
        fine_tune_at =  hp.Int('tune_at_layers', 5, 15, step = 1, default = 13)
    elif pre_tuned == 'resnet':
        fine_tune_at =  hp.Int('tune_at_layers', 30, 47, step = 5, default =  40)

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    fine_tune_lr = pre_tuner_lr * hp.Float('learning_rate_multiplier',
                            min_value=.001, #.01
                            max_value=.1, #1
                            sampling='LOG',
                            default= .1)

    # aaa = model 
    # aaa.compile 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                     tf.keras.metrics.AUC(name = 'auc'),
                    tfa.metrics.CohenKappa(num_classes = num_class, name = 'kappa'),
                    sensitivity,
                    specificity])  


    return model 

class finetune_CVTuner(keras_tuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, **kwargs):
        reset_seeds(123)
        
        val_objective = []                                   # 5 + pre_tuner_epochs   #20
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 2 + pre_tuner_epochs, 15 + pre_tuner_epochs, step = 3, default = 10) 
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', pre_tuner_bs - 5, pre_tuner_bs + 5, step = 1, default = pre_tuner_bs)
        inital_epochs = pre_tuner_epochs - 1
    
    
#         #for train_indices, test_indices in StratifiedKFold(n_splits=8, shuffle = True, random_state = 123).split(x[0], y):
        val_objective = []
        for train_indices, test_indices in StratifiedShuffleSplit(n_splits=4, test_size = 1/10,  random_state = 123).split(x[0], y):

            model = self.hypermodel.build(trial.hyperparameters)

            model.fit([x[0][train_indices], x[1][train_indices]], y[train_indices], epochs=kwargs['epochs'] , verbose = 0,
                      batch_size = kwargs['batch_size'],
                    shuffle = True,
                    class_weight = class_weight,
                     initial_epoch = inital_epochs)
            val_objective.append(model.evaluate([x[0][test_indices], x[1][test_indices]], y[test_indices]))

        # AUC 
        self.oracle.update_trial(trial.trial_id, {'val_auc': np.mean(val_objective, axis = 0)[2]})
        
def train_evaluate(x_train, y_train, x_test, y_test, train, test, fold_num):
    reset_seeds(123)
        
################################################################################################### FS

    # save original test features 
    x_test_original = x_test[1]

    # CAD and SEX evaluation 
    
    cad_mask = x_test_original[:,34] == 1  #CAD mask 
    male_mask = x_test_original[:,37] == 1  #gender mask 

    bluh = [x_test[0][cad_mask],x_test[1][cad_mask] ]
    bluh = y_test[cad_mask]


    #Standardize 
    scaler = preprocessing.StandardScaler().fit(x_train[1])
    xt = scaler.transform(x_train[1])
   

    if fs_method == 'anovaf':
        # ANOVA F-statistic FS
        fs = SelectKBest(score_func = f_classif, k = 'all')
        fs.fit(x_train[1], y_train)

        high_fstats = fs.scores_ >= fs_level

        x_train[1] = x_train[1][:,high_fstats]
        x_test[1] = x_test[1][:,high_fstats]
        
    elif fs_method == 'bortua':
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, alpha = 0.1, random_state=1)

        # find all relevant features - 5 features should be selected
        feat_selector.fit(x_train[1], y_train)

        # call transform() on X to filter it down to selected features
        x_train[1] = feat_selector.transform(x_train[1])
        x_test[1] = feat_selector.transform(x_test[1])
        
    elif fs_method == 'info':
        fs = SelectKBest(score_func = mutual_info_classif, k = 'all')
        fs.fit(x_train[1], y_train)

        high_fstats = fs.scores_ >= 0.001

        x_train[1] = x_train[1][:,high_fstats]
        x_test[1] = x_test[1][:,high_fstats]
        
    elif fs_method == 'automatic':
        fs = FeatureSelector(objective = 'classification', auto = True, 
                            allowed_score_gap = 0.05)
        # Convert Pandas Dataframe 
        x_train[1] = pd.DataFrame(x_train[1])
        x_test[1] = pd.DataFrame(x_test[1])
        y_train = pd.Series(y_train)
             
        x_train[1] = fs.fit_transform(x_train[1], y_train).to_numpy()
        x_test[1] = fs.transform(x_test[1]).to_numpy()
        y_train = y_train.to_numpy()
    elif fs_method == 'rfe':
        xt, fs_column = rfe_fs(xt, y_train, 80, True, fold_num)
        x_test[1] = x_test[1][:,fs_column]
        x_train[1] = x_train[1][:,fs_column]
################################################################################################### Preprocess

    if smote_bol:
        oversample = SMOTE(random_state = 123)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
           
    global class_weight
    class_weight = weights(y_train)
    

    # Cor 
    corr_col = find_correlation(x_train[1], .8)
    x_train[1] = np.delete(x_train[1], corr_col, axis = 1)
    nzv = VarianceThreshold(0.01).fit(x_train[1])
    x_train[1] = nzv.transform(x_train[1])
    
    x_test[1] = np.delete(x_test[1], corr_col, axis = 1)
    x_test[1] = nzv.transform(x_test[1])
    
    #Standardize 
    scaler = preprocessing.StandardScaler().fit(x_train[1])
    x_train[1] = scaler.transform(x_train[1])
    x_test[1] = scaler.transform(x_test[1])


# BUFFER ZEROS ON IMAGE 
    perf_temp = np.expand_dims(x_test[0][:,:,:,0], axis = -1)
    syst_temp = np.expand_dims(x_test[0][:,:,:,1], axis = -1)
    wall_temp = np.expand_dims(x_test[0][:,:,:,2], axis = -1)

    perf50_temp = np.where(perf_temp < 0.5, 0, perf_temp)
    #perf50_temp = perf_temp

    img = np.concatenate((perf_temp,syst_temp), axis = 1)
    img2 = np.concatenate((wall_temp, perf50_temp), axis = 1)
    x_test[0] = np.concatenate((img, img2), axis = 2)
    x_test[0] = np.tile(x_test[0], 3)
    x_test[0] = x_test[0] * 255
# train
    # add images AND CONVERT TO RGB
    perf_temp = np.expand_dims(x_train[0][:,:,:,0], axis = -1)
    syst_temp = np.expand_dims(x_train[0][:,:,:,1], axis = -1)
    wall_temp = np.expand_dims(x_train[0][:,:,:,2], axis = -1)

    perf50_temp = np.where(perf_temp < 0.5, 0, perf_temp)
    #perf50_temp = perf_temp


    img = np.concatenate((perf_temp,syst_temp), axis = 1)
    img2 = np.concatenate((wall_temp, perf50_temp), axis = 1)
    x_train[0] = np.concatenate((img, img2), axis = 2)
    x_train[0] = np.tile(x_train[0], 3)
    x_train[0] = x_train[0] * 255
    print('train max',np.max(x_train[0]))
    print('train min', np.min(x_train[0]))



################################################################################################### models

    
    global base_model
    if pre_tuned == 'xception':       
        base_model = tf.keras.applications.MobileNetV2(input_shape=(192,64,3),
                                               include_top=False,
                                               weights='imagenet',
                                                pooling = 'avg')
        x = tf.keras.applications.Xception.preprocess_input()     
        x = tf.keras.applications.Xception.preprocess_input()     

    elif pre_tuned == 'vgg16':

        x_train[0] = tf.keras.applications.vgg16.preprocess_input(x_train[0])     
        x_test[0] = tf.keras.applications.vgg16.preprocess_input(x_test[0])    



    elif pre_tuned == 'resnet':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(128,128,3),
                                               include_top=False,
                                               weights='imagenet',
                                                pooling = 'avg')
        x_train[0] = tf.keras.applications.resnet.preprocess_input(x_train[0])     
        x_test[0] = tf.keras.applications.resnet.preprocess_input(x_test[0])    
        rot_imgs = tf.keras.applications.resnet.preprocess_input(rot_imgs)  
        base_model.trainable = False
        
        
    elif pre_tuned == 'mobil2':
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(128,128,3),
                                               include_top=False,
                                               weights='imagenet',
                                                pooling = 'avg',
                                                alpha = 0.5)
        x_train[0] = tf.keras.applications.mobilenet_v2.preprocess_input(x_train[0])     
        x_test[0] = tf.keras.applications.mobilenet_v2.preprocess_input(x_test[0])    
        rot_imgs = tf.keras.applications.mobilenet_v2.preprocess_input(rot_imgs)  
        base_model.trainable = False

    global var_len
    var_len = x_train[1].shape[1]
    print('train max after vgg',np.max(x_train[0]))
    print(' train min after vgg', np.min(x_train[0]))
################################################################################################### Fit pretune
    
    tuner = pretune_CVTuner(
    hypermodel = tune_pre,
    oracle=keras_tuner.oracles.BayesianOptimization(
    objective= keras_tuner.Objective('val_auc', direction = 'max'),
    max_trials=70, #30
    tune_new_entries = True,
    allow_new_entries = True,
    seed = 123),
    overwrite=True,
    directory = "CNN CRT",
    project_name = "keras_tuner_pre")
    
    
    tuner.search(x_train, y_train)
    global model
    model = tune_pre(tuner.get_best_hyperparameters(1)[0])
    
    # Final epoch model
    model.fit(x_train, y_train, 
        epochs = tuner.get_best_hyperparameters(num_trials=1)[0].get('epochs'),
        batch_size = tuner.get_best_hyperparameters(num_trials=1)[0].get('batch_size'),
        verbose = 0,
        class_weight = class_weight, 
        shuffle = True)
    
    model.save('DL_final_model' + str(fold_num) +'.h5')
    
    
    
    pre_tune_scores = model.evaluate(x_test, y_test)
    pre_tune_parms = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    viz = RocCurveDisplay.from_predictions(
        y_true = y_test,
        y_pred = model.predict(x_test),
        name=f"ROC fold {fold_num + 1}",
        alpha=0.3,
        lw=1,
        ax=ax)
    

    
    return [pre_tune_scores, pre_tune_parms, viz]


from sklearn.metrics import RocCurveDisplay

reset_seeds(123)
num_class = 2
def score(dataframe, img,  smote_bole, fs_lev, fs_meth, pre_tuned_model):
    global smote_bol
    smote_bol = smote_bole
   
    global pre_tuned
    pre_tuned = pre_tuned_model
   
    rows = len(df)
   
    param_grid = {"bin_threshold_perf": [ 0.6, 0.65]
                                         ,
                  "bin_threshold_syst": [0.4, 0.45],
                  "bin_threshold_thk": [  0.2 ],
                  "filter": ['rad',
                             #'vitrip',
                             #'Gray',
                             #'TGray',
                             'bin','den'
                            ]}

    metric_list = [
        {"metric": "bottleneck", "metric_params": {}},
        {"metric": "wasserstein", "metric_params": {"p": 1}},
        {"metric": "wasserstein", "metric_params": {"p": 2}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
    ]
 
   
    param_combination = []
    a = 0
   
    global fs_level
    fs_level = fs_lev
   
    global fs_method
    fs_method = fs_meth
   

    clinical_data = dataframe.drop(columns = ['resp', 'Race', 'NYHA', 'Hospitalization'])
    #X =  np.concatenate([clinical_data, data[:,1:]], axis = 1)
    X =  np.array(clinical_data)
    # foprmat data
    Y = np.where(np.array(response)  == 'yes', 1, 0)    
    #Y = np.array(super_response)
   
 
 
    X = [conc, X]
   
    # evaluate ML
    combo_output = []
    ensemble_output = []
    pre_tune_output = []
    fine_tune_output = []
   
    scores = np.zeros((5,6))

    
    final_model_parameter_list = []
    ensemble_scores = np.zeros((5,6))
    pre_tune_score = np.zeros((5,6))
    pre_tune_parm = []
    fine_tune_score = np.zeros((5,6))
    fine_tune_parm = []
    roc_curves = []
    idx = 0
    # kFold = StratifiedKFold(n_splits=9, shuffle = True, random_state = 123)
    # for train, test in kFold.split(X[0], Y):
    
    #[pre_tune_scores, pre_tune_parms, viz, cad_scores, non_cad_scores, male_scores, female_scores]
   
    # 6
    for train, test in StratifiedShuffleSplit(n_splits=5 , test_size = 1/11,  random_state = 123).split(X[1], Y):
        # if idx == 0 or idx == 1 or idx == 2:
        #     idx += 1
        #     continue
        print('Fold', idx)
        scores[idx], best_params, roc_curve = train_evaluate([X[0][train], X[1][train]], Y[train],  [X[0][test], X[1][test]], Y[test], train, test, idx)
        final_model_parameter_list.append(best_params)
        roc_curves.append(roc_curve)
        idx += 1
 
    combo_output.append(scores)

    
    return combo_output, final_model_parameter_list, roc_curves
 


overall_scores, final_model_parameter_list, list_of_roc_curve = score(df, conc, smote_bole = False,  fs_lev = 2, fs_meth = 'rfe', pre_tuned_model = 'vgg16')
means = []
sd = []


