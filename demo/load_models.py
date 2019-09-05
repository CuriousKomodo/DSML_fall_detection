import numpy as np
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')
import Past_Autoencoder_model
from Past_Autoencoder_model import *
from Dropout_Autoencoder import *
import pickle

def load_encoder(VAE=True, weight_path=None,latent_dim = 128,hidden_dim= 512,input_dim=34):
    if not weight_path:
        weight_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/past_VAE_enc_model4_weights.h5'

    params = {'latent_dim': latent_dim, 'input_dim':input_dim,'hidden_dim':hidden_dim,'GPU':False}
    past_ae = past_LSTM_autoencoder(**params)
    enc_model = past_ae.enc_model
    enc_model.compile(optimizer='adam',loss=past_ae.vae_loss)
    enc_model.load_weights(weight_path)
    return enc_model


def load_encoder_new(VAE=True, weight_path=None,latent_dim = 128,hidden_dim= 512,input_dim=34):
    if not weight_path:
        weight_path = '/Users/kefei/Documents/results/win60/past/VAE/klanneal/with_classifier/'+'classifier_model.h5'
    params = {'latent_dim': latent_dim, 'input_dim':input_dim,'hidden_dim':hidden_dim,'GPU':False,'classifier':True}
    past_ae = past_LSTM_autoencoder(**params)
    classifier_model = past_ae.classifier_model
    classifier_model.load_weights(weight_path)
    classifier_mode.compile(optimizer='adam',loss=past_ae.vae_loss)
    return classifier_mode


def load_classifier(pca_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/pca.pkl',
                    svm_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/svm.pkl'):
    with open(pca_dir, 'rb') as file:
        pca = pickle.load(file)
    with open(svm_dir, 'rb') as file:
        svm = pickle.load(file)
    return pca, svm

def load_DAE(weight_path = None):
    if not weight_path:
        weight_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/DAE/weights2048.h5'
    model_params = {'hidden_dim': 2048, 'input_dim':34, 'dropout': 0.01}
    dae = DAE(**model_params)
    model = dae.model
    model.compile(optimizer='adam',loss=losses.mean_squared_error)
    model.load_weights(weight_path)
    return model
