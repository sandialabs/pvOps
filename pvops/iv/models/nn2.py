'''
Prior to running this, one must optimize a electrical parameter fit on the 5 input parameters to the diode model
'''

import sys
sys.path.append('C:/Users/mwhopwo/Desktop/IVTC/faultsims_and_ivtc')
import classification_assets as assets
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import copy
import numpy as np
import matplotlib.pyplot as plt
import datetime

#labeled = list(sim.string_cond.keys())
labeled = set(bigdf['mode'].tolist())
#labeled = ['Unstressed', 'Partial Soiling (1M)', 'Partial Soiling (6M)', 'Partial Soiling (10M)']
print(labeled)

print(bigdf)
# assets.plot_allmodes_iv_fillplot(bigdf, labeled)

# bigdf_s = bigdf.sample(n=200)
# import matplotlib.pyplot as plt
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'aquamarine', 'blueviolet', 'fuchsia', 'orchid', 'crimson']
# for idx,lbl in enumerate(labeled):
#     df_iter = bigdf_s[bigdf_s['mode'] == lbl]
#     for ind,row in df_iter.iterrows():
#         plt.plot(row['voltage'],row['current'],colors[idx])
#     plt.plot([],[],colors[idx],label=lbl)
# #plt.title(lbl)
# plt.xlabel('V (Volts) a')
# plt.ylabel('I (Amps)')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

bigdf = assets.balance_df(bigdf, 'truncate')


# assets.plot_allmodes_iv_fillplot(bigdf, labeled)

# bigdf_s = bigdf.sample(n=200)
# import matplotlib.pyplot as plt
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'aquamarine', 'blueviolet', 'fuchsia', 'orchid', 'crimson']
# for idx,lbl in enumerate(labeled):
#     df_iter = bigdf_s[bigdf_s['mode'] == lbl]
#     for ind,row in df_iter.iterrows():
#         plt.plot(row['voltage'],row['current'],colors[idx])
#     plt.plot([],[],colors[idx],label=lbl)
# #plt.title(lbl)
# plt.xlabel('V (Volts) b')
# plt.ylabel('I (Amps)')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

if nn_type is '1DCNN':
    RESAMPLE_RESOLUTION = 0.009
elif nn_type is 'LSTM_multihead':
     RESAMPLE_RESOLUTION = 0.01

bigdf['E'] = [1000] * len(bigdf.index)
bigdf['T'] = [25] * len(bigdf.index)
bigdf = assets.feature_generation(bigdf, RESAMPLE_RESOLUTION, correct_gt = sim_corrGT, CECmodule_parameters = module_parameters, n_mods = 12)
bigdf = bigdf.sample(frac=1).reset_index(drop=True)
bigdf.dropna(inplace=True)
print(bigdf['mode'].value_counts())

if show_XY_plots:
    assets.plot_allmodes_iv_fillplot(bigdf, labeled)

# bigdf_s = bigdf.sample(n=200)
# import matplotlib.pyplot as plt
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'aquamarine', 'blueviolet', 'fuchsia', 'orchid', 'crimson']
# for idx,lbl in enumerate(labeled):
#     df_iter = bigdf_s[bigdf_s['mode'] == lbl]
#     for ind,row in df_iter.iterrows():
#         plt.plot(row['voltage'],row['current'],colors[idx])
#     plt.plot([],[],colors[idx],label=lbl)
# #plt.title(lbl)
# plt.xlabel('V (Volts)')
# plt.ylabel('I (Amps)')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

''' IMPORT MEASURED DATA (AKA live data if model deployed) '''
meas_df = pd.read_pickle(folder+'FaultedData_FSEC.pkl')
meas_df = meas_df[meas_df['mode'].isin([1,2,3,4,5])]#,3,4])]
meas_df['date'] = meas_df.index
meas_df[meas_df['mode'] == 3] =  meas_df[(meas_df['mode'] == 3) & (meas_df['date'] < datetime.datetime.strptime('6/01/2019','%m/%d/%Y'))]
del meas_df['date']

labeled_meas = ['Partial Soiling (1M)', 'Partial Soiling (6M)','Cell cracking (4M)','Interconnection (4M)','Resistive load']
meas_df['mode'] = np.array(meas_df['mode'].tolist()) - 1

print(meas_df['mode'].value_counts())

meas_df.dropna(inplace=True)
meas_df['mode'] = [labeled_meas[int(lbl)] for lbl in meas_df['mode'].tolist()]

meas_df = meas_df[meas_df['E'] > irrFilter]

plt.hist(meas_df['E'].tolist())
plt.show()
plt.hist(meas_df['T'].tolist())
plt.show()
#yte_enc = encoder.transform(meas_df['mode'].tolist())

#assets.plot_allmodes_iv_fillplot(meas_df, labeled_meas)

# colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'aquamarine', 'blueviolet', 'fuchsia', 'orchid', 'crimson']
# for idx,lbl in enumerate(['Partial Soiling (6M)', 'Partial Soiling (1M)']):
#     df_iter = meas_df[meas_df['mode'] == lbl]
#     for ind,row in df_iter.iterrows():
#         plt.plot(row['voltage'],row['current'],colors[idx])
#     plt.plot([],[],colors[idx],label=lbl)
# plt.title(lbl)
# plt.xlabel('V (Volts)')
# plt.ylabel('I (Amps)')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

meas_df = assets.feature_generation(meas_df, RESAMPLE_RESOLUTION, correct_gt = meas_corrGT, CECmodule_parameters = module_parameters, n_mods = 12)
meas_df = meas_df.sample(frac=1).reset_index(drop=True)
meas_df.dropna(inplace=True)
print(meas_df)
print(meas_df['mode'].value_counts())

# IVproc = IVProcessor(CR_curve, 'S2current', 'S2voltage', 'POA', 'Temp', 'S2WindSpeed_Max_3s')
# self_Tcs, self_Irrs, meas_Iscs, sim_Iscs, meas_Vocs, sim_Vocs = IVproc.train_met_fit(sim_fsec_reference)
# Efit, Tcfit = IVproc.met_fit(Vsimcutoff, Isimcutoff, POA, Tcell, meas_Iscs[curvenum], meas_Vocs[curvenum])

if show_XY_plots:
    assets.plot_allmodes_iv_fillplot(meas_df, labeled_meas)

# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'aquamarine', 'blueviolet', 'fuchsia', 'orchid', 'crimson']
# for idx,lbl in enumerate(labeled_meas):
#     df_iter = meas_df[meas_df['mode'] == lbl]
#     for ind,row in df_iter.iterrows():
#         plt.plot(row['voltage'],row['current'],colors[idx])
#     plt.plot([],[],colors[idx],label=lbl)
# plt.title('GT Corrected')
# plt.xlabel('V (Volts)')
# plt.ylabel('I (Amps)')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

#sys.exit()

outfolder = 'C://Users//mwhopwo//Desktop//IVTC//faultsims_and_ivtc//IVTC_w_sims//results//'

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'
import keras
keras.backend.clear_session()
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten, Permute, Activation, RepeatVector, Add
from keras.layers import InputLayer, Input
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, dot, concatenate
from keras.layers import Bidirectional
#from keract import get_activations
from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

params = ['current', 'power', 'differential']
max_epochs = 50

y = np.array(bigdf['mode'].tolist())

if nn_type is 'LSTM_multihead':
    c_X, p_X, d_X = np.array(bigdf['current'].tolist()), np.array(bigdf['power'].tolist()), np.array(bigdf['differential'].tolist())

    def list_to_lol(lst,ln):
        # lst: list input
        # len: num vals in each sublist
        maxlen = len(lst)
        i = 0
        l = []
        while i+ln < maxlen:
            x = lst[i:i+ln]
            l.append(x)
            i += ln
        #print(l)
        return l

    n_filters = 10.
    print(f'Making {n_filters} and sample {len(c_X[0])}')
    length = int(len(c_X[0]) / n_filters) # lists
    c_X_restructure = []
    p_X_restructure = []
    d_X_restructure = []
    for ividx in range(len(c_X)):
        c_X_restructure.append(list_to_lol(c_X[ividx], length))
        p_X_restructure.append(list_to_lol(p_X[ividx], length))
        d_X_restructure.append(list_to_lol(d_X[ividx], length))

    c_X_restructure = np.asarray(c_X_restructure)
    p_X_restructure = np.asarray(p_X_restructure)
    d_X_restructure = np.asarray(d_X_restructure)

if nn_type is '1DCNN':

    # Below is really all variables, but defining c_X to allow input into rest of code easier
    c_X = assets.convert_ivdata_to_cnn_structure(bigdf, params)

    num_samples_in_IV = len(c_X[0])
    num_params = 3

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
_to_train_encoder = encoder.fit_transform(y)

encoded_length = len(_to_train_encoder[0])

cv = StratifiedKFold(n_splits = 5)
cvscores = []
for train_idx, test_idx in cv.split(c_X, y):
    ytr, yte = y[train_idx], y[test_idx]
    ytr = encoder.transform(ytr)
    yte = encoder.transform(yte)
    encoded_length = len(ytr[0])

    if nn_type is 'LSTM_multihead':
        xtr = [c_X_restructure[train_idx],
                p_X_restructure[train_idx],
                d_X_restructure[train_idx]]

        n_sequences = np.asarray(xtr).shape[2]
        n_samples_in_sequence = np.asarray(xtr).shape[3]

        xte = [c_X_restructure[test_idx],
                p_X_restructure[test_idx],
                d_X_restructure[test_idx]]
    
    elif nn_type is '1DCNN':
        xtr, xte = c_X[train_idx], c_X[test_idx]

    print('xtr shape: ', np.array(xtr).shape)
    print('xte shape: ', np.array(xte).shape)
    print('ytr shape: ', np.array(ytr).shape)
    print('yte shape: ', np.array(yte).shape)

    sys.exit()

    if nn_type is 'LSTM_multihead':
        units = 500

        inputs_A = Input(shape=(n_sequences,n_samples_in_sequence), name = 'aa')
        inputs_B = Input(shape=(n_sequences,n_samples_in_sequence), name = 'ab')
        inputs_C = Input(shape=(n_sequences,n_samples_in_sequence), name = 'ac')
        #inputs_D = Input(shape=(n_sequences,n_samples_in_sequence), name = 'ad')

        activations_A = LSTM(units, return_sequences = False, name = 'ae')(inputs_A)
        activations_B = LSTM(units, return_sequences = False, name = 'af')(inputs_B)
        activations_C = LSTM(units, return_sequences = False, name = 'ag')(inputs_C)
        #activations_D = LSTM(units, return_sequences = False, name = 'ah')(inputs_D)

        activations_A = Dropout(0.5)(activations_A)
        activations_B = Dropout(0.5)(activations_B)
        activations_C = Dropout(0.5)(activations_C)
        #activations_D = Dropout(0.5)(activations_D)

        # LSTM_A = LSTM(units, return_sequences = False)(activations_A)
        # LSTM_B = LSTM(units, return_sequences = False)(activations_B)
        # LSTM_C = LSTM(units, return_sequences = False)(activations_C)
        # LSTM_D = LSTM(units, return_sequences = False)(activations_D)

        # hidden_sizeA = int(activations_A.shape[2])
        # hidden_sizeB = int(activations_B.shape[2])
        # hidden_sizeC = int(activations_C.shape[2])
        # hidden_sizeD = int(activations_D.shape[2])
        # h_t_A = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_sizeA,), name='last_hidden_state_A')(activations_A)
        # h_t_B = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_sizeB,), name='last_hidden_state_B')(activations_B)
        # h_t_C = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_sizeC,), name='last_hidden_state_C')(activations_C)
        # h_t_D = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_sizeD,), name='last_hidden_state_D')(activations_D)

        # pre_mlp = concatenate([h_t_A, h_t_B, h_t_C, h_t_D], name='attention_output')

        #pre_mlp = concatenate([activations_A, activations_B, activations_C, activations_D], name='concatenate_1')
        pre_mlp = concatenate([activations_A, activations_B, activations_C], name='concatenate_1')

        #pre_mlp = concatenate([LSTM_A, LSTM_B, LSTM_C, LSTM_D], name='attention_output')

        #pre_mlp = Dropout(0.5)(pre_mlp)

        d = Dense(52, activation = 'relu', name = 'dense_1')(pre_mlp)
        d2 = Dense(16, activation='relu', name = 'dense_2')(d)
        activations = Dense(encoded_length, activation='softmax', name = 'dense_3')(d2)

        #model = Model(inputs = [inputs_A, inputs_B, inputs_C, inputs_D], outputs = [activations])
        model = Model(inputs = [inputs_A, inputs_B, inputs_C], outputs = [activations])

    if nn_type is '1DCNN':
        model = Sequential()
        #filters= 64 -> 256 because experiment369
        print(num_samples_in_IV,num_params)
        model.add(Conv1D(filters=256, kernel_size=32, activation='relu', input_shape=(num_samples_in_IV,num_params)))
        model.add(Conv1D(filters=256, kernel_size=32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(encoded_length, activation='softmax'))

    if train_model:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        import time
        st_time = time.time()
        batchsize = 8
        history = model.fit(xtr, ytr, epochs=max_epochs, batch_size=batchsize, verbose = 0)
        print('FIT TIME: ', time.time() - st_time)
        scores = model.evaluate(xte, yte, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #print('\tnum.train {} num.test {}'.format(len(xtr), len(xte)))
        cvscores.append(scores[1] * 100)
    
    else:
        model.load_weights(outfolder+f'{nn_type}_'+addendum+'.h5')
        #model.load_weights(outfolder+'model_multiLSTM_unstressed_soiling_goodresults.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        break

if train_model:
    # Save model
    model.save_weights(outfolder+f'{nn_type}_'+addendum+'.h5')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# # Save model to h5
# if nn_attention:
#     model.save_weights('model_1DCNN_BL_PS_Cr.h5')
# else:
#     model.save_weights('model_multiLSTM_BL_PS_Cr.h5')

#plot_model(model, show_shapes=False, show_layer_names=False, to_file='D://master-data//{}//model_architecture.png'.format(folder))

#y_te = np.array(meas_df['mode'].tolist())

if nn_type is 'LSTM_multihead':
    c_teX, p_teX, d_teX = np.array(meas_df['current'].tolist()), np.array(meas_df['power'].tolist()), np.array(meas_df['differential'].tolist())
    length = int(len(c_X[0]) / n_filters) # lists
    c_teX_restructure = []
    p_teX_restructure = []
    d_teX_restructure = []
    for ividx in range(len(c_teX)):
        c_teX_restructure.append(list_to_lol(c_teX[ividx], length))
        p_teX_restructure.append(list_to_lol(p_teX[ividx], length))
        d_teX_restructure.append(list_to_lol(d_teX[ividx], length))

    c_teX_restructure = np.asarray(c_teX_restructure)
    p_teX_restructure = np.asarray(p_teX_restructure)
    d_teX_restructure = np.asarray(d_teX_restructure)

    Xtest = [c_teX_restructure,
            p_teX_restructure,
            d_teX_restructure]

if nn_type is '1DCNN':
    Xtest = assets.convert_ivdata_to_cnn_structure(meas_df, params)

# plot model
#plot_model(model, show_shapes=False, show_layer_names=False, to_file=outfolder+'model_architecture.png')

if train_model:
    plt.close()
    # print(history.history.keys())
    plt.plot(history.history['categorical_accuracy'])
    #plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('D://master-data//{}//accuracy_v_epoch.png'.format(folder))
    #plt.close()
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

#_, accuracy = model.evaluate(X_test, y_test, batch_size=8, verbose=1)
yhat = model.predict(Xtest, batch_size=25, verbose=1)

num_correct = 0
num_total = len(yhat)
y_hat_decoded = []
yacts = np.array(meas_df['mode'].tolist())
for yguess,yact in zip(yhat,yacts):
    estimated_class = encoder.classes_[np.argmax(yguess)]
    #print(encoder.classes_)
    #print(f'yact:{yact}, estimated:{estimated_class}')
    #print(f'probs:{yguess}')
    #print()
    if estimated_class == yact:
        num_correct += 1
    y_hat_decoded.append(estimated_class)

print('Accuracy on test data: {:.2f}% ({}/{})'.format(num_correct/num_total, num_correct, num_total))

print(confusion_matrix(yacts,y_hat_decoded))
print(classification_report(yacts,y_hat_decoded))

print('classes: ')
print(encoder.classes_)
for Itest,Vtest,yguess,yact in zip(np.array(meas_df['current'].tolist()),np.array(meas_df['voltage'].tolist()),yhat,yacts):

    print('Probs: ')
    print('\t',yguess)
    plt.plot(Vtest,Itest)
    plt.xlabel('V / Vmax')
    plt.ylabel('I / Imax')
    plt.title('Actual: {}, Estimated: {}, confidence: {:.2f}'.format(yact,encoder.classes_[np.argmax(yguess)],round(yguess[np.argmax(yguess)],2)))
    plt.show()

