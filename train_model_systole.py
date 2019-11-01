from Model import get_model
from keras import callbacks
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from data_generator import data_generator

model_systole = get_model(n_ch=32)
adam = Adam(lr=0.0001)
model_systole.compile(optimizer=adam, loss=sparse_categorical_crossentropy)
model_systole.summary()

#  --- Define data and training ----#
path_train = 'Training_arrays/systole/'
path_val = 'Validation_arrays/systole'
pathModel = 'SystoleModel/'
pathLog = 'SystoleLog/'

batchsize = 4

n_epochs, n_iter_train, n_iter_val = 400, int(491 / batchsize), int(194 / batchsize)
data_train = data_generator(path_train, batchsize=batchsize, mode='Systole', shuffle=True, train=True)
data_val = data_generator(path_val, batchsize=batchsize, mode='Systole', shuffle=False, train=False)
modelvalsave = callbacks.ModelCheckpoint(pathModel + 'model-0-{epoch:03d}.hdf5', verbose=1, save_best_only=True,
                                      monitor='val_loss',
                                      mode='min', save_weights_only=False, period=1)
modeltrainsave = callbacks.ModelCheckpoint(pathModel + 'train-model-0-{epoch:03d}.hdf5', verbose=1, save_best_only=True,
                                      monitor='loss',
                                      mode='min', save_weights_only=False, period=1)
TfBoardCb = callbacks.TensorBoard(log_dir=pathLog, histogram_freq=0)
callbacks_list = [TfBoardCb, modelvalsave, modeltrainsave]
systole_model_history = model_systole.fit_generator(data_train, steps_per_epoch=n_iter_train, epochs=n_epochs, initial_epoch=0,
                            callbacks=callbacks_list, validation_data=data_val, validation_steps=n_iter_val,
                            use_multiprocessing=False)
