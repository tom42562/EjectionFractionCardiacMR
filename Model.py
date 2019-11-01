from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D
from keras.layers import Dense, Dropout, Input


def get_model(n_ch=32):
    input = Input(shape=(96, 160, 160, 1))
    C1_0 = Conv3D(n_ch, (3, 3, 3), padding="valid", activation='relu')(input)
    C1_1 = Conv3D(n_ch, (3, 3, 3), padding="valid", activation='relu')(C1_0)
    MP1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(C1_1)
    D1 = Dropout(0.25)(MP1)

    C2_0 = Conv3D(2*n_ch, (3, 3, 3), padding="valid", activation='relu')(D1)
    C2_1 = Conv3D(2*n_ch, (3, 3, 3), padding="valid", activation='relu')(C2_0)
    MP2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(C2_1)
    D2 = Dropout(0.25)(MP2)

    C3_0 = Conv3D(4 * n_ch, (3, 3, 3), padding="valid", activation='relu')(D2)
    C3_1 = Conv3D(4 * n_ch, (3, 3, 3), padding="valid", activation='relu')(C3_0)
    MP3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(C3_1)
    D3 = Dropout(0.25)(MP3)

    C4_0 = Conv3D(8 * n_ch, (3, 3, 3), padding="valid", activation='relu')(D3)
    C4_1 = Conv3D(8 * n_ch, (3, 3, 3), padding="valid", activation='relu')(C4_0)
    MP4 = GlobalMaxPooling3D()(C4_1)
    D4 = Dropout(0.25)(MP4)

    Den2 = Dense(600, activation='softmax')(D4)
    model = Model(inputs=input, outputs=Den2)
    return model



