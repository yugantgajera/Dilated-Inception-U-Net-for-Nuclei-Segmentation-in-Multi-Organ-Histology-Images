from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_pred_f * y_true_f)
    return intersection/(K.sum(y_pred_f) + K.sum(y_true_f) - intersection)

def aji(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_pred_f * y_true_f)
    return intersection/(2 * K.sum(y_pred_f) + 2 * K.sum(y_true_f) - 3 * intersection)

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def di_block(inputs, fn, batch_norm = True, upsampling = False, downsampling = False):
    

    f1 = Conv2D(fn, (1, 1), activation = 'relu')(inputs)
    f1 = Conv2D(fn, (3, 3), activation = 'relu', dilation_rate = (1, 1), padding = 'same', kernel_initializer = 'he_normal')(f1)

    f2 = Conv2D(fn, (1, 1), activation = 'relu')(inputs)
    f2 = Conv2D(fn, (3, 3), activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(f2)

    f3 = Conv2D(fn, (1, 1), activation = 'relu')(inputs)
    f3 = Conv2D(fn, (3, 3), activation = 'relu', dilation_rate = (3, 3), padding = 'same', kernel_initializer = 'he_normal')(f3)

    filter_concat = concatenate([f1, f2, f3], axis = -1)

    if batch_norm:
        output = BatchNormalization()(filter_concat)
    else:
        output = filter_concat

    if downsampling:
        skip_out = output

    if upsampling or downsampling:
        if upsampling:
            output = Conv2DTranspose(fn, (2, 2), strides = (2, 2), activation = 'relu', padding = 'same')(output)
        elif downsampling:
            output = MaxPooling2D((2, 2), 2)(output)
            
    if downsampling:
        return output, skip_out
    else:
        return output

def diu_net_model(shape, compile = True):
    inputs = Input(shape)

    di_block_1, skip_1 = di_block(inputs, 32, batch_norm = True, upsampling = False, downsampling = True)
    di_block_2, skip_2 = di_block(di_block_1, 64, batch_norm = True, upsampling = False, downsampling = True)
    di_block_3, skip_3 = di_block(di_block_2, 128, batch_norm = True, upsampling = False, downsampling = True)
    di_block_4, skip_4 = di_block(di_block_3, 192, batch_norm = True, upsampling = False, downsampling = True)

    di_block_5 = di_block(di_block_4, 256, batch_norm = True, upsampling = False, downsampling = False)
    di_block_6 = di_block(di_block_5, 256, batch_norm = True, upsampling = True, downsampling = False)
    
    di_block_7 = di_block(concatenate([di_block_6, skip_4], axis = -1), 192, batch_norm = True, upsampling = True, downsampling = False)
    di_block_8 = di_block(concatenate([di_block_7, skip_3], axis = -1), 128, batch_norm = True, upsampling = True, downsampling = False)
    di_block_9 = di_block(concatenate([di_block_8, skip_2], axis = -1), 64, batch_norm = True, upsampling = True, downsampling = False)
    di_block_10 = di_block(concatenate([di_block_9, skip_1], axis = -1), 32, batch_norm = True, upsampling = False, downsampling = False)

    output = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'valid')(di_block_10)

    diunet = Model(inputs, output)

    if compile:
        diunet.compile(optimizer = Adam(), loss = [dice_coef_loss], metrics = [dice_coef, iou, aji])

    return diunet