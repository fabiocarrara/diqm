#
# Code for the model taken from:
# https://github.com/zizhaozhang/unet-tensorflow-keras
#

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, merge, Dense, Flatten, Conv2DTranspose, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate


class BaseNet():
            
    def create_model(self, img_shape, architecture='normal'):
        print('Building {}: {}'.format(self.__class__.__name__, architecture))
        
        references = Input(shape=img_shape)
        distorted = Input(shape=img_shape)
        inputs = concatenate([references, distorted])
        
        # call a _{arch}_architecture methods
        outputs = getattr(self, '_{}_architecture'.format(architecture))(inputs)

        model = Model(inputs=(references, distorted), outputs=outputs)

        return model
        

class QNet(BaseNet):
        
    @staticmethod
    def _normal_architecture(inputs):
        concat_axis = 3
    
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        
        avgpool = GlobalAveragePooling2D()(conv5)
        fc1 = Dense(256, activation='relu')(avgpool)
        q = Dense(1, activation='sigmoid', name='q')(fc1)
        
        return q

    def get_losses(self):
        return 'mse'


def _get_crop_shape(target, refer):
    # print target.get_shape()[2], refer.get_shape()[2]
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0), cw
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0), ch
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


class VDPNet(BaseNet):
    
    @staticmethod
    def _normal_architecture(inputs):
        concat_axis = 3
    
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = _get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = _get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = _get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = _get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = _get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        pmap = Conv2D(1, (1, 1), activation='sigmoid', name='pmap')(conv9)

        flat_conv5 = Flatten()(conv5)
        fc1 = Dense(256, activation='relu')(flat_conv5)
        q = Dense(1, activation='sigmoid', name='q')(fc1)
        
        return pmap, q
        
    @staticmethod    
    def _small_architecture(inputs):
        concat_axis = 3
    
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        #conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        #conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 512 -> 256

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        #conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        #conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)

        up_conv2 = UpSampling2D(size=(2, 2))(conv2)
        ch, cw = _get_crop_shape(conv1, up_conv2)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up3 = concatenate([up_conv2, crop_conv1], axis=concat_axis)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
        #conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        #conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

        ch, cw = _get_crop_shape(inputs, conv3)
        conv3 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv3)
        pmap = Conv2D(1, (1, 1), activation='sigmoid', name='pmap')(conv3)

        q_conv1 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv1')(pmap)
        q_conv2 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv2')(q_conv1)
        q_conv3 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv3')(q_conv2)
        q_conv4 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv4')(q_conv3)
        q_fmap = Conv2D(1, (1, 1), activation='relu', padding='same', name='q_fmap')(pmap)
        q_flat = Flatten(name='q_flat')(q_fmap)
        q = Dense(1, activation='sigmoid', name='q')(q_flat)
        
        return pmap, q
    
    @staticmethod
    def _fixed_res_architecture(inputs):
        concat_axis = 3
        
        def multisize_conv(in_layer, num, prefix):
            n = num / 4
            conv1x1 = Conv2D(n, (1, 1), activation='relu', padding='same', name=prefix + '_1x1')(in_layer)
            conv3x3 = Conv2D(n, (3, 3), activation='relu', padding='same', name=prefix + '_3x3')(in_layer)
            conv5x5 = Conv2D(n, (5, 5), activation='relu', padding='same', name=prefix + '_5x5')(in_layer)
            conv7x7 = Conv2D(n, (7, 7), activation='relu', padding='same', name=prefix + '_7x7')(in_layer)
            out_layer = concatenate([conv1x1, conv3x3, conv5x5, conv7x7], axis=concat_axis)
            return out_layer
            
        multi1 = multisize_conv(inputs, 64, 'multi1')
        multi2 = multisize_conv(multi1, 128, 'multi2')
        multi3 = multisize_conv(multi2, 64, 'multi3')
        pmap = Conv2D(1, (1, 1), activation='sigmoid', name='pmap')(multi3)
        
        q_conv1 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv1')(pmap)
        q_conv2 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv2')(q_conv1)
        q_conv3 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv3')(q_conv2)
        q_conv4 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='q_conv4')(q_conv3)
        q_fmap = Conv2D(1, (1, 1), activation='relu', padding='same', name='q_fmap')(pmap)
        q_flat = Flatten(name='q_flat')(q_fmap)
        q = Dense(1, activation='sigmoid', name='q')(q_flat)
        
        return pmap, q
        
    def get_losses(self):
        return ['binary_crossentropy', 'mse']


class DRIIMNet(BaseNet):

    @staticmethod
    def _normal_architecture(inputs):
        concat_axis = 3
    
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = _get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = _get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = _get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = _get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = _get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        maps = Conv2D(3, (1, 1), activation='sigmoid', name='alr_maps')(conv9)

        flat_conv5 = Flatten()(conv5)
        fc1 = Dense(256, activation='relu')(flat_conv5)
        p75 = Dense(3, activation='sigmoid', name='p75')(fc1)
        p95 = Dense(3, activation='sigmoid', name='p95')(fc1)
        
        return maps, p75, p95

    def get_losses(self):
        return ['binary_crossentropy', 'mse', 'mse']


if __name__ == '__main__':
    net = DRIIMNet()
    model = net.create_model(img_shape=(512, 512, 1))

    print model.summary()
