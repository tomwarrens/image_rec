import keras
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

ISZ = 160
N_Cls = 10 #10
smooth = 1e-12
ids_to_show = ['6050_4_4', '6080_1_3', '6080_4_4', '6100_1_3',
               '6110_2_3', '6140_3_1', '6150_4_0', '6150_4_3']

classes_name = {'1' : 'Buildings', '2': 'Misc. Manmade Structures',
                '3': 'Road', '4': 'Track', '5': 'Trees',
                '6': 'Crops', '7': 'Waterway',
                '8': 'Standing Water', '9': 'Vehicle Large',
                '10': 'Vehicle Small'}

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def RGB(image_id):
    filename = os.path.join(inDir, "three_band/three_band", '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join("tf_files", 'sixteen_band/sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def M_to_RGB(image_id):
    filename = os.path.join("tf_files", 'sixteen_band/sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img[:,:,4] #red
    rgb[:,:,1] = img[:,:,2] #green
    rgb[:,:,2] = img[:,:,1]
    return stretch_8bit(rgb)

def predict_id(id, model, trs):
    img = M(id)
    x = stretch_n(img)
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]

def check_predict(id='6120_2_3'):

    for id in sorted(DF.ImageId.unique().tolist()):
        msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
        #msk2 = predict_id(id, model, [0.1, 0.1, 0.4, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005])
        #img_M = M(id)
        img  = RGB(id).astype(np.uint8)
        id_folder = "results_2/{}".format(id)
        if (os.path.isdir(id_folder)== False):
            os.makedirs(id_folder)

        for i in range(1):
            #msk_file = os.path.join(id_folder, "msk_{}.png".format(classes_name[str(i + 1)]))
            #msk_file_2 = os.path.join(id_folder, "msk2_{}.png".format(classes_name[str(i + 1)]))
            fig = plt.figure(frameon= False)
            fig.set_size_inches(msk.shape[1] / 100, msk.shape[2]/ 100)
            ax =plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(msk[i], cmap = plt.get_cmap('gray'))
            plt.show()
          #  fig.savefig(msk_file)
			#cv2.imwrite(msk_file, msk[i])
           # plt.savefig(msk_file,pad_inches = 0, bbox_inches ="tight")
           # plt.close()
           # plt.figure(figsize=(msk2.shape[1]/100, msk2.shape[2]/100))
           # plt.imshow(msk2[i], cmap = plt.get_cmap('gray'))
           # plt.savefig(msk_file_2, pad_inches = 0, bbox_inches ="tight")
           # plt.close()

def predict_single_id(id = '6100_1_3'):
    img = plt.imread("images_/example_images_dstl/{}.png".format(id))
    fig = plt.figure(figsize=(14, 24))
    plt.imshow(img)
    plt.show()
    fig = plt.figure(frameon = False, figsize = (14, 24))
    columns = 4
    rows = 5
    for i in range(1, 9):
        msk = plt.imread("results/{}/msk_{}.png".format(id, classes_name[str(i)]))[:, :, :3]

       # fig = plt.figure(frameon=False, figsize=(14, 24))
       # fig = plt.figure(figsize=(14, 24))
        #fig.add_subplot(rows, columns, i)
       # fig.set_size_inches(msk.shape[1] / 100, msk.shape[2] / 100)
       # ax = plt.Axes(fig, [0., 0., 1., 1.])
       # ax.set_axis_off()
       # fig.add_axes(ax)
        print(classes_name[str(i)])
        fig = plt.figure(frameon=False, figsize=(14, 24))
        plt.imshow(msk, cmap=plt.get_cmap('gray'))
        plt.show()

def get_unet():
    inputs = Input((8, ISZ, ISZ))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
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
    print("conv5 has shape {}".format(conv5.get_shape))
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(N_Cls, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model

