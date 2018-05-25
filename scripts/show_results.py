import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd

classes_name = {'1' : 'Buildings',# '2': 'Misc. Manmade Structures',
                '3': 'Road', '4': 'Track', '5': 'Trees',
                '6': 'Crops', '7': 'Waterway',
                '8': 'Standing Water', '9': 'Vehicle Large',
                '10': 'Vehicle Small'}

DF = pd.read_csv("tf_files/train_wkt_v4.csv")
ALL_IMAGES = DF.ImageId.unique().tolist()

def get_mask(image_id, class_type):
    mask = os.path.join(FLAGS.mask_dir, image_id)
    filename = os.path.join(mask, "msk_{}.png".format(classes_name[class_type]))
    return plt.imread(filename)

def overlay_image(image_id, class_type):
    filename = os.path.join(FLAGS.rgb_dir, image_id + ".png")
    output_name = os.path.join(FLAGS.output_folder, image_id + "_overlay_{}".format(classes_name[class_type]))
    img = plt.imread(filename)
    mask = get_mask(image_id, class_type)
    mask[:, :, 2] = 0
    fig = plt.figure(frameon=False)
    fig.set_size_inches(mask.shape[0] / 100, mask.shape[1] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, interpolation='none', cmap='gray')
    ax.imshow(mask[:, :, :3], interpolation='none', cmap=plt.get_cmap("Reds"), alpha = 0.3)
    fig.savefig(output_name, pad_inches = 0, bbox_inches = 'tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='results/',
        help='Path to folders of Masks'
    )
    parser.add_argument(
        '--rgb_dir',
        type=str,
        default='tf_files/M_to_RGB/',
        help='Where to get the RGB images.'
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default='tf_files/mask_over_image/',
        help='Where to save Images overlayed.'
    )

    parser.add_argument(
        '--class_type',
        type=str,
        default='ALL',
        help='Whether to show a specific class or all (default)'
    )

    parser.add_argument(
        '--image_ids',
        type=str,
        default='ALL',
        help='Whether to show a specific image or all (default)'
    )
    FLAGS, unparsed = parser.parse_known_args()

    classes = FLAGS.class_type
    images = FLAGS.image_ids
    range_without_manmade = [i for i in range(1, 11) if i != 2]

    if images != 'ALL':
        if classes != 'ALL':
            overlay_image(images, classes)
        else:
            for i in range_without_manmade:
                overlay_image(images, str(i))
    else:
        if classes != 'ALL':
            for i in ALL_IMAGES:
                overlay_image(i, classes)
        else:
            for i in ALL_IMAGES:
                for j in range_without_manmade:
                    overlay_image(i, str(j))

