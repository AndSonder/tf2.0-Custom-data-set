"""
Load PoKemon dataset
"""
import os, glob
import random, csv
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def load_csv(root, filename, name2label):
    # this will create a csv file when I first run it.
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            # 'pokemon\\name\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        print(len(images), images)
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                lable = name2label[name]
                writer.writerow([img, lable])
            print('written into csv file', filename)
    images, labels = [], []
    # read from csv file
    with open(os.path.join(root, filename), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)
    assert len(images) == len(labels)
    return images, labels


def load_pokemon(root, model='train'):
    """
    load pokemon dataset info
    :param root: the root path of dataset
    :param model: train val or test
    :return: image,labels,name2able
    """
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.exists(os.path.join(root, name)):
            continue
        # code each category (use the length of the key)
        name2label[name] = len(name2label.keys())
    images, labels = load_csv(root, 'image.csv', name2label)
    if model == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif model == 'val':  # 20%
        images = images[:int(0.2 * len(images))]
        labels = labels[:int(0.2 * len(labels))]
    else:  # 20%
        images = images[:int(0.2 * len(images))]
        labels = labels[:int(0.2 * len(labels))]
    return images, labels


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean) / std
    return x


def preprocess(x, y):
    """
    preprocess the data
    :param x: the path of the images
    :param y: labels
    """
    # data augmentation, 0~255
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    # resize the image,you can change the value in the another net
    x = tf.image.resize(x, [224, 224])
    # turn around images
    x = tf.image.random_crop(x, [224, 224, 3])
    # # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return x,y


if __name__ == '__main__':
    image_train, lab_train = load_pokemon('pokemon', model='train')
    image_val, lab_val = load_pokemon('pokemon', model='val')
    image_test, lab_test = load_pokemon('pokemon', model='test')
    train_db = tf.data.Dataset.from_tensor_slices((image_train, lab_train))
    train_db = train_db.shuffle(1000).map(preprocess).batch(32)
    val_db = tf.data.Dataset.from_tensor_slices((image_val, lab_val))
    val_db = val_db.map(preprocess).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((image_test, lab_test))
    test_db = test_db.map(preprocess).batch(32)
    print(train_db)
    print(val_db)
    print(test_db)