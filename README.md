# tf2.0 :Custom data set

Chinese link: https://blog.csdn.net/python_LC_nohtyp/article/details/104685251

## Brief

In the process of learning Deep Learning,we will inevitably use custom DS beacesuse of requirements. This article briefly introduces how to customize DS.

## PoKemon Dataset

We will use PoKemon Dataset at this time 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305215327993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305215437409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70)

### Splitting
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305215556872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70)
## steps
### Load data in csv

First let's deal with the pictures. Below are our dataset floders. Each folder corresponds to the corresonding PoKemon.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305215808654.png)

Now we want to write them into csv file. The effect is as follows.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200306103143838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70)

How can we do that? First let's think about the idea,first we should read the folder with Pokemon image,and then write the csv file.

We use a dictionary to save our picture addresses and labels

```python
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
        ...
```
​	Ok !Now we have the dictionary include Pokemon pictures path and tags,and then write it into csv file.

```python
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
```
In this way we have the address of the Pokemon pictures and tags.

Now we can read csv file in the function load_csv().

```python
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
```
After the we call load_csv function in function load_pokemon. function load_csv will return the pictures' path and the tags to us

```python
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
```
### Preprocess the data

Now we only get the address of the pictures but it cannot be directly used for training,so we need to preprocess
#### Read and Resize
First we read the image information and crop it.
Define a preprocess function

```python
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
```

#### Data Augmentation

```python
 # # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255
```

####  Normalize
Here img_mean and img_std are the values obtained from the millions of data sets in imgNet,we can just use them directly.

```python
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean) / std
    return x
```
Ok. Now we can add codes after preprocess as follows

```python
    # 0~1 => D(0,1)
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return x,y
```

Now we can load our train_db ,val_db and test_db.

```python
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
```

Complete code is in the pokemon.py

"# tf2.0-Custom-data-set" 
"# -tf2.0-Custom-data-set" 
"# -tf2.0-Custom-data-set" 
