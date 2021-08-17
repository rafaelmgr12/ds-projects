
# **CIFAR-10 - Object Recognition in Images**

Identify the subject of 60,000 labeled images

## Description

[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)  is an established computer-vision dataset used for object recognition. It is a subset of the [80 million tiny images dataset](http://groups.csail.mit.edu/vision/TinyImages/) and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

Kaggle is hosting a CIFAR-10 leaderboard for the machine learning community to use for fun and practice. You can see how your approach compares to the latest research methods on Rodrigo Benenson's [classification results page](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

![https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

Please cite this technical report if you use this dataset: [Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

Submissions are evaluated on classification accuracy (the percent of labels that are predicted correctly).

## **Submission Format**

For each image in the test set, predict a label for the given id. Your labels must match the official labels exactly {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}. Your submission should have a header.

```
id,label
1,cat
2,cat
3,cat
4,cat
...
```
