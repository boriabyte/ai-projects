# Building Façade Classification using Unsupervised Learning (k-means clustering)

computer vision, unsupervised learning, k-means clustering

## Summary

Computer Vision project that places building façades into three categories: **industrial**, **vernacular**, **transitional**, based on handcrafted geometric processing of data obtained from a 400 image dataset, found [here](https://www.kaggle.com/datasets/balraj98/facades-dataset).

## Mathematics

The image is turned into grayscale, after which three fundamental feature are extracted in order to obtain meaningful features: **edges** ([Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator)), **reflection symmetry** and **periodicity**. These three act as 
distinct characterstics chosen to differentiate between types of façades. Industrial façades tend to exhibit greater geometric regularity and modular repetition, while vernacular façades tend to show greater contextual variation and locally derived proportions. Below are some 
examples:


![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
