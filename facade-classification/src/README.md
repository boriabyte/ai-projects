# Building Façade Classification using Unsupervised Learning (k-means clustering)

keywords: computer vision, unsupervised learning, k-means clustering

## Summary

Computer Vision project that places building façades into three categories: **industrial**, **vernacular**, **transitional**, based on handcrafted geometric processing of data obtained from a 400 image dataset, found [here](https://www.kaggle.com/datasets/balraj98/facades-dataset).

## Mathematics

The image is turned into grayscale, after which three fundamental feature are extracted in order to obtain meaningful features: **edges** ([Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator)), **reflection symmetry** and **periodicity**. These three act as 
distinct characterstics chosen to differentiate between types of façades. Industrial façades tend to exhibit greater geometric regularity and modular repetition, while vernacular façades tend to show greater contextual variation and locally derived proportions. Below are some 
examples:

<p align="center">
  <img src="images/vernacular.png" alt="Vernacular façade" width="600">
  <br>
  <em>Example of a vernacular façade</em>
</p>

<p align="center">
  <img src="images/industrial.png" alt="Industrial façade" width="600">
  <br>
  <em>Example of an industrial façade</em>
</p>

The category and clustering assignments are also made by refraining from using pre-built methods and libraries, and done as manually as possible. Standardization of features is the first step in the categorization, followed by dimensionality reduction to a two-dimensional space using Principal Component Analysis. K-means clustering on the obtained results is done. A performance metric, expressed as 'assignment confidence' is computed as well, to track the performance of the system.

## Running the project

Running the application is trivial:

* for feature extraction, be sure to have images in local path; the **features.csv** file will be automatically generated after running _symmetry_features.py_
* training is done by running _train_cluster_model.py_ and an **.npz** file will be generated in the project dir
* _predict_cluster.py_ will then assign each data point to the nearest center by following the rules briefly explained above
* _metrics.py_ is optionally run if metrics are of interest to the user
