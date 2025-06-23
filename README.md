# Machine-Learning-Model-Development
This project demonstrates the implementation and application of foundational machine learning algorithms using Python, Numpy, Pandas, and Matplotlib. The notebooks include hands-on solutions for real-world classification and clustering tasks such as university admissions prediction and digit recognition.

## Logistic Regression

The Logistic Regression part divided into two main parts:
1. *Part 1: Neural Network using NumPy*  
2. *Part 2: Neural Network in PyTorch* 

The goal is to implement and analyze neural network models first from scratch in NumPy, then using PyTorch.

### Key Features
- Comprehensive Data Analysis: Detailed exploration and manipulation of datasets to uncover underlying patterns and relationships.
- Predictive Modeling: Introduction to building and refining predictive models using advanced statistical methods.
- Interactive Visualizations: Leveraging Python's visualization libraries to create interactive plots that effectively communicate the results of data analysis.

### Data Cleaning and Preparation
- Techniques for handling missing data, anomalies, and extracting relevant features for analysis.
![image](https://github.com/user-attachments/assets/26df7643-258f-4bc9-9164-d4d90f3b61bf)

### Statistical Analysis and Data Visualization
- Application of statistical tests to determine significant differences and correlations within the data. 
- Advanced visualization techniques to create dynamic plots and charts for an impactful presentation of data insights.

![image](https://github.com/user-attachments/assets/7d6479bf-974a-41b8-9142-1ae4d845b3ca)

### Predictive Analytics and Case Study
- Construction of models to forecast future trends based on historical data.
- A real-world case study to apply learned techniques, demonstrating the workflow from raw data to actionable insights.
![image](https://github.com/user-attachments/assets/781cbcaa-4e18-4217-8a4d-891cf8f9a2e1)

---

## Neural Network and KNN

The Neural Network and KNN part divided into two main parts:

1. *Feedforward Neural Network using NumPy:*
   
- Implement and analyze a simple neural network for classification from scratch using NumPy.

2. *K-Nearest Neighbors (KNN) Algorithm:*
   
- Apply and evaluate the KNN algorithm for supervised classification, including parameter selection and model assessment.

The goal is to gain hands-on experience with two classic classification algorithms—neural networks (implemented from the ground up) and KNN—comparing their effectiveness on different datasets.

### Key Features

1. Custom Model Implementation: Build neural networks and KNN classifiers from first principles.

2. Hyperparameter Tuning: Experiment with neural network architecture and KNN's 'K' value to optimize accuracy.

3. Comprehensive Evaluation: Visualize decision boundaries and model predictions using Matplotlib.

### Data Preparation
Includes steps for data normalization, feature selection, and handling of training/test splits to ensure fair algorithm comparison.


#### Dataset
For part 1:

The MNIST (Modified National Institute of Standards and Technology database) dataset contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits (10 digits). The handwritten digit images have been size-normalized and centered in a fixed size of 28x28 pixels.

![image](https://github.com/user-attachments/assets/3b4eb621-f9de-4b39-b242-1a9665e63de0)

For part 2:

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes - 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'.

![image](https://github.com/user-attachments/assets/b98e12f5-8d49-44e8-9c41-b8777b5862dc)



### Performance Visualization & Analysis
Plot of three random images and print their labels:

![image](https://github.com/user-attachments/assets/96d32f63-a5a9-4a2f-baf5-ad5edac24ab0)


After choosing a random image and passing it through the network. It should return a prediction - confidences for each class. The class with the highest confidence is the prediction of the model for that image:

![image](https://github.com/user-attachments/assets/e038c939-7b27-4fd2-abc0-740cb5e081da)

![image](https://github.com/user-attachments/assets/2d5dc74c-5f70-4528-860d-a80b03c00318)


### Neural Network - Training

1. Split the dataset into a training set and a validation set. Train-set size: 80% of the total data. Val-set size: the rest 20%.
   
2. Create a dataloader for each set (train_loader and val_loader, see Section 2 for examples).

3.Choose hyperparameters (for now we choose learning_rate=0.005 and num_epochs=5). 

4. Use SGD (Stochastic Gradient Descent) as the optimizer.
   
5. Since it is a multi-class classification task, use "negative log-likelihood loss" as the loss criterion.

6. Train your model on the train-set and evaluate it on the validation-set.

7. During training, for each epoch, track the training loss and validation loss.

training loss drop with each epoch:

![image](https://github.com/user-attachments/assets/fc84b83f-ee72-4c75-9401-3e306c7251d5)

Plot train loss and validation loss as a function of epoch:

![image](https://github.com/user-attachments/assets/fae06544-08b4-43c0-a305-d9fa5517257a)

With the network trained, we can check out it's predictions:

![image](https://github.com/user-attachments/assets/3dc6d7a2-0171-4634-9dd9-a78e19a24e59)

The model's accuracy on the validation-set:

![image](https://github.com/user-attachments/assets/f3667e37-c93e-4ff6-9b94-dabd47baf10c)

---

## KMeans Clustering
This notebook focuses on unsupervised clustering using the K-means algorithm:

### Main Parts:

1. K-Means Clustering from Scratch with NumPy

2. Step-by-step implementation of K-means, from centroid initialization to convergence.

3. Cluster Analysis and Visualization

4. Assign data points to clusters and interpret the results visually.

*The goal is to explore how K-means can be used to identify natural groupings in data, with all components built manually in NumPy.*

### Key Features
Full Clustering Pipeline: Develop all clustering steps (init, assignment, update) without ML libraries.

In-Depth Cluster Evaluation: Assess clustering effectiveness with inertia and visual inspection.

Result Interpretation: Use plots to communicate clustering outcomes and data structure.

### Data Preprocessing
Covers standardization and dimensionality reduction (if needed) to improve clustering quality.

Plot the a convex 2D dataset to explore how many clusters it contains:

![image](https://github.com/user-attachments/assets/bc632ddd-b460-4147-8b92-27428f51e3cf)

Plot the non convex 2D dataset to explore how many clusters it contains:

![image](https://github.com/user-attachments/assets/2a523259-d269-4ad7-9a29-6aa1428db252)


### Visualization and Results
Includes scatter plots and color-coded cluster assignments for clear interpretation.

![image](https://github.com/user-attachments/assets/94a34804-19b3-4711-9d6a-e24f921e0929)

![image](https://github.com/user-attachments/assets/12c97b7f-f1a3-4346-bf3b-268773f4316d)

![image](https://github.com/user-attachments/assets/a7c068ce-97b8-420a-bbc2-074eab7dec07)

![image](https://github.com/user-attachments/assets/818032bd-05b6-45fd-8d8a-fb5dcfcabde5)

![image](https://github.com/user-attachments/assets/7ac92941-3aaa-4dcb-9fa8-774c998f6677)

![image](https://github.com/user-attachments/assets/227423d8-5a6d-40b0-b55c-5dcbfbdaa0ed)

![image](https://github.com/user-attachments/assets/fbb95139-a41a-4ca5-9cfc-657c5a66e468)

![image](https://github.com/user-attachments/assets/6e4ba4ad-95d2-4607-a2b3-34cd6408d6d7)

![image](https://github.com/user-attachments/assets/ff0c2e07-7487-4d97-b2d6-5e1e1e767f1e)

![image](https://github.com/user-attachments/assets/40a37590-0923-4597-8006-dbe03d6faba6)

![image](https://github.com/user-attachments/assets/6472a9b6-4352-4572-a58d-d119c0c29e28)

![image](https://github.com/user-attachments/assets/3894e6e6-ab02-42ce-a9e0-426b5ec8cf4f)

![image](https://github.com/user-attachments/assets/ceb32715-9962-4ba9-918c-b4cc7168239d)

![image](https://github.com/user-attachments/assets/acc35221-f6db-4bdc-b9c5-39b0c6a223fe)


*SUMMARY OF RESULTS:*

2 clusters: 16441.7080

3 clusters: 10065.2186

4 clusters: 3957.6831

6 clusters: 3380.2836

8 clusters: 2897.2321

10 clusters: 2149.9190

20 clusters: 1137.4731

![image](https://github.com/user-attachments/assets/46791e02-a767-4374-8a2b-fa6fb79feaa7)

### PCA

![image](https://github.com/user-attachments/assets/c97f969c-8b64-4324-945c-5c3ac5722645)

VV^T This matrix is the product of V and its transpose. It represents the projection matrix onto the space spanned by the eigenvectors in V, VV^T is not necessarily diagonal but symmetric, with its rank being r and the rest of the elements being zeros. The siize of VV^T is d on d
V^tV relay on the orthonormality of V VV^t secribes the sbspace of span(V)

![image](https://github.com/user-attachments/assets/26bb0320-8369-4182-8b2e-be49517b0bf7)

Performance of the reconstruction above from spaces of dimensions: 3, 10, 100.

![image](https://github.com/user-attachments/assets/af584c1c-99e5-4004-8e50-3e265c8d4592)

### Convolutional Neural Networks



- Convolutional operation: A “filter”, also called a “kernel”, is passed over the image, viewing a few pixels at a time (for example, 3X3 or 5X5). The convolution operation is a dot product of the original pixel values with weights defined in the filter. The results are summed up into one number that represents all the pixels the filter observed.

- Pooling: “Pooling” is the process of further downsampling and reducing the size of the matrix. A filter is passed over the results of the previous layer and selects one number out of each group of values (typically the maximum, this is called max pooling). This allows the network to train much faster, focusing on the most important information in each feature of the image. By sliding the window along the image, we compute the mean or the max of the portion of the image inside the window in case of MeanPooling or MaxPooling.

- Stride: The number of pixels to pass at a time when sliding the convolutional kernel.

- Padding: To preserve exactly the size of the input image, it is useful to add zero padding on the border of the image.

#### The architecture:

Conv layer (10 5x5 Kernels) -> Max Pooling (2x2 kernel) -> Relu -> Conv layer (20 5x5 Kernels) -> Max Pooling (2x2 kernel) -> Relu -> Hidden layer (320 units) -> Relu -> Hidden layer (50 units) -> Output layer (10 outputs).

![image](https://github.com/user-attachments/assets/7aca4309-70f6-4aaa-a7b5-a4e3009e2b2c)

Training the model on the train set:

![image](https://github.com/user-attachments/assets/3bf0befc-c5c1-4135-aeda-fc6a5a05db34)

![image](https://github.com/user-attachments/assets/ffeeb320-3586-4a9c-b451-e524cbf1c3ac)






---

### Requirements
- Python 3.7 or higher  
- Jupyter Notebook or JupyterLab  
- NumPy  
- Matplotlib  
- PyTorch  
- Torchvision  

You can install the dependencies using pip:
```bash
pip install numpy matplotlib torch torchvision

