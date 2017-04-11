#**Traffic Sign Recognition** 
John Bang, 4/8/17

##Overview

###As a part of the Udacity Self-Driving Car Engineer Nanodegree program, I build a traffic sign recognition classifier using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The educational motive is to put into practice convolutional neural networks and deep learning on a classification task.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1.0]: ./data_visualizations/num_examples_by_sign_type.png "Number of Examples by Sign Type"
[image1.1]: ./data_visualizations/16_random_training_images.jpg "16 Random Training Images with Sign Labels"
[image2.0]: ./data_visualizations/4_original_training_images.jpg "4 Training Images with Sign Labels"
[image2.1]: ./data_visualizations/4_grayscaled_training_images.jpg "4 Grayscaled Images with Sign Lables"
[image3.0]: ./data_visualizations/original_training_image.jpg "Original Image"
[image3.1]: ./data_visualizations/64_augmentation_examples.jpg "64 Augmentation Examples"
[image4.0]: ./data_visualizations/web_image_resizing.jpg "Web Images and Their Resized Versions"
[image4.1]: ./data_visualizations/web_image_top5.jpg "Web Image Top 5 Predicted Classes"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Admittedly, my jupyter notebook code could use a bit of refactoring, cleanup and additional comments, but it is functional and should be readable as it pertains to a high level view of what is being accomplished, especially since it's written in python. Here is a link to my [project code](https://github.com/johnybang/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples of each sign type exist in the training, validation, and test sets. The nonuniform distribution of sign types is representative of the prior probability of encountering such signs in a typical driving scenario. Therefore, the nonuniform class distribution is a desireable property of this dataset.

![Number of examples of each sign type][image1.0]

And here is a grid of 16 random images from the test set, with sign type.

![16 random test images][image1.1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook. I define several helper functions for preprocessing which will be used during the training experiments. The actual preprocessing occurs at experimentation runtime so that I can programmatically permute several different preprocessing options, but I provide a few visual examples and motivations below.

#####Grayscale
I decided to experiment with grayscale for a number of reasons:

* It may make the system more robust to poor lighting conditions where color may not be reliable
* It reduces the number of input features. With all other factors held constant (number of layers, etc.), this could reduce risk of overfitting and makes the training/model run faster
* [This paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) had some success with grayscale on this dataset

Here are 4 training images before and after grayscaling.

![4 original training images][image2.0]
![4 grayscaled training images][image2.1]

#####Normalization
Normalization a common first step in machine learning for making sure that the input features are similar in scale and zero mean due to the behavior and statistical assumptions of various machine learning algorithms. Ignoring this concern can lead to slow convergence or lack of convergence.

In our case, relative scale isn't really a concern since all the pixels are on the same scale. However, depending on architectural elements like activation function, different normalization (and initialization) strategies can be used. We wish to enable our neural network to adapt efficiently and avoid common problems like "dead units" which can occur when the gradient of a particular node approaches zero. For example, sigmoid's gradient approaches zero at +-infinity which motivates zero mean features. Relu, instead has zero gradient when less than zero and a constant gradient above zero so I'm motivated to find out if zero-mean is necessary for relu. To that end, I made a helper function to try 3 different normalization approaches for my own edification:

1. MaxScale: Scale by max possible pixel value (255) so the range is (0,1)
2. MaxScaleMinusMean: MaxScale then subtract the mean so the range is (-0.5, 0.5)
3. ImageStandardize: Treating all channels of pixels as one statistical group, standardize the distribution of pixels within each image. (A possibly naive attempt to reduce the effect of abnormally bright or dark images.)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

##### Train/Validation Split
In the latest version of this project template, the training/validation split is already done by the Udacity instructor, so that aspect of the question is no longer relevant. The pickle files return training, validation and test sets. If that hadn't been the case, I probably would have aimed for a 80/20 split on training vs validation. Interestingly, [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) pointed out that it must be done carefully in order to avoid accidently having highly correlated validation and test sets, since the images (at least in the original GTSRB) exist in sets of 30 images of the same sign. You'd have to be careful not to include a different instance of the exact same sign in the training and validation sets. 

The number of examples in each set was mentioned above but I repeat the info here for convenience:

* The size of training set is 34799
* The size of validation set is 4410

#####Image Augmentation
The code for this step is in code cell 5.

I create an image augmentation sequence that consists of:

* Random brightness from 75% to 125% of the original
* Random translation +-4 pixels
* Random rotation +-5 degrees
* Random shearing +-8 degrees

The transformations are subtle based on the assumption that the sign detection front-end of a real-world system should behave reasonably. The sequence is meant to perturb the image and encourage the network to be invariant/robust to variations that may naturally occur due to time of day, cameral angle, sign detector error, etc. The sequence defined here is used later at experimentation runtime so I can permute the use of different numbers of augmentation images. Here is a visual example of 64 realizations of the random augmentation sequence:

![Original Image][image3.0]

![64 Augmentation Examples][image3.1]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook, the function LeNet(). 

My final model consisted of the following layers:

| Layer           | Description                                                                             |
|:----------------|:----------------------------------------------------------------------------------------|
| Input           | 32x32x3 RGB Image                                                                       |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6                                              |
| tanh            | nan                                                                                     |
| Max Pooling     | 2x2 stride,  outputs 14x14x6                                                            |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16                                             |
| tanh            | nan                                                                                     |
| Max Pooling     | 2x2 stride, outputs 5x5x16                                                              |
| Flatten         | outputs 400                                                                             |
| Fully Connected | outputs 120                                                                             |
| Fully Connected | outputs 84                                                                              |
| Fully Connected | outputs 43 (the number of classes)                                                      |
| Softmax         | For training; not necessary for classifier deployment if probabilities are not required |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in code cells 7 and 8 of the ipython notebook. 

To train the model, I used an Adam optimizer with a cross-entropy loss function of the softmaxed output to the network. Preliminarily, I manually varied batch size, number of epochs, and learning rate as hard-coded fixed constants. 

However, I eventually recognized a need for a framework which would allow me to test various permutations of these parameters and others. Thus, cell 7 became a helper function which encapsulates a single experiment that returns a "best validation accuracy."  I recognized that when I was manually watching an experiment, I tended to decide to stop training when the validation accuracy stopped improving epoch over epoch.  I decided to quantify/automate this by stopping a particular experiment when the validation accuracy (or loss, it's debatable which to use) hadn't improved for 5 epochs in a row.  This ultimately enabled me to automate various permutations of experiments without babysitting each and every one. Batch size and learning rate were studied using this trial automation framework.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the final accuracy of the model is located in code cell 11 of the Ipython notebook.

As I described above, the seventh cell contains a function encapsulating a training session of an arbitrary parameter set. The eighth cell uses [itertools.product()](https://docs.python.org/3/library/itertools.html) to iterate through all permutations of interest. The result of each permuation is printed in markdown-friendly format using [tabulate()](https://pypi.python.org/pypi/tabulate) and is also written to a csv using [pandas.DataFrame.to_csv()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html).

My final model results were:

* training set accuracy of 98.9%
* validation set accuracy of 96.4% 
* test set accuracy of 93.9%

##### Core Questions

* What was the first architecture that was tried and why was it chosen?
  * My first architecture was LeNet5. This architecture had been used to successfully classify images of handwritten digits. Although the task was not identical to our current task, it was indeed an image classification task.
* What were some problems with the initial architecture?
  * One thing to watch out for would potentially be that the digit recognition task seems less complex. For instance, the digit images were grayscale and there were only 10 output classes.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I first did the parameter sweep on LeNet5 which I describe in the next question. However, I being somewhat unsatisfied with my ~96%-97% validation accuracy, I decided to try a few things without initial success. I didn't ultimately have time to figure out how to make them bear fruit, though I still believe they would with sufficient time to carefully debug and analyze. I describe them in my "Loose Ends" section below after the core questions.
* Which parameters were tuned? How were they adjusted and why?
  * I initially did a few experiments to settle on some of the basics. I wanted to choose a batch size, learning rate, and normalization preprocessing which I could stick with for the remainder of experimentation. I tried various permuations, eventually settling on 256, 0.002, and MaxScaleMinusMean (as described above) respectively.
  * Thereafter, I suspected overfitting may be occurring due to high training accuracy (~99%) vs. lower validation accurcay (~90%). I incorporated various amounts of L2 regularization and added Dropout layers. I had separate tuning coefficients (beta and dropout probability) in the convolutional and fully connected layers of the model as I had read that some practitioners opt not to use dropout, for instance, in the convolutional layers. Ultimately, dropout seemed to help the most.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * It's important to use convolutional layers to build a meaningfully heirarchical feature-detection frontend for the fully connected classifier at the backend. Dropout layers are a powerful way to prevent overfitting by keeping various neurons from becoming overly reliant on each other's information. Another pleasing way to think of Dropout is that it is a way of training an exponential number of networks (albeit not completely independet) and taking an ensemble of their results. The use of ensembles of predictors is known to help with the variance of any particular one, hence it helps with overfitting.

##### Example Outcomes of Core Iterative Experimentation
The following table shows the data that led me to my final set of parameters, sorted by validation accuracy; some (perhaps most) differences in performance are admittedly statistically insignificant and/or can be attributed to random data order shuffling. Nonetheless, I finalized on the parameter set that produced the best validation accuracy on these experiments.

What the fields mean:

* act\_type: activation
* b\_conv: L2 reg param, convolutional layers
* b\_fc: L2 reg param, fully connected layers
* batch\_size: number of examples per call to training\_operation
* best\_epoch: best epoch (number of passes through the entire dataset)
* best\_iter: best iteration (number of calls to training_operation)
* best\_valid\_accy: best validation accy out of all epochs tested (experiment stops when 5 epochs in a row don't improve)
* kp\_conv: keep probability, convolutional layers
* kp\_fc: keep probability, fully connected layers
* learn\_rate: perhaps a little weird to try different values for Adam optimizer, but I did it nonetheless
* norm\_type: normalization approaches described above
* run\_time: compute time for the experiment in minutes

| act_type   |   b_conv |   b_fc |   batch_size |   best_epoch |   best_iter |   best\_valid\_accy |   kp_conv |   kp_fc |   learn_rate | norm_type         |   run_time |
|:-----------|---------:|-------:|-------------:|-------------:|------------:|------------------:|----------:|--------:|-------------:|:------------------|-----------:|
| tanh       |    0.001 |  0.001 |          256 |           22 |        2992 |          0.970975 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   2.09816  |
| tanh       |    0     |  0.001 |          256 |           33 |        4488 |          0.968027 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   2.93872  |
| tanh       |    0.001 |  0.001 |          256 |           18 |        2448 |          0.965533 |       1   |     0.5 |        0.002 | MaxScale          |   1.69413  |
| tanh       |    0.001 |  0.001 |          256 |           16 |        2176 |          0.965533 |       1   |     0.5 |        0.002 | ImageStandardize  |   1.50452  |
| tanh       |    0.001 |  0     |          256 |           17 |        2312 |          0.962132 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   1.7112   |
| relu       |    0     |  0.001 |          256 |           18 |        2448 |          0.962132 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   1.7205   |
| tanh       |    0.001 |  0.001 |          256 |           38 |        5168 |          0.961905 |       1   |     0.5 |        0.001 | MaxScaleMinusMean |   3.15909  |
| tanh       |    0     |  0.001 |          256 |           44 |        5984 |          0.961451 |       1   |     0.5 |        0.001 | MaxScaleMinusMean |   3.5172   |
| tanh       |    0     |  0     |          256 |           18 |        2448 |          0.960544 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   1.88612  |
| tanh       |    0     |  0.001 |          256 |           42 |        5712 |          0.95873  |       1   |     1   |        0.002 | MaxScaleMinusMean |   3.6656   |
| relu       |    0     |  0     |          256 |           23 |        3128 |          0.956916 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   2.14523  |
| tanh       |    0.001 |  0     |          256 |           19 |        2584 |          0.956236 |       1   |     1   |        0.002 | MaxScaleMinusMean |   1.8585   |
| relu       |    0.001 |  0     |          256 |           21 |        2856 |          0.955556 |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   1.94728  |
| relu       |    0.001 |  0.001 |          256 |           25 |        3400 |          0.94966  |       1   |     0.5 |        0.002 | MaxScaleMinusMean |   2.2471   |
| relu       |    0     |  0     |          256 |           46 |        6256 |          0.949433 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   3.86023  |
| tanh       |    0.001 |  0.001 |          256 |           22 |        2992 |          0.948753 |       1   |     1   |        0.002 | MaxScaleMinusMean |   2.09331  |
| tanh       |    0     |  0     |          256 |           11 |        1496 |          0.945125 |       1   |     1   |        0.002 | MaxScaleMinusMean |   1.24519  |
| relu       |    0.001 |  0.001 |          256 |           16 |        2176 |          0.943537 |       1   |     1   |        0.002 | MaxScaleMinusMean |   1.60393  |
| relu       |    0     |  0.001 |          256 |           42 |        5712 |          0.94263  |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   3.6258   |
| relu       |    0.001 |  0.001 |          256 |           40 |        5440 |          0.941723 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   3.56088  |
| tanh       |    0     |  0     |          256 |           15 |        2040 |          0.934694 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   1.55082  |
| tanh       |    0     |  0     |          256 |           19 |        2584 |          0.9322   |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   1.86198  |
| tanh       |    0.001 |  0.001 |          256 |           38 |        5168 |          0.930839 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   3.30828  |
| relu       |    0.001 |  0     |          256 |           25 |        3400 |          0.928798 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   2.30266  |
| tanh       |    0.001 |  0     |          256 |           25 |        3400 |          0.927438 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   2.31044  |
| tanh       |    0.001 |  0     |          256 |           11 |        1496 |          0.925397 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   1.24426  |
| relu       |    0     |  0.001 |          256 |           40 |        5440 |          0.92449  |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   3.38724  |
| relu       |    0     |  0     |          256 |           29 |        3944 |          0.924263 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   2.55995  |
| tanh       |    0.001 |  0.001 |          256 |           28 |        3808 |          0.92381  |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   2.5358   |
| relu       |    0     |  0.001 |          256 |           12 |        1632 |          0.921995 |       1   |     1   |        0.002 | MaxScaleMinusMean |   1.29116  |
| relu       |    0.001 |  0.001 |          256 |           23 |        3128 |          0.914059 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   2.08646  |
| tanh       |    0     |  0.001 |          256 |           18 |        2448 |          0.912245 |       0.5 |     0.5 |        0.002 | MaxScaleMinusMean |   1.76894  |
| tanh       |    0     |  0.001 |          256 |           15 |        2040 |          0.910884 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   1.50905  |
| relu       |    0.001 |  0     |          256 |            8 |        1088 |          0.910431 |       1   |     1   |        0.002 | MaxScaleMinusMean |   1.00921  |
| relu       |    0     |  0     |          256 |            7 |         952 |          0.908617 |       1   |     1   |        0.002 | MaxScaleMinusMean |   0.901069 |
| relu       |    0.001 |  0     |          256 |           21 |        2856 |          0.908617 |       0.5 |     1   |        0.002 | MaxScaleMinusMean |   1.95833  |

##### Loose Ends
Being somewhat unsatisfied with my ~96%-97% validation accuracy, I decided to try a few things without initial success. I didn't ultimately have time to figure out how to make them bear fruit, though I still believe they would with sufficient time to carefully debug and analyze. Here they are:

* Data augmentation
  * I added random translation, rotation, and shearing as described above, but sadly this didn't yield an improvement for any of the architectures under test. It even showed degradation; perhaps my perturbations are too dramatic? I'll have to revisit this and debug/analyze when I have time in the future.
* LeNetComplexified
  * This was my attempt to be inspired by [this post](https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad). I added an extra convolutional layer and increased the number of filters in each one by quite a bit. Unfortunately, probably through a fault of my own, it didn't produce an improvement for me. Additionally, the AWS g2.2xlarge EC2 wasn't plugging through this very quickly, limiting the ability to do efficient debugging/experimentation.
* LeNetMultilevel
  * This was my attempt to be inspired by [this Udacity student's project](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and the multilevel feature idea proposed in [this Sermanet et al paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
  * I sent the first convolutional layers to the fully connected network along with the second layer. No improvement right off the bat, so I'll have to revisit this in the future.


###Test a Model on New Images

####1. Choose 9 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web as well as its manually cropped/resized counterpart to be used in test. The resizing is done in code cell 12:

![web images and cropped/resized versions][image4.0]

Some comments regarding each image:

1. **children.jpeg:** This will likely be easy to classify, but perhaps the remainder of the Kindergarten portion could add slight confusion.
2. **danger\_manface.jpeg:** Though most of the sign is captured very well, the man's face may be confusing.
3. **double\_curve.jpeg:** The smallness and skewed angle of the primary sign as well as the inclusion of the secondary sign underneath may make this difficult.
4. **no\_stopping.jpeg:** This is a good shot of the sign, but not examples of this sign type exist in the original dataset, thus it will not make a correct predicition. It will make an incorrect guess.
5. **pedestrian\_speed\_30.jpeg:** The smallness of the two signs and the inclusion of two signs simultaneously will make this very difficult to get even half correct. Whether this would occur in a real system depends on the sign-detection algorithm's behavior.
6. **restricted\_no\_stopping.jpeg:** This is a good shot of the sign, but not examples of this sign type exist in the original dataset, thus it will not make a correct predicition. It will make an incorrect guess.
7. **speed\_60.jpeg:** This is a pretty good shot; should be easier to classify.
8. **speed\_130.jpeg:** This is a good shot, but this speed is not included in the original dataset, thus it will make an incorrect guess.
9. **stop.jpeg:** This is an ideal angle and lighting scenario; should be easy.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model are located in the 13th and 14th cells of the Ipython notebook.

Here are the results of the prediction:

| Actual                               | isCorrect   | isIn43Classes   | Predicted            |
|:-------------------------------------|:------------|:----------------|:---------------------|
| Children crossing                    | True        | True            | Children crossing    |
| General caution                      | True        | True            | General caution      |
| Double curve                         | False       | True            | No entry             |
| No stopping                          | False       | False           | Speed limit (30km/h) |
| Pedestrians and Speed limit (30km/h) | False       | False           | Priority road        |
| Restricted no stopping               | False       | False           | Speed limit (30km/h) |
| Speed limit (60km/h)                 | True        | True            | Speed limit (60km/h) |
| Speed limit (130km/h)                | False       | False           | Speed limit (30km/h) |
| Stop                                 | True        | True            | Stop                 |

Among the 5 images that are among the 43 sign classes in the GTSRB dataset, 4 out of 5 or 80% were predicted correctly. This 80% number is appropriately in the ballpark of the 94% test accuracy computed on the GTSRB dataset.

On the other hand, the model was only able to correctly guess 4 out of all 9 traffic signs or 44% when including classes outside the GTSRB dataset. This motivates some careful thought about the probability of seeing an out-of-dataset sign class and the potential severity of an incorrectly guessed class. I suspect there would at least need to be some prediction probability threshold (and/or a minimum relative probability between the top 2 predicitons) beneath which the system should return a None response to be safe.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for visualizing softmax probabilites of these web images is in the 14th cell of the Ipython notebook.

![web images with top 5 prediction probabilities][image4.1]

Some comments regarding each image:

1. **Children crossing:** This one seems straightforward for the classifier at 95% softmax probability for the correct outcome.
2. **General caution:** This is even more confident at 99.6% for the correct outcome despite the man's face.
3. **Double curve:** The top probability was appropriately low at 55% given that it was an incorrect outcome. The fact that the image was a skewed angle, poorly cropped and included the supplemental sign underneath all may have contributed to this sign's failure.  In addition, the prior probability of this sign type imposed by the relative number of examples of this sign type in the GTSRB dataset is quite low, thus decreasing the likelihood of properly identifying this particular sign type. It would be good to consider carefully modifying the relative frequency of sign types in the original dataset according to not only the probability of encounter but the potential negative cost of misidentification or lack of identification.
4. **No stopping:** Being a class outside the dataset, this sign interestingly is somewhat split on whether it might be one of two incorrect signs at around 40% each. It makes sense that the relative frequency of Speed limit (30km/h) in the GTSRB dataset would make it more likely to come up here in a wild-guess situation.
5. **Pedestrians and Speed limit (30km/h):** Poor confidence. Top number at only 30%.
6. **Restricted no stopping:** Again, not in the dataset classes, but the highest probability is a somewhat high 65% for the prior-biased 30km/h sign. This indicates that a "unknown" probability threshold would likely need to be higher than 65% (unless this one example is a lone outlier).
7. **Speed limit (60km/h):** Easy one at 94% confidence on the right answer.
8. **Speed limit (130km/h):** This is perhaps the trickiest. It is 97% confident that it is a 30km/h sign and it indeed is very similar except for the number one preceding the number 3 on the sign. Something would certainly need to be done to generalize speed limit sign value detection/classification, perhaps a number detector one step deeper after "speed limit" class is found. At the very least, it would need to return "unknown". It seems unsafe for the car to suddenly think the speed limit is 30km/h rather than 130km/h on the highway! Of course, other context would probably prevent such a catastrophe in real life. Then again, at least it didn't mistake a 30km/h sign for a 130km/h sign which would be more likely to occur if both sign types existed in the GTSRB dataset...
9. **Stop:** Clearly the easiest of the bunch. 99.8% confident on the correct outcome.