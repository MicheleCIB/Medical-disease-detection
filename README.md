# Medical-disease-detection
Parkinson, Heart Disease, and Breast Cancer detection with python
Dataset 1
The dataset was created by Max Little of the University of Oxford, in collaboration with the National 
Centre for Voice and Speech, Denver, Colorado, who recorded the speech signals. The original study 
published the feature extraction methods for general voice disorders. This dataset is composed of a 
range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each 
column in the table is a particular voice measure, and each row corresponds one of 195 voice 
recording from these individuals ("name" column). The main aim of the data is to discriminate healthy 
people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD. 
The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one 
voice recording. There are around six recordings per patient, the name of the patient is identified in 
the first column.
Attribute information:
• name - ASCII subject name and recording number
• MDVP:Fo(Hz) - Average vocal fundamental frequency
• MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
• MDVP:Flo(Hz) - Minimum vocal fundamental frequency
• Five measures of variation in Frequency
▪ MDVP:Jitter(%) - Percentage of cycle-to-cycle variability of the period duration
▪ MDVP:Jitter(Abs) - Absolute value of cycle-to-cycle variability of the period duration
▪ MDVP:RAP - Relative measure of the pitch disturbance
▪ MDVP:PPQ - Pitch perturbation quotient
▪ Jitter:DDP - Average absolute difference of differences between jitter cycles
• Six measures of variation in amplitude
▪ MDVP:Shimmer - Variations in the voice amplitdue
33 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection',
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)
▪ MDVP:Shimmer(dB) - Variations in the voice amplitdue in dB
▪ Shimmer:APQ3 - Three point amplitude perturbation quotient measured against the 
average of the three amplitude
▪ Shimmer:APQ5 - Five point amplitude perturbation quotient measured against the 
average of the three amplitude
▪ MDVP:APQ - Amplitude perturbation quotient from MDVP
▪ Shimmer:DDA - Average absolute difference between the amplitudes of consecutive 
periods
• Two measures of ratio of noise to tonal components in the voice
▪ NHR - Noise-to-harmonics Ratio and
▪ HNR - Harmonics-to-noise Ratio
• status - Health status of the subject (one) - Parkinson's, (zero) - healthy
• Two nonlinear dynamical complexity measures
▪ RPDE - Recurrence period density entropy
▪ D2 - correlation dimension
• DFA - Signal fractal scaling exponent
• Three nonlinear measures of fundamental frequency variation
▪ spread1 - discrete probability distribution of occurrence of relative semitone variations
▪ spread2 - Three nonlinear measures of fundamental frequency variation
▪ PPE - Entropy of the discrete probability distribution of occurrence of relative 
semitone variations
Data cleaning, data visualization and preprocessing
After importing the libraries needed to carry out our study, we proceed to import the dataset and check 
its information, any null or missing values. Given the absence of the latter, we can proceed with the 
analysis. We now study through data visualisation the characteristics of our dataset, seeing the distribution 
frequencies and any outliers. We also look at the correlation matrix, to see the relationships between 
the variables. We set the variable 'status' as the output of our model, after which we proceed with the scaling of all 
variables. Of the various techniques, we choose the standard scaler. This is a normalisation technique 
commonly used in the preparation of data for machine learning. Its main function is to transform the 
characteristics (variables) of a dataset so that they have a zero mean and a unit standard deviation. In 
other words, the Standard Scaling process makes the characteristics of the dataset more 'standardised' 
or 'normal'. Here is what the StandardScaler does: it calculates the mean and standard deviation for 
each feature in the dataset; it subtracts the mean of each feature from the feature values. This process 
centralises the data, bringing the mean of the features to zero; it divides each feature value by the 
standard deviation of the corresponding feature. Finally, we split the data into traing and test sets, 
with a ratio of 3 to 1.
Comparison of models and choice of the best performing
All that remains is to run the chosen models and evaluate their performance. Let’s see which models 
we decided to use and briefly the characteristics of each one:
LOGISTIC REGRESSION: it is a powerful statistical tool and machine learning algorithm used to 
analyse and predict relationships between a binary (or categorical) dependent variable and one or 
more independent variables. Unlike linear regression, which is suitable for continuous dependent 
variables, logistic regression is specifically designed to model the probability of a binary outcome, 
such as the success or failure of an event, the presence or absence of a characteristic, or the 
classification into two categories. The algorithm is based on a logistic (or sigmoid) function that maps 
a wide range of values into a range between 0 and 1. This mapping allows the output of the algorithm 
to be interpreted as the probability of belonging to one of the two classes. Logistic regression 
estimates the coefficients of the independent variables that maximise the likelihood of the observed 
data according to the model. The use of logistic regression is wide and varied. In the health sector, 
for example, it can be applied for the diagnosis and prediction of diseases. Clinical data and patient 
characteristics can be used as independent variables to predict the probability of a clinical outcome, 
such as the presence of a medical condition or the success of a therapy. This approach provides 
physicians with a tool for making informed and personalised decisions. Furthermore, logistic 
regression is also used in other contexts within the healthcare system, such as the assessment of 
patient satisfaction, the analysis of factors influencing therapeutic compliance or the identification of 
determinants of healthcare expenditure. The algorithm can be adapted to the specific requirements of 
diagnostic or predictive problems, allowing a better understanding of the factors influencing medical 
decisions and the effectiveness of treatment; 
NAIVE BAYES: it is an algorithm is a probabilistic classifier based on Bayes' Theorem. This 
algorithm is particularly suitable for the classification of text and categorical data, and is known for 
its simplicity and speed of training. The name 'Naive' derives from the assumption of conditional 
independence made by the algorithm, i.e. the assumption that all features are independent of each 
other given a certain output. This assumption greatly simplifies probabilistic calculations, although it 
may not always be realistic in practice. However, despite this simplification, Naive Bayes has proven 
surprisingly effective in many real-world applications. The basic idea of Naive Bayes is to calculate 
the probability that a data item belongs to a certain class. This probability is calculated using Bayes' 
Theorem, which involves the a priori probability of the class, the probability of the features given a 
certain output, and the probability of the features themselves. A common example of the application 
of Naive Bayes is in text analysis, such as the categorisation of emails into spam or non-spam. In this 
case, the features could be the words in the text and the output could be the class 'spam' or 'non-spam'. 
The algorithm calculates the probability that a certain email is spam or non-spam based on the words 
in the text. Naive Bayes is particularly suitable when dealing with high feature sizes and categorical 
data, as in the case of textual analysis. However, it is important to note that the assumption of 
independence between features may lead to suboptimal performance when dependencies between 
features are significant. Furthermore, Naive Bayes may have difficulty handling continuous attributes 
or missing data. In the medical field, Naive Bayes can be used for diagnosing diseases or predicting 
clinical outcomes. For example, it could be used to classify patients according to clinical 
characteristics in order to determine the presence or absence of a certain disease. It is important to 
adapt the algorithm to the specific needs and characteristics of the medical domain, ensuring that the 
assumptions are appropriate and that the model is accurate and interpretable;
RANDOM FOREST: The Random Forest is an algorithm that has gained prominence in the field of 
machine learning due to its robust nature and ability to provide accurate and reliable predictions on a 
wide range of problems. Its strength comes from combining multiple models, each of which is a 
decision tree, into a 'forest' structure. Each tree is trained on a different subset of the training data, 
randomly selected through a process known as 'bagging' (bootstrapped aggregating), which involves 
the random selection of data with replacement. This variation in the training of individual trees helps 
prevent over-training and ensure that the forest as a whole is generalisable to new data.The distinctive 
aspect of the Random Forest is its ability to combine the predictions of each tree within the forest to 
obtain a final prediction. In classification tasks, this is often done through the process of 'voting', 
where each tree 'votes' for the predicted class and the class with the most votes becomes the final 
prediction. In regression problems, the tree predictions are instead averaged to obtain the final value. 
The Random Forest offers numerous advantages, including the ability to handle missing data, detect 
complex patterns in the data and handle both numerical and categorical variables without the need 
for intensive pre-processing. Furthermore, random trees can be easily parallelised, making rapid 
training possible even on large data sets. This versatility makes them a popular choice for tackling 
complex diagnostic tasks in the healthcare sector. In the field of medical diagnosis, for instance, the 
use of Random Forests can be extremely advantageous. The algorithm can be trained on clinical data, 
medical images and other relevant information to provide predictions about the presence of certain 
diseases or conditions. Furthermore, it can be used to select the most significant variables that 
contribute to diagnosis, providing medical experts with valuable information on the most relevant 
features. Similarly, Random Forest can be applied to optimise and accelerate hospital processes, e.g. 
in resource allocation or forecasting the demand for healthcare services. In both cases, the 
combination of machine learning and Random Forest can contribute significantly to the effectiveness, 
efficiency and accuracy of decisions made in the healthcare sector; 
EXTREME GRADIENT BOOSTING (XGBOOST): is an extremely powerful and versatile machine 
learning algorithm. Based on the concept of boosting, XGBoost combines a number of weak models, 
often decision trees, to create a strong and robust model. This means that it can successfully tackle 
both classification and regression problems. One of the distinguishing features of XGBoost is its 
ability to handle overfitting through various regularisation techniques, including reducing the 
quadratic error (L2 regularisation) and limiting the number of terminal nodes in tree leaves 
(min_child_weight). These techniques help to create more stable models suitable for a variety of 
datasets. XGBoost is known for its computational efficiency, which stems from its optimised 
implementation and ability to exploit parallelism. It can be run on multicore and distributed hardware, 
greatly accelerating training times. In addition, XGBoost offers intelligent handling of missing data, 
eliminating the need to impute missing values in advance. This flexibility is especially valuable when 
working with real data, which often contains missing values. Another advantage of XGBoost is its 
interpretability. It provides information on the importance of features, allowing users to understand 
which variables most influence the model's predictions; 
KNN: The k-Nearest Neighbours (k-NN) algorithm is a machine learning method mainly used for 
classification and regression problems. The basic idea of k-NN is quite intuitive: when assigning a 
new observation to a certain class, the algorithm considers the 'nearest neighbours' of that observation 
in the training set to determine which class it should belong to. The term "k" in k-NN represents the 
number of nearest neighbours that the algorithm considers. For example, if k is set to 3, the algorithm 
will look at the three closest points to the observation to be classified and determine the most common 
class among these three neighbours. This approach is based on the assumption that similar 
observations tend to have similar labels. The implementation of k-NN requires a distance or similarity 
metric to assess the closeness between observations. One of the most commonly used distance metrics 
is the Euclidean distance. Once the distances between the observation to be classified and all other 
observations in the training set have been calculated, the algorithm selects the k closest neighbours 
and assigns the observation to the class most represented among these neighbours. In the medical 
context, the k-NN algorithm can be used for various applications. For example, it can be used to 
classify patients according to certain clinical characteristics in order to diagnose a particular disease 
or predict the risk of developing it. Furthermore, it can be used for medical image segmentation, 
identifying regions of interest within an image. However, the effectiveness of the k-NN algorithm 
depends on the appropriate choice of k-value and the quality of the training data. Too low a k-value 
may lead to decisions that are too influenced by individual points, while too high a value may lead to 
too general a classification. Furthermore, the algorithm can be sensitive to noise in the data;
DECISION TREE: it is a machine learning algorithm widely used for both classification and 
regression problems. This algorithm is based on a tree-like structure in which each node represents a 
decision or test on a data characteristic and each branch corresponds to a possible outcome of that 
decision. The leaves of the tree represent the membership classes or output values of the regression. 
The main objective of a decision tree is to subdivide the dataset into homogeneous subsets in terms 
of class or output value. This subdivision takes place through a series of decisions based on the 
characteristics of the data. The idea is to create a model that can make decisions based on a series of 
"if-yes/if-no" questions about the data, leading to a classification or value estimate.The construction 
of a decision tree starts with a root node representing the entire dataset. The algorithm then selects 
the best feature to split the data according to a certain impurity metric, such as the Gini index or 
entropy. Once the subdivision has been performed, child nodes representing the resulting subsets are 
created. The process of subdivision and creation of child nodes is repeated iteratively until a 
termination condition is reached, such as when a maximum tree depth is reached or when the number 
of points in a node is below a defined threshold. Decision trees offer several advantages, including 
ease of interpretation and the ability to capture complex relationships in the data. However, they can 
be prone to overfitting, i.e. fitting too closely to training data and losing the ability to generalise to 
new data; 
SVMS: Support Vector Machines (SVMs) represent a powerful machine learning algorithm used for 
classification and regression problems. The main objective of SVMs is to optimally separate different 
classes of data in a multidimensional space by exploiting the so-called support vectors, i.e. the 
boundary points between the different classes. The idea behind SVMs is to find a hyperplane that 
maximises the margin between the different data classes. This margin represents the distance between 
the closest training examples of each class and the hyperplane.The support vectors are precisely those 
examples that are closest to the hyperplane and are crucial in determining its position and orientation. 
The SVM approach is also particularly effective when the data are complex and non-linear. In this 
case, SVMs can exploit a technique called 'kernel trick', which allows them to transform the original 
data space into a high-dimensional space where class separation can be easier. This allows SVMs to 
tackle problems that would not be solvable by a simple hyperplane. In the medical context, SVMs 
have been widely used for the classification of medical images, for instance to diagnose diseases from 
radiological images or to identify cancerous cells from microscopic images. SVMs can also be used 
to analyse laboratory data and clinical signs in order to predict the onset of certain diseases or to aid 
in the selection of therapies. 
Model evaluation
We then move on to the evaluation of the individual models, via the classification report: this is an 
important tool used to evaluate the performance of a machine learning model in a classification 
problem. This report provides a detailed overview of the model's performance for each target class 
within the classification problem. One of the main metrics contained in the classification report is 
"accuracy", which indicates the percentage of positive predictions made by the model that are actually 
correct. In other words, accuracy measures the model's ability to avoid false positive errors. Another 
crucial metric is 'recall' (or 'sensitivity'), which measures the percentage of actual positive instances 
that the model is able to correctly identify. Recall is particularly useful when you want to make sure 
that you do not miss important positive instances. F1-score' is a harmonic average between precision 
and recall and is useful when you want to balance these two metrics. The F1-score is particularly 
advantageous when classes in the dataset are unbalanced or when balanced performance is sought. 
Finally, the 'support' represents the total number of instances in the target class. This value can be 
useful in understanding how well the different classes are represented in the dataset. From the data 
obtained, we see that the best result is obtained through SVMS:
Figure 12: results obtained implementing the support vector machine
Hyperparameter tuning
To try and increase performance, we try the technique of hyperparametric tuning. For support vector 
machines (SVMs), it is an essential process to optimise the performance of an SVM model in a 
classification problem. This involves selecting the kernel type, such as linear, polynomial or RBF, 
and tuning the hyperparameters C and gamma. C determines the trade-off between the accuracy on 
the training set and the generalisation of the model, while gamma influences the flexibility of the 
RBF kernel function. It is also important to consider kernel-specific parameters, such as the degree 
of the polynomial or the width of the RBF kernel, depending on the type of kernel chosen. The tuning 
process can be guided by search methods such as grid search, random search or Bayesian optimisation 
and should include cross-validation to avoid overfitting. In this case we have the grid search: is a 
systematic method that examines all possible combinations of hyperparameters from a predefined 
grid. For each combination of hyperparameters, it trains the model using the training set and evaluates 
the performance using the validation set or through a cross-validation technique. At the end of the 
process, it returns the hyper-parameter combination that performed best according to a specified 
metric (e.g. accuracy, F1-score, etc.). Grid search is useful when one has sufficient computing 
resources and wishes to examine all possible combinations. Finally, the optimised model is tested on 
an independent test set to confirm its performance.
Final considerations
Medical data may vary significantly between different geographic, ethnic or demographic 
populations. It is important to consider this heterogeneity when collecting, analysing and interpreting 
data. It may be necessary to subdivide the data into subgroups to account for the differences. When 
developing a model or drawing conclusions based on medical data, it is important to consider the 
model's ability to generalise to different populations. The model should be valid not only for the 
source population but also for other similar populations. To be more statistically significant, the 
sample should be enlarged. Such high accuracy indicates that the model is able to make correct 
predictions in the vast majority of cases, which is a promising result. SVMs are known for their ability 
to create an optimal separation hyperplane that maximizes the distance between classes, which makes 
them effective in handling classification problems even in high-dimensional spaces. SVMs allow the 
use of custom kernels, such as the polynomial kernel or the radial kernel (RBF), which allow 
modeling nonlinear relationships in the data, thus extending their applicability to a wide range of 
problems. SVMs also offer greater control over the decision functions through the use of parameters 
such as the penalty term C. This allows you to balance the trade-off between maximizing the margin 
and reducing classification errors. As possible downsides, training an SVM on large datasets can 
require a significant amount of time and computational resources, especially when using complex 
kernels. Choosing the most suitable kernel itself can be a challenge and may require careful 
experimentation to achieve the best performance. Furthermore, an inappropriate choice of kernel can 
lead to overfitting or underfitting problems. SVMs tend to be less interpretable than simpler models 
such as linear regressions or decision trees. The optimal separating hyperplane is often difficult to 
visualize and explain intuitively.
Dataset 2
The data used in this study were gathered from 188 patients with PD (107 men and 81 women) with 
ages ranging from 33 to 87 at the Department of Neurology in CerrahpaÅŸa Faculty of Medicine, 
Istanbul University. The control group consists of 64 healthy individuals (23 men and 41 women) with ages varying between 41 and 82. During the data collection process, the microphone is set to 44.1 KHz and following the physicians examination, the sustained phonation of the vowel was collected from each subject with three repetitions.
Attribute Information: various speech signal processing algorithms including Time Frequency 
Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform based Features, Vocal 
Fold Features and TWQT features have been applied to the speech recordings of Parkinson's Disease 
(PD), patients to extract clinically useful information for PD assessment.
Data cleaning, data visualization and preprocessing
Once again, we inspect the entire dataset and look to see if there are any null or missing values to be 
deleted or replaced, but the structure of the dataset is complete. However, by means of the pie chart, 
we can also visually notice the disproportionality between the two different output classes, which 
leads us to some reflections. The model may have a tendency to mainly predict the majority class, 
while ignoring the minority class. This may lead to a low ability of the model to correctly identify 
examples of the minority class. In the presence of a dominant majority class, the model may be prone 
to overfitting this class, especially if the amount of training data is limited. This may lead to poor 
generalisation of the model to new data. Traditional evaluation metrics such as accuracy may also be 
misleading in the presence of imbalance, as a model that always predicts the majority class may still 
achieve high accuracy.
Variables selection and scaling 
To solve the above problem, we try to re-balance the dataset, using SMOTE: an acronym for Synthetic 
Minority Over-sampling Technique, it is a technique used to solve the problem of imbalance between 
classes in a machine learning dataset. This problem occurs when one of the classes, called the minority 
class, is underrepresented compared to the other class, called the majority class. SMOTE is designed 
to address this imbalance by increasing the number of examples in the minority class through the 
generation of synthetic data. The SMOTE process begins by randomly selecting examples from the 
minority class and identifying their 'neighbours' within the same class. These neighbours are chosen 
based on distance measures between the examples, such as Euclidean distance. Once the neighbours 
are selected, SMOTE generates new synthetic examples for each example in the minority class 
through a linear combination of the original example and one or more of its neighbours. This process 
of generating synthetic data is repeated for many examples in the minority class, thus increasing the 
number of examples in this class. The main objective of SMOTE is to improve the ability of the 
machine learning model to learn correctly from the minority class, enabling the model to make more 
accurate predictions for this class. This is particularly important in scenarios where the minority class 
is of special interest or has a significant impact, such as in rare event detection or medical data 
processing.
Once this is done, we can proceed with the scaling of the variables, and this time we use the MinMax Scaler: this is a normalisation technique commonly used in the field of machine learning to 
transform the features of a dataset into a specific range, usually between 0 and 1. This normalisation 
process is useful when the features of the dataset have very different scales and when we want all the 
features to have a similar footprint when training a machine learning model. The first stage of the 
process involves the selection of features to be normalised. These features represent the input 
variables that will be used to train the model. Once the features have been selected, the Min-Max 
Scaler calculates the minimum and maximum value for each of them within the entire dataset. The 
next step is the actual normalisation. For each value in each of the selected features, the Min-Max 
Scaler applies a mathematical transformation. This transformation rescales the values so that each 
original value falls in the range between 0 and 1. In other words, the minimum value becomes 0 and 
the maximum value becomes 1, while all other values are scaled accordingly in a proportional manner. 
The end result of the Min-Max normalisation is a new dataset in which all features have been 
transformed so that they have values between 0 and 1. This new dataset is now ready to be used to 
train machine learning models. Finally, we split the data into traing and test sets, with a ratio of 3/1.
Comparison of the models and Model Evaluation
The algorithms used are the same as those first mentioned and then described, so let's see which one 
performed best, in our case the Random Forest, with the values shown in the following image:
Hyperparameter tuning
Let us try to increase the performance of our model, this time using the Random Search technique: 
this is an approach based on a random search for combinations of hyperparameters. Instead of 
examining all combinations, the Random Search randomly selects a certain number of hyperparameter combinations from the distribution specified for each hyper-parameter. This random 
process allows a wide range of combinations to be explored more efficiently than Grid Search. In the 
end, it returns the combination of hyper-parameters that performed best according to the specified 
metric. Random Search is particularly useful when one has a limited computational budget or when 
the number of possible combinations is very large.
Final considerations
A Parkinson's disease detection model with an accuracy of between 88% and 90% can be considered 
a good quality model, but there are a few considerations to keep in mind: carefully examine the 
important features identified by the model to understand which attributes most influence the 
prediction of Parkinson's disease. This can be useful in uncovering relevant medical information. A 
model with 90% accuracy can make mistakes. We should share the results with medical professionals 
and subject the model to further clinical testing and evaluation before using it for diagnostic or 
decision-making purposes in a medical context. Random Forest is known for its high predictive 
accuracy. It combines predictions from multiple decision trees, thereby reducing the risk of overfitting 
and improving performance compared to a single decision tree. It can also handle noisy or missing 
data without requiring excessive data preprocessing. This robustness makes it suitable for a wide 
range of applications. Decision trees within a Random Forest can be trained in parallel, making it 
suitable for implementations on multiprocessor or distributed hardware. On the other hand, training 
a Random Forest can be computationally expensive, especially with large amounts of data and a large 
number of trees. This can require significant resources in terms of time and computing power. Due to 
the complexity of the model, Random Forest can be less interpretable than simpler models such as a 
single decision tree. It is difficult to understand how each individual decision tree contributes to the 
overall predictions. In general then, even if the Random Forest is designed to mitigate overfitting, 
with an excessive number of trees, it is possible that the model can still suffer from this problem, 
especially if adequate parameters are not set. As a final point against it, it must be said that Random 
Forest can require a significant amount of memory to store all the trees and associated information, 
if you use a large number of trees or have many variables.


BREAST CANCER DETECTION MODEL 
Dataset
The Wisconsin Breast Cancer (Diagnosis) dataset, commonly known as the WDBC dataset, is a wellknown and widely used dataset in the field of machine learning and data analytics. It was originally collected and compiled by Dr. William H. Wolberg of the University of Wisconsin Hospital in Madison, USA. The dataset is available through the UCI Machine Learning repository. The main goal 
of the WDBC dataset is to support the classification of breast cancer tumors into two types: malignant 
(cancerous) and benign (non-cancerous). This classification is based on various features extracted 
from digital fine needle aspiration (FNA) images of large breast lesions. These features are calculated 
from cell nuclei present in the image and are designed to capture different aspects of cell 
characteristics. The dataset includes 30 features calculated from each cell nucleus, including attributes 
such as radius, texture, circumference, surface area, smoothness, compactness, concavity, symmetry 
and fractal dimension. For each feature, the dataset provides three values: mean value, standard error, 
and worst (maximum) value. This gave a total of 10 features measured three times, resulting in a 
total of 30 features. In terms of the structure of the dataset, it contains a total of 569 cases, each 
corresponding to a breast lesion. Among these cases: 212 were labeled as malignant (1), indicating a 
cancerous tumor, 357 were labeled as benign (0), indicating the tumor was not cancerous. The WDBC 
dataset is commonly used for tasks such as binary classification, where machine learning algorithms 
are trained to distinguish between malignant and benign tumors based on the provided features. 
Researchers and data scientists often use this dataset to test different classification algorithms, 
evaluate their performance, and explore feature selection and engineering techniques.
Data cleaning, data visualization and preprocessing
We import the dataset, search for any null values and since they are not present, we proceed with the 
analysis. From the correlation analysis, we are able to say that maybe collinearity is present: 
collinearity, or high correlation between independent variables, may make it difficult to identify each 
variable's unique contribution to predicting the target. In other words, when many independent 
variables are highly correlated, it can be difficult to determine which of them significantly influences 
the output variable. The high correlation between the independent variables could also be the result 
of noise in the data. In other words, there may be random correlations between variables that have no 
true causal relationship with each other or with the output variable. Moving forward, through data 
visualization, we can further analyze our variables: the average values of cell radius, perimeter, area, 
compaction, concavity, and depression can be used in cancer classification. Higher values of these 
parameters tend to show correlation with malignancies. Mean values of texture, smoothness, 
symmetry, or crack size do not indicate any particular preference of one diagnosis over the other. 
None of the graphs had any notable large outliers that needed further cleaning. 
Variables selection and scaling 
However, we decide to include all the variables in the creation of our model, once again using a 
standard scaler.
Model Implementation
We choose to operate via an "ANN". This is an acronym that stands for "Artificial Neural Network" 
in English, translated into Italian as "Artificial Neural Network" (ANN). An ANN is a computational 
model based on the structure and functioning of neurons in the human brain. It is a type of neural 
network used in machine learning and artificial intelligence for automatic learning from data. An 
ANN is composed of a set of artificial units called "neurons" or "nodes" connected to each other 
through connections with weights. These weights determine the importance of connections between 
neurons and are adjusted during the training process to allow the network to learn from the data. I our 
case: an empty sequential model is created. A sequential model is a linear stack of neural network 
layers (layers), one on top of the other, in which data flows through the model from left to right. Then 
you add the first hidden layer to the network. This layer has 30 neurons (units) and receives input 
data with a shape that indicates there are 30 features in each input sample. The layer uses the ReLU 
(Rectified Linear Activation) activation function, which is common in classification problems. This 
layer is also the input layer. Then we add a dropout layer after the first hidden layer. Dropout is a 
regularization technique that helps prevent overfitting. The value 0.2 indicates that 20% of the units 
in this layer will be randomly deactivated during training in each step. This helps make the network 
more robust. Yet a second hidden layer with 20 neurons and ReLU activation function. This layer is 
designed to capture additional patterns in the data. Then another dropout layer is added after the 
second hidden layer with the same 20% dropout rate. This layer also helps prevent overfitting. We 
continue by adding a third hidden layer with 20 neurons and ReLU activation function. This additional 
layer can help capture additional relationships in the data, making the network deeper. We then arrive 
at the output layer with 1 neuron. The sigmoid activation function is used because it is a binary 
classification problem, and the sigmoid returns a prediction between 0 and 1, which can be interpreted 
as the probability of belonging to one of the two classes (0 or 1). Finally, we define how the model 
must be trained. We use the “Adam” optimizer, the “binary_crossentropy” loss function (suitable for 
binary classification) and measures the accuracy metric during training.
Model evaluation
The confusion matrix is a fundamental tool in machine learning and the analysis of the performance 
of a classification model. It is used to evaluate how well a model is able to classify observations in a 
classification problem compared to real labels or "ground truth". The confusion matrix is represented 
as a table with rows and columns, where the rows represent the real classes of the observations and 
the columns represent the classes predicted by the model. Within this table, there are four main boxes:
True Positives (TP): These are cases where the model correctly predicted a class as positive, and this 
prediction was indeed correct. In other words, they are the observations that have been correctly 
classified as belonging to the class of interest; True Negatives (TN): These are cases where the model 
correctly predicted a class as negative, and this prediction was actually correct. These are observations 
that have been correctly classified as not belonging to the class of interest; False Positives (FP): These 
are cases where the model incorrectly predicted a class as positive when it actually belonged to a 
negative class. These are also called "Type I errors"; False Negatives (FN): These are cases where the 
model incorrectly predicted a class as negative when it actually belonged to a positive class. These 
are also called "Type II errors". In our case, there are 86 cases where the model correctly predicted 
the positive class (TP). This indicates that the model correctly classified 86 cases as belonging to the
positive class, and these predictions are indeed correct. There are 53 cases where the model correctly 
predicted the negative class (TN). This indicates that the model correctly classified 53 cases as 
belonging to the negative class, and these predictions are indeed correct. There are 3 cases where the 
model incorrectly predicted the positive class when in reality they belonged to the negative class (FP). 
These are errors in which the model "wrong" to classify 3 cases as positive when they were not. There 
is then only 1 case in which the model erroneously predicted the negative class when in reality it 
belonged to the positive class (False Negatives, FN ). This indicates that the model "wrong" in 
classifying 1 case as negative when it was actually positive. Based on these values, it appears that the 
model has very good performance, with a high number of TP and TN and a low number of FP and 
FN errors. However, it is important to consider the context of the problem and the implications of 
classification errors to evaluate whether this performance meets the specific needs of the application. 
As further proof of the success of our model, we evaluated the accuracy at around 97%, another good 
sign for the work done.
Final considerations
An artificial neural network (ANN) model with an accuracy of 97% is generally considered very good 
and suggests that the model has a significant ability to make correct predictions based on the training 
data. However, to fully evaluate the performance of the model, it was also very important to also 
examine the confusion matrix and other evaluation metrics. ANNs in general are able to learn from 
data and improve their performance as they are exposed to more training data. This makes them very 
flexible for a wide range of tasks. They can be trained on parallel hardware, such as GPUs, to speed 
up the learning process, and they can also handle noisy and incomplete data, making them suitable 
for many real-world situations. However, training deep neural networks on large datasets can require 
significant computational resources and time. ANNs can be sensitive to overfitting, that is, they can 
overfit to training data and lose the ability to generalize to new data. They are also often considered 
“black boxes,” meaning they can be difficult to interpret and explain how they make decisions. 
Designing a neural network requires choosing numerous hyperparameters, such as the number of 
layers and units, which can be a complicated process. Finally we also say that they require a 
significant amount of training data to learn effective models, which could be a problem in situations 
where data is limited.


HEART DISEASE DETECTION MODEL
Dataset
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and 
Long Beach V. It contains 76 attributes, including the predicted attribute, but all published 
experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart 
disease in the patient. It is integer valued 0 = no disease and 1 = disease.
Attribute Information:
1. Age
2. sex
3. Chest pain type (4 values)
4. Resting blood pressure
5. Serum cholestoral in mg/dl
6. Fasting blood sugar > 120 mg/dl
7. Resting electrocardiographic results (values 0,1,2)
8. Maximum heart rate achieved
9. Exercise induced angina
10. Oldpeak = ST depression induced by exercise relative to rest
11. The slope of the peak exercise ST segment
12. Number of major vessels (0-3) colored by flourosopy
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
The names and social security numbers of the patients were recently removed from the 
database, replaced with dummy values.
Data cleaning, data visualization and preprocessing
After importing the dataset, we operate with the usual routine to check it and the information it 
reports, we notice, however, that some null values are present. We can either choose to replace those 
values with new ones, using various techniques, or we can decide to delete them. We delete them as 
we only go from 303 observations to 297, still a substantial number. We then notice the presence of 
three categorical variables ("Chest Pain", "Thal" and also our output variable "AHD") which we have 
to transform in order to perform the model. We rely on a type of encoding called LabelEncoder: it is 
a technique used in data preprocessing to convert categorical variables into numerical values. This 
process is crucial when working with machine learning algorithms that require numerical inputs. In 
practice, each unique category present in the categorical variable is associated with a unique 
numerical value. For example, if we are processing a 'Colour' variable with categories such as 'Red', 
'Green' and 'Blue', label encoding could assign them sequential numerical values such as 0, 1 and 2. 
An important aspect to bear in mind is that label encoding implicitly assumes a certain degree of 
ordinality between the categories. This means that the algorithm may misinterpret that a category 
with a higher numerical value has a higher 'value' than categories with lower numerical values. This 
can lead to distorted results if the categorical variable has no real ordering meaning. In the end, 
through data visualisation we appreciate the male-female diversification in diagnosis and the 
correlation matrix, which again does not denote much correlation between the output variable and the 
other variables. 
Variables selection and scaling 
As before, we choose 'AHD' as the variable output and apply the standard scaler, as we did for some 
models before.
Comparison of the models and choice of the best performing 
All that remains is to run the chosen models and evaluate their performance. Let’s see which models 
we decided to use and briefly the characteristics of each one:
ADABOOST: Its effectiveness is derived from an adaptive learning process that draws inspiration 
from the ability to learn from mistakes made in the past. AdaBoost starts by training an initial weak 
model on the training data. This weak model is generally a relatively simple classifier, like a decision 
tree with limited depth. During the training process, AdaBoost initially assigns each training example 
the same weight. However, examples that are misclassified by the weak model receive a higher 
weight, while those classified correctly receive a lower weight. This allows AdaBoost to focus more 
on difficult-to-classify points. At the end of each iteration, the weak models are combined to create a 
strong model. This combination is done by assigning a weight to each weak model based on its ability 
to correctly classify examples. More accurate models receive higher weights in the final combination. 
Once all the iterations are completed, you get a strong model which is the combination of the weak 
models. This final model is capable of providing more accurate predictions than individual weak 
models and has been shown to have great generalization ability;
QUADRATIC DISCRIMINANT ANALYSIS (QDA): is a classification algorithm that stands out for 
its generative model-based approach. This means that, rather than directly trying to learn the decision 
boundaries between classes, QDA tries to model the probabilistic distributions of the data in each 
class. One of the distinguishing features of QDA is its flexibility in addressing assumptions about the 
shape of data distributions. QDA assumes that the distributions of data features within each class are 
Gaussian distributions. This is a rather flexible assumption, as Gaussian distributions can represent a 
wide range of data shapes. To obtain the best description of the class probability distributions, QDA 
estimates the covariance matrices for each class. These matrices reflect how different data 
characteristics are related to each other within each class. This information is crucial for calculating 
the probability density functions of the classes. When it comes to classifying new data, QDA estimates 
the probability of membership in each class using previously calculated probability density functions. 
Then, assign the class label to the new data based on the class with the highest probability. QDA 
proves to be particularly effective when data distributions have non-linear shapes or when the 
variances of different classes are significantly different. In situations where classes have different 
shapes and dispersions, QDA can adapt better than other linear classification algorithms such as 
Logistic Regression. Its ability to flexibly model class distributions makes it a useful tool in a number 
of machine learning and data analytics contexts;
LINEAR DISCRIMINANT ANALYSIS (LDA): unlike QDA, assumes that all classes have the same 
covariance matrix. This means that LDA assumes that feature distributions are similar across different 
classes. This is a pretty strong assumption, but it can be useful in many situations. The algorithm uses 
probability theory to estimate the probabilities of membership in each class. In other words, it tries to 
calculate how likely a given data point is to belong to each class. To classify new data, LDA calculates 
the probability of membership in each class and assigns the data point label to the class with the 
highest probability. LDA is often used when classes are suspected to be linearly separable in the data. 
This means that classes can be distinguished by linear hyperplanes, making LDA an appropriate 
choice for problems where separations between classes are well delineated. However, when data 
distributions are not Gaussian or when classes have very different variances, LDA may not be the 
best choice. In summary, LDA is another classification technique based on generative models. Its 
strength lies in its ability to handle situations where classes are linearly separable in the data, but it 
may not be suitable for problems with complex class distributions or widely varying variances. The 
choice between QDA and LDA often depends on the specific characteristics of the data and the 
objective of the analysis;
GRADIENT BOOSTING: it is a powerful machine learning technique used to tackle regression and 
classification problems. It is based on an ensemble approach, which combines several weak models 
(usually decision trees) to create a stronger and more accurate model. The main idea behind Gradient 
Boosting is to iteratively train new models that attempt to correct errors made by previous models, 
focusing on the most difficult cases. The process of Gradient Boosting is driven by the minimisation 
of the loss function, which represents the difference between model predictions and actual values. In 
each iteration, a new model is trained to capture the residual errors of the previous model. This new 
model is then added to the ensemble in order to give more weight to the cases where the previous 
model made significant errors.The key to Gradient Boosting is the concept of a 'gradient', which 
represents the direction and magnitude in which successive models should be fitted to reduce the 
overall error. This gradient-driven approach allows new models to focus on the remaining errors, 
progressively improving the accuracy of the ensemble. One of the best-known algorithms based on 
Gradient Boosting is XGBoost (Extreme Gradient Boosting), which further optimises the process by 
introducing advanced regularisation and optimisation.This makes XGBoost a very popular tool in 
various machine learning competitions and real-world applications. The use of Gradient Boosting is 
widespread in various fields, including medicine. For example, in the field of medical diagnostics, 
Gradient Boosting-based models can be used to combine information from different sources, such as 
clinical data, medical images and genomic data, for a more accurate diagnosis. In addition, Gradient 
Boosting can be applied to predicting the risk of certain medical conditions or personalising 
treatments based on individual data; 
EXTRA TREES: also known as Extremely Randomized Trees, represent a fascinating variant of 
random decision trees (Random Forests) that has proven to offer significant advantages in various 
machine learning contexts. The defining feature of Extra Trees is their "extraordinary randomness" 
in training individual decision trees. This additional randomness over Random Forests can lead to 
significant improvements in model performance. In Random Forests, each tree is trained on a random 
subset of the training data and a limited selection of the best discriminant functions is made for each 
node. This is a process that tries to find a balance between randomness and information to avoid 
overfitting. However, in Extra Trees, this randomness is taken even further. When you train an Extra 
Trees tree, you share the same training dataset, but instead of carefully selecting the best discriminant 
functions for each node, you randomly select the discriminant functions for each node. This means 
that the resulting trees are "extra random" in the sense that their divisions are based on random 
choices. This extraordinary randomness has important advantages. First, it makes Extra Trees very 
diverse from each other, which is useful because highly diverse models tend to improve generalization 
ability, reducing the risk of overfitting to the training data. Additionally, this additional randomness 
makes them less sensitive to noise in the data and uninformative features. Extra Trees are particularly 
useful when the dataset contains noisy data or when some of the discriminant functions are not 
particularly informative. However, it is important to note that because of this extraordinary 
randomness, Extra Trees may be less interpretable than standard decision trees or Random Forests. 
Because the divisions are based on random choices, the logic behind each division may be less clear 
and interpretable;
THE MLP CLASSIFIER: short for Multi-Layer Perceptron Classifier, is a machine learning 
algorithm that belongs to the family of artificial neural networks (ANN), inspired by the functioning 
of the human brain. This type of network, known as a "Multilayer Perceptron," is characterized by 
the presence of one or more hidden layers between the input layer and the output layer. Each of these 
layers contains a set of artificial neurons, or nodes, and each node is connected to those in adjacent 
layers through weighted connections. The main goal of the MLP Classifier is to learn from labeled 
data, i.e. data where you already know the desired output. During the training process, the algorithm 
uses advanced techniques, such as error backpropagation, to update the weights of connections 
between neurons so that the neural network can make accurate predictions. MLP's flexible 
architecture allows you to model complex relationships between input variables and output labels, 
making it suitable for a wide range of applications, including natural language processing, computer 
vision, and many other fields. However, it is important to note that training an MLP requires careful 
design of the network architecture, including choosing the number of hidden layers and neurons in 
each layer, as well as tuning parameters, such as learning rates . Furthermore, on very large datasets, 
training an MLP can require significant computational resources;
CATBOOST: is a decision tree-based boosting algorithm that has proven to be a powerful tool in the 
field of machine learning, especially when dealing with data that contains categorical variables. What 
makes CatBoost remarkable is its ability to directly handle these categorical variables without the 
need to convert them to numerical representations or apply other data preparation techniques. This is 
a significant advantage, as handling categorical variables can be a complicated task in many other 
machine learning techniques. The "ordered boosting" technique is one of the most distinctive and 
powerful aspects of CatBoost. This method adds weak estimators sequentially during the model 
training process. This means that CatBoost focuses on more difficult cases or mistakes made in 
previous iterations, helping to progressively improve the model's performance. This feature is critical 
for capturing complex and subtle relationships in data, which is especially useful in situations where 
dependencies are intricate. Another point in favor of CatBoost is its ease of use. Many times, users 
can achieve satisfactory results using the default settings, thus reducing the need for extensive 
parameter tuning. This is especially important for those who want an effective solution without having 
to invest excessive time and effort in experimenting with parameters.
Model evaluation 
We observe how QDA is the best performing algorithm, via the classification report. As an additional 
means of validation, we draw the Roc-Auc (Receiver Operating Characteristic Area Under the Curve) 
Curve. This is a metric used to evaluate the performance of a machine learning classification model. 
This measure is particularly useful in binary classification problems, where the model must 
distinguish between two classes, such as positive and negative. To calculate the ROC AUC, the model 
must be able to estimate the probabilities that observations belong to classes. These probabilities 
represent the model's confidence in its predictions. The ROC curve is created by varying the 
probability threshold at which the model classifies observations into one of two classes. The ROC 
curve is a graph that shows the true positive rate (TPR) on the ordinate and the false positive rate 
(FPR) on the abscissa. Ideally, we want the model to have a high TPR (close to 1) and a low FPR 
(close to 0), which indicates that it can distinguish between classes perfectly. The ROC AUC 
represents the area under the ROC curve. This area ranges from 0 to 1, where a value of 1 indicates 
perfect classification and a value of 0.5 indicates random classification. In summary, ROC AUC is a 
metric that measures the discrimination ability of a classification model. The larger the area under the 
ROC curve, the better the model performs in distinguishing between classes. It is especially useful 
when you have unbalanced classes or when you want to examine model performance at different 
probability thresholds.
Final considerations
88% accuracy is a general indicator of model performance, but recall is also important to consider. A 
recall of 88% suggests that the model is correctly identifying 88% of the positive cases (heart disease) 
in the dataset. For example, if the model is used as a screening tool, high recall could be prioritized 
to identify the majority of patients with heart disease, even at the cost of an increase in false positives. 
Although 88% accuracy and recall are good, you can always try to further improve the model's
performance. This could involve hyperparameter optimization, feature engineering, or the use of 
ensemble algorithms. Overall, obtaining a model with high accuracy and recall is a positive outcome, 
but it is important to carefully consider the context and specific needs of the clinical application to 
evaluate whether the model is adequate for its intended purpose. Unlike Linear Discriminant Analysis 
(LDA), QDA is more flexible in modeling the relationships between independent variables and output 
classes. This means it can capture nonlinear decision boundaries between classes. QDA provides a 
quadratic decision boundary between classes, which makes it more interpretable than more complex 
models such as neural networks or Support Vector Machines (SVMs). The decision boundary can be 
easily visualized even in two-dimensional spaces. It also does not require the assumption of 
homoscedasticity: Unlike LDA, QDA does not require the assumption of homoscedasticity, making 
it more suitable for data in which the class variances are significantly different from each other. 
However, due to its flexibility in modeling nonlinear decision boundaries, QDA can be susceptible to 
overfitting, especially when you have a limited amount of training data. This can lead to poor 
generalization on unseen data. Estimating a separate covariance matrix for each class is 
computationally expensive, especially when there are many variables in the dataset. This can slow 
down model training on large dataset
