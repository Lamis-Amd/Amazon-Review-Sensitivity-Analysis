# Amazon-Review-Sensitivity-Analysis
This project applies three text vectorization techniques including Bag of Words, TF-DF and n-Grams, and develops sentiment analysis predictive models in Python using logistic regression, SVM, random forest classification algorithms.
The project covers four parts:

## 1. Data Preparation
* Parse the json file and read the raw data into a pandas dataframe.
* Plot the distribution of the product ratings and classify the overall rating into a binary variable called "sentiment". For reviews with overall ratings ranging from 1-3, we assign value of 0, representing negative sentiment. For reviews with overall ratings ranging from 4 to 5, we assign value of 1, representing positive sentiment.
* Split the data into training and testing set. Upsampling the data, since 80% of the reviews are positive (overall rating 4-5). The upsampling is only applied on training set.

## 2. Text Processing
* Handle the negation: 
Convert "n't" to "not" and attach any "not" in the text to the subsequent word e.g. "not_nextword"
* Turn all the characters into lower case
* Remove punctuation
* Remove stop words and stem all words using `PorterStemmer` in `nltk`

## 3. Text Vectorization
Three text vectorization techniques are applied to training and testing sets, including Bag of Words, TF-DF and n-Grams using `CountVectorizer` and `TfidfVectorizer` in  `sklearn`. As a result of this step, there are three pairs of training and testing sets. Each pair is the outcome of one text vectorization method.

## 4. Machine Learning
Three classification algorithms are employed: logistic regression, SVM and random forest. Each algorithm is implemented on one pair of training and testing at a time. There are 9 machine learning models estimated in total. For each model, `GridSearchCV` is utilized to find the best hyperparameters. Model performance is assessed by computing metrics including confusion matrix, precision, recall, F1 score and ROC.
