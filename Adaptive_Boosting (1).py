#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import function to generate synthetic classification datasets
from sklearn.datasets import make_classification
# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import evaluation metrics for classification performance
from sklearn.metrics import accuracy_score, classification_report
# Import Decision Tree classifier algorithm
from sklearn.tree import DecisionTreeClassifier


# In[14]:


# Generate a synthetic classification dataset with specified parameters
X, y = make_classification(
    n_samples = 500,        # Total number of samples to generate
    n_features = 20,        # Total number of features in the dataset
    n_informative = 10,     # Number of informative features (useful for classification)
    n_redundant = 2,        # Number of redundant features (linear combinations of informative features)
    random_state = 42       # Random seed for reproducible results
)

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(
    X, y,test_size = 0.3,    # Reserve 30% of data for testing, 70% for training
    random_state = 42       # Random seed for reproducible train/test splits
)


# In[15]:


# Import AdaBoost classifier from scikit-learn
from sklearn.ensemble import AdaBoostClassifier

# Create a weak learner (decision stump) as the base estimator
# max_depth=1 creates a decision stump (tree with only one split)
base_model = DecisionTreeClassifier(
    max_depth = 1, 
)

# Initialize AdaBoost classifier with the base estimator
abc = AdaBoostClassifier(
    estimator = base_model,        # Use decision stump as weak learner
    n_estimators = 100,            # Number of boosting iterations
    random_state = 42              # Set seed for reproducible results
)


# In[20]:


# Train the ABC model using the training features and target labels
abc.fit(X_train,y_train)


# In[22]:


# Generate predictions on the test dataset using the trained ABC model
y_pred = abc.predict(X_test)


# In[24]:


# Calculate and display the accuracy score by comparing predicted values with actual test values
print("Accuracy: ",accuracy_score(y_pred,y_test))


# In[26]:


# Print a detailed classification report comparing predicted labels (y_pred) with actual test labels (y_test)
# This report includes precision, recall, F1-score, and support for each class
print("Classification Report: ",classification_report(y_pred,y_test))


# In[ ]:




