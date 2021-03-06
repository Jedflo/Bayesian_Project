from matplotlib import pyplot as plt
import pandas as pd

#scikit-learn is used to build models. It should not be used for reading the data, manipulating and summarizing it.
#Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for
# the Python programming language.[3] It features various classification, regression and clustering algorithms
# including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to
# interoperate with the Python numerical and scientific libraries NumPy and SciPy.
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

TARGET = "stroke";
ENCODING_EXCLUDED = ["age","avg_glucose_level","bmi",];

# Load the CSV data file using pandas read_csv method
stroke_dataset = pd.read_csv("healthcare-dataset-stroke-data-clean.csv")
print(stroke_dataset)   #1. Print the original table
headers = stroke_dataset.columns

# LabelEncoder() converts a categorical data into a number ranging from 0 to n-1
# where n is the number of classes in the variable.
number = LabelEncoder()

for header in headers:
    print(header);
    if header not in ENCODING_EXCLUDED:
        stroke_dataset[header] = number.fit_transform(stroke_dataset[header])

smote = SMOTE()
test_df  = stroke_dataset[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']].sample(int(stroke_dataset.shape[0]*0.2),random_state=42)
train_df = stroke_dataset.drop(index=test_df.index)

X_test, y_test   = test_df[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']], test_df['stroke']
X_train, y_train = train_df[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status',]], train_df['stroke']


X_train, y_train = smote.fit_resample(X_train, y_train)
upsampled_df = X_train.assign(Stroke = y_train)

X_test, y_test = smote.fit_resample(X_test, y_test)
up_test_df = X_test.assign(Stroke = y_test)

fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(15, 5))

stroke_dataset.stroke.value_counts().plot(kind='bar', color='tab:red', title='Stroke - Before Upsampling', ax=axes[0])
upsampled_df.Stroke.value_counts().plot(kind='bar', color='tab:green', title='Stroke(Train set) - After Upsampling', ax=axes[1])
up_test_df.Stroke.value_counts().plot(kind='bar', color='tab:blue', title='Stroke (Test set) - After Upsampling', ax=axes[2]);

plt.show()


print(stroke_dataset)  #2 Print the converted table after transformation

features = headers.drop(TARGET);
target = TARGET;


# Create a train, test split. We build the model using the train dataset
# and we will validate the model on the test dataset
#if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21,
#  The important thing is that everytime you use 42, you will always get the same output the first time you make the split.
#Data Slicing Let???s split the data into training and test set. We can easily perform this step using sklearn???s train_test_split() method.
features_train, features_test, target_train, target_test = train_test_split(stroke_dataset[features],
stroke_dataset[target], test_size = 0.30,
   random_state = 40)

# Displaying the split datasets
print('\tTraining Features\n ',features_train)  #3 Print all of these
print('\tTesting Features\n ',features_test)
print('\tTraining Target\n ',target_train)
print('\tTesting Target\n ',target_test)
#Gaussian Naive Bayes Implementation .After completing the data preprocessing. it???s time to implement machine learning
# algorithm on it. We are going to use sklearn???s GaussianNB module.
# Creating a Gaussian Naive Bayes model
model = GaussianNB()
#We have built a GaussianNB classifier. The classifier is trained using training data.
# We can use fit() method for training it. After building a classifier, our model is ready to make predictions.
# We can use predict() method with test set features as its parameters.
# Fitting the training dataset to the model
model.fit(X_train.values, y_train.values)

# After fitting, we will make predictions using the testing dataset
pred = model.predict(X_test.values)
#It???s time to test the quality of our model. We have made some predictions. Let???s compare
# the model???s prediction with actual target values for the test set. By following this method,
# we are going to calculate the accuracy of our model.
# measuring the accuracy of the model, using the actual data and the predicted results
accuracy = accuracy_score(y_test.values, pred)

# Displaying the accuracy of the model
print("\nModel Accuracy = ",accuracy*100,"%") #4 Print the percentage of accuracy

# Now suppose we want to predict for the conditions:
# Outlook = Rain (Rain is represented as 1 in the Outlook class)
# Temperature = Mild (Mild is represented as 2 in the Temperature class)
# Humidity = High (High is represented as 0 in the Humidity class)
# Wind = Weak (Weak is represented as 1 in the Wind class)
# Should we play (1) or not (0) ? According to our data set, given these features play should be 1
answer = model.predict([[
    1, # gender [Male = 1, Female = 0]
    67,# age
    0, # hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    1, # heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    1, # ever_married: [yes = 1, no = 0]
    2, # work_type: [Govt_job = 0, Never_worked = 1, Private = 2, Self-employed = 3, children = 4, ]
    0, # Residence_type: [Rural = 0  or Urban = 1]
    228.69,# avg_glucose_level: average glucose level in blood
    36.6,# bmi: body mass index
    2 # smoking_status: [formerly smoked = 1, never smoked = 2, smokes = 3, Unknown = 4]
    ]]) #5 Try to  model.predict other parameter values

if answer == 1:
    print("\nPatient had a stroke")
elif answer == 0:
    print("\nPatient had no stroke")



