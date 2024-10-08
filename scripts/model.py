#importing the necessary packages
import pandas as pd 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score , roc_auc_score  , confusion_matrix
import pickle
#--------------------------------------------------------------------------------------------

#  --------------------------READING THE DATASETS--------------------------------------------

pd.reset_option('display.max_columns')
x_train = pd.read_csv('../data/X_Train_Data_Input.csv')
y_train = pd.read_csv('../data/Y_Train_Data_Target.csv')
x_test = pd.read_csv('../data/X_Test_Data_Input.csv')
y_test = pd.read_csv('../data/Y_Test_Data_Target.csv')

#--------------------------------------------------------------------------------------------
# Data Cleaning
#--------------------------------------------------------------------------------------------

# Imputation of missing values

# Extreme Skewness :**  Columns - 14 , 5 , 15 , 6 -> Use median imputation
# Slight skewness :**   Columns - 3 , 4 , 0 , 8 . -> Use mean imputation

class imputation:
    
    """
        Class for handling missing value imputation.
    """
    
    """
        Drop unnecessary columns.
    """
    x_train.drop('Column9' , axis = 1 , inplace=True)
    x_test.drop('Column9' , axis = 1, inplace=True)
    x_train.drop('ID' , axis = 1 , inplace= True)
    x_test.drop('ID', axis= 1 , inplace= True)
    y_train.drop('ID' , axis = 1 , inplace =True)
    y_test.drop('ID' , axis = 1 , inplace= True)
    
    # Seperating the columns based on its distribution for imputations
    median_cols = ['Column14' , 'Column5' , 'Column15' , 'Column6'] # Has Extreme Skewness
    mean_cols = ['Column3' , 'Column4' , 'Column0' , 'Column8'] # Has Slight Skewness
    
    #impute mean values for the missing data 
    def imp_mean(self , x):
        """
            Impute mean values for the missing data.
        """
        for col in self.mean_cols:
            x[col] = SimpleImputer(strategy='mean').fit_transform(x[[col]])
        return x 
        
    #impute median values for the missing data 
    def imp_median(self ,x):
        """
            Impute median values for the missing data.
        """
        for col in self.median_cols:
            x[col] = SimpleImputer(strategy='median').fit_transform(x[[col]])
        return x
        
imp = imputation()
x_train = imp.imp_mean( x = x_train)
x_train = imp.imp_median(x = x_train)
x_test = imp.imp_mean( x = x_test)
x_test = imp.imp_median(x = x_test)

#--------------------------------------------------------------------------------------------
# Feature Scaling
#--------------------------------------------------------------------------------------------
# Here none of the columns follow normal distribution .so we will be using normalization technique
# Columns - 14 , 10 , 6 , 7 , 0 , 3 have outliers . So we will be using robust scaling technique

class DataScaler:
    """
        Class for scaling data using RobustScaler and MinMaxScaler.
    """
    def __init__(self, robust_columns):
        self.robust_columns = robust_columns
        self.rs = RobustScaler()
        self.mm = MinMaxScaler()
    
    def fit_transform(self, x_train, x_test):
        """
            Fit and transform the training and test data.
        """
        normalize = [i for i in x_train.columns if i not in self.robust_columns]
        
        for col in self.robust_columns:
            x_train[col] = self.rs.fit_transform(x_train[[col]])
            x_test[col] = self.rs.transform(x_test[[col]])
        
        for col in normalize:
            x_train[col] = self.mm.fit_transform(x_train[[col]])
            x_test[col] = self.mm.transform(x_test[[col]])
        
        return x_train, x_test
    
# define robust columns and apply scaling 
robust_columns = ['Column14', 'Column10', 'Column6', 'Column7', 'Column0', 'Column3']
scaler = DataScaler(robust_columns)
x_train_scaled, x_test_scaled = scaler.fit_transform(x_train, x_test)

#--------------------------------------------------------------------------------------------
# Model Building
#--------------------------------------------------------------------------------------------

# According to Forward Feature Selection we will be going with the following columns for model training 
# Column18, Column7, Column1, Column3, Column5, Column0, Column15, Column6, Column10, Column8, Column13, Column16
# After checking its feature importance and removing features having negligible importance 

fs  = ['Column18','Column7','Column1','Column3']

# We will be using ADA-Boost with hyperparameters obtained from RandomSearchCV.
# Parameters : {'n_estimators': 100, 'learning_rate': 1, 'algorithm': 'SAMME.R'}

class model_training:
    """
        Class for training and evaluating the AdaBoost model.
    """
    def __init__(self):
        self.ada = AdaBoostClassifier(n_estimators=100, learning_rate=1, algorithm='SAMME.R')
        
    def ada_model(self ,x_train , x_test , y_train , y_test):
        """
            Train and evaluate the AdaBoost model.
        """
        self.ada.fit(x_train[fs] ,y_train.values.reshape(-1))
        y_test_pred = self.ada.predict(x_test[fs])
        
        # Test set performance
        model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
        model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
        model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
        model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
        model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc auc score
        model_conf_matrix = confusion_matrix(y_test , y_test_pred) #Calculate confusion matrix
        
        print('- Accuracy: {:.4f}'.format(model_test_accuracy))
        print('- F1 score: {:.4f}'.format(model_test_f1))
        print('- Precision: {:.4f}'.format(model_test_precision))
        print('- Recall: {:.4f}'.format(model_test_recall))
        print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
        print('-'*35)
        print('Confusion Matrix:')
        print(model_conf_matrix)
        print('-'*35)
        print('\n')

# Instantiate and train the model 
model = model_training()
model.ada_model(x_train , x_test , y_train , y_test)

#Importing the model as PICKLE file 
filename = 'model.sav'
pickle.dump(model , open(filename , 'wb'))