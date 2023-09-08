#%%
# # Initital Dataset Cleaning and Manipulation 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rfit
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
# # %%
cwd = os.getcwd()

print(cwd)

dataset = pd.read_csv('Dataset/OnlineNewsPopularity.csv')
# # %%
# Basic checks

rfit.dfchk(dataset)

# 61 total attributes, 39644 rows

# No missing values/ No nulls

# %%
# Info on Dataset Features:
#
# Some of the attributes in the dataset have already been encoded for machine learning. However, we will decode it into a single column for visualization purposes. Such columns include: 
# 1. Data_Channel : Type of article (Entertainment, lifestyle, Media, Technology, World etc.)
# 2. Publish Day : Day the article was pubished (Monday, Tuesday, etc.)

# dataset['Data_Channel'] = np.where(dataset['data_channel_is_lifestyle']==1,'Lifestyle',np.where(dataset['data_channel_is_entertainment']==1,"Entertainment",np.where(dataset['data_channel_is_bus']==1,'Business',np.where(dataset['data_channel_is_socmed']==1,'Social Media',np.where(dataset['data_channel_is_tech']==1,'Technology','World')))))

# For some reason, above code was giving key-error. After further checking, i foudn out that several column titles in the dataset have leading or trailing empty spaces. Needed to fix this
#%%
# First identifying what columns have these extra spaces
bad_columns = [x for x in dataset.columns if x.endswith(' ') or x.startswith(' ')]
print('Number of Columns with unwanted spaces: ',len(bad_columns))

# Almost all columns have this problem, so we'll fix this
dataset.columns = dataset.columns.str.strip()

# Checking to see if the problem is resolved
bad_columns_validation = [x for x in dataset.columns if x.endswith(' ') or x.startswith(' ')]
print('\nAfter Fix:\nNumber of Columns still with issue: ',len(bad_columns_validation))

#%%
# Running the code again

dataset_viz = dataset.copy()

dataset_viz['Data_Channel'] = np.where(dataset_viz['data_channel_is_lifestyle']==1,'Lifestyle',np.where(dataset_viz['data_channel_is_entertainment']==1,"Entertainment",np.where(dataset_viz['data_channel_is_bus']==1,'Business',np.where(dataset_viz['data_channel_is_socmed']==1,'Social Media',np.where(dataset_viz['data_channel_is_tech']==1,'Technology','World')))))
# dataset.head()

#%%
# Now doing the same thing for Day of the Week
dataset_viz['Publish_DOW'] = np.where(dataset_viz['weekday_is_monday']==1,'Monday',np.where(dataset_viz['weekday_is_tuesday']==1,"Tuesday",np.where(dataset_viz['weekday_is_wednesday']==1,'Wednesday',np.where(dataset_viz['weekday_is_thursday']==1,'Thursday',np.where(dataset_viz['weekday_is_friday']==1,'Friday',np.where(dataset_viz['weekday_is_saturday'],'Saturday','Sunday'))))))
# dataset.head()

#%%
# We can go ahead and remove the columns that have been utilized
dataset_viz = dataset_viz.drop(['weekday_is_saturday','weekday_is_friday','weekday_is_sunday','weekday_is_thursday','weekday_is_wednesday','weekday_is_tuesday','weekday_is_monday','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world'],axis=1)
# dataset_viz.shape

#%%
# Saving out this dataset for collaboration
#dataset_viz.to_csv('Dataset/OnlineNewsPopularity_Viz.csv', index=False)

# We're going to use dataset_viz for visualizations and dataset for modeling

#%%
# Some of the features are dependent of particularities of the Mashable service (whose articles have been used as data source): articles often reference other articles published in the same service; and articles have meta-data, such as keywords, data channel type and total number of shares (when considering Facebook, Twitter, Google+, LinkedIn, Stumble-Upon and Pinterest). The minimum, average and maximum number of shares was determined of all Mashable links cited in the article were extracted to prepare the data. Similarly, rank of all article keyword average shares was determined, in order to get the worst, average and best keywords. For each of these keywords, the minimum, average and maximum number of shares was extracted as a feature. [Reference: Research Paper]

# Several features are extracted by performing natural language processing on the original articles. The Latent Dirichlet Allocation (LDA) algorithm was applied to all Mashable articles in order to first identify the five top relevant topics and then measure the closeness of current article to such topics. To compute the subjectivity and polarity sentiment analysis, Pattern web mining module was adopted, allowing the computation of sentiment polarity and subjectivity scores. These are such features:
# 1. Closeness to top 5 LDA topics 
# 2. Title subjectivity ratio 
# 3. Article text subjectivity score and its absolute difference to 0.5 
# 4. Title sentiment polarity 
# 5. Rate of positive and negative words 
# 6. Pos. words rate among non-neutral words 
# 7. Neg. words rate among non-neutral words 
# 8. Polarity of positive words (min./avg./max.) 
# 9. Polarity of negative words (min./avg./max.) 
# 10. Article text polarity score and its absolute difference to 0.5
# [Reference: Research Paper]

# Even though we don't yet understand what these variables represent exactly, we will keep them for the purpose of model building.

#%%
# Reading the csv file

sharedf = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
print(sharedf.head())

# %%
print(len(sharedf))

#There are total 39644 rows in the entire dataset

# %%
sharedf=sharedf.drop_duplicates()
print(sharedf.isna().sum())

# Any duplicates values in the data set are removed, and there are no 
#null values the data set.

# %%
sharedf.describe()

#%%
sharedf = sharedf[sharedf['n_tokens_title']!=0]

# %%
sharedf = sharedf[sharedf['n_tokens_content']!=0]

#The n_tokens_title and n_tokens_content columns which contains the 
# value 0 is removed

#%%
print(len(sharedf))
#After removing these 0 values, the length of the dataframe is 38463.
#1180 rows are removed and the dataset is stored.
 
#%%
print(sharedf['shares'].mean())
print(sharedf['shares'].median())

# The mean value of the shares is 3355.56, and the median value of the shares is 1400

# %%
#correlation between the columns
plt.figure(figsize=(15,15))

correlations = sharedf.corr()

print(correlations)

sns.heatmap(correlations, cmap="Blues")

# From the heat map(correlation plot) we can observe that n_non_stop_unique_tokens, n_non_stop_words, kw_avg_min

# has the high correlation

#%%
sharedf = sharedf.drop('url',axis=1)

#%%
#From the collerations we can observe that n_non_stop_words, n_non_stop_unique_tokens, kw_avg_min has high correlations, we are dropping these columns

sharedf= sharedf.drop(["n_non_stop_unique_tokens","n_non_stop_words","kw_avg_min"],axis=1)

# %%
print(sharedf.head())
# %%
plt.figure(figsize=(15,10))

sns.scatterplot( x='n_tokens_content', y='shares',hue="target", data=sharedf)
# %%
plt.figure(figsize=(15,10))

sns.scatterplot( x='n_tokens_title', y='shares',hue="target", data=sharedf)

# %%
group_1= pd.DataFrame(sharedf.groupby("Publish_DOW").mean()["shares"])

sns.barplot(x= group_1.index, y="shares", data=group_1)
# %%
group_2= pd.DataFrame(sharedf.groupby("Data_Channel").mean()["shares"])

sns.barplot(x= group_2.index, y="shares", data=group_2)
# %%
fig = plt.subplots(figsize=(10,10))


sns.scatterplot(x='avg_positive_polarity', y='shares', data=sharedf, hue="target", alpha=0.5)
# %%
fig = plt.subplots(figsize=(10,10))
sns.scatterplot(x='num_imgs', y='shares', hue="target",data=sharedf)

# %%
#pair plots between all the kw values

plt.figure(figsize=(30,30),dpi=200)

columnskw = ['kw_min_min', 'kw_max_min',  'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg', 'shares']
sns.pairplot(data = sharedf, vars=columnskw, hue="target",diag_kind="kde")
# %%
plt.figure(figsize=(15,10))
sns.scatterplot(y = "shares", x = "num_imgs",hue="target", data=sharedf)
plt.title("scatter plot between shares and number of images")

# %%
plt.figure(figsize=(15,10))
sns.scatterplot(y = "shares", x = "num_videos",hue="target", data=sharedf)
plt.title("scatter plot between shares and number of videos")

# %%
group_3= pd.DataFrame(sharedf.groupby("is_weekend").mean()["shares"])
sns.barplot(x= group_3.index, y="shares", data=group_3)
plt.title("bar plot for shares based on whether day is weekend or not")

# %%
group_4= pd.DataFrame(sharedf.groupby("is_weekend").count()['shares'])
print(group_4)
sns.barplot(x= group_4.index, y="shares", data=group_4)
plt.ylabel("count of the shares for weekend vs weekday")
plt.title("count of the shares between weekend or weekday")

# %%
#model building
#Decision Tree Regression
from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
X = pd.get_dummies(sharedf.drop('shares',axis=1),drop_first=True)

y = sharedf['shares']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
Scaled_Xtrain = scaler.fit_transform(X_train)
Scaled_Xtest= scaler.transform(X_test)





#%%

# %%
print(sharedf.columns)
# %%


# %%
# %%
threshold = sharedf.shares.median()
sharedf['target'] = np.where(sharedf.shares>threshold,1,0)
# %%
print(sharedf['target'])
# %%
## KNN Classification
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, plot_confusion_matrix
#%%
X = pd.get_dummies(sharedf.drop(['shares','target'],axis=1),drop_first=True)
y = sharedf['target']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
Scaled_Xtrain = scaler.fit_transform(X_train)
Scaled_Xtest= scaler.transform(X_test)
knn_model = KNeighborsClassifier(n_neighbors=14)
knn_model.fit(Scaled_Xtrain,y_train)
y_pred = knn_model.predict(Scaled_Xtest)
accuracy_score(y_test,y_pred)
cf_matrix = confusion_matrix(y_test,y_pred)
#%%
plot_confusion_matrix(knn_model, Scaled_Xtest, y_test)
#%%
print(classification_report(y_test,y_pred))
import seaborn as sns
sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')

# By using the KNN algorithm, we can observe that at k = 14 the accuracy of the model is 62%.
# 
# From the heat map, we can observe that 1724 rows which must be classified as the 0, the algorithm classified as 1.
#Moreover 3075 rows, which the algorithm should classify as 1, the algorithm predicted as 0.
# %%
#Choosing K value
test_error_rates = []


for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(Scaled_Xtrain,y_train) 
   
    y_pred_test = knn_model.predict(Scaled_Xtest)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)
# %%
plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")
# We can observe from the plot that the optium value for the k is 14, and increasing the 
# k values furthermore will result in the decrease of the error percentage from 0.38 to 0.37.
#%%
#ROC for KNN
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_scores=knn_model.predict_proba(Scaled_Xtest)
fpr,tpr,threshold =roc_curve(y_test,y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

# %%
print(len(y_pred_test))

# %%
# Logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import plot_roc_curve
log_model = LogisticRegression()
log_model.fit(Scaled_Xtrain,y_train)
y_pred = log_model.predict(Scaled_Xtest)
accuracy_score(y_test,y_pred)
cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
print(classification_report(y_test,y_pred))
plot_roc_curve(log_model,Scaled_Xtest,y_test)
#%%
#ROC for lm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_scores=log_model.predict_proba(Scaled_Xtest)
fpr,tpr,threshold =roc_curve(y_test,y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of logistic regression')
plt.show()


# %%
#Decision Tree classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

model = DecisionTreeClassifier(max_depth= 5)
model.fit(Scaled_Xtrain,y_train)
base_pred = model.predict(Scaled_Xtest)
from sklearn.metrics import confusion_matrix,classification_report
cf_matrix = confusion_matrix(y_test,base_pred)
sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
print(classification_report(y_test,base_pred))
print(model.feature_importances_)
pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance'])
plt.figure(figsize=(12,8),dpi=150)
plot_tree(model,filled=True,feature_names=X.columns);
# %%
dff = pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(dff.head(7))
#%%
for depth in range(2, 25):
 
    model_dc = DecisionTreeClassifier(max_depth=depth, random_state=101)
    model_dc.fit(Scaled_Xtrain,y_train)
 
    preds = model_dc.predict(Scaled_Xtest)
 
    print(f'{depth} accuracy score: {accuracy_score(y_test, preds)}')


#%%
#ROC for decision tree
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_scores=model.predict_proba(Scaled_Xtest)
fpr,tpr,threshold =roc_curve(y_test,y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Decision Tree')
plt.show()


#%%
#Pruned decision tree
pruned_tree_1 = DecisionTreeClassifier(max_depth=2, random_state=101)
pruned_tree_1.fit(Scaled_Xtrain,y_train)
preds = pruned_tree_1.predict(Scaled_Xtest)
print(classification_report(y_test,preds))
dff = pd.DataFrame(index=X.columns,data=pruned_tree_1.feature_importances_,columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(dff.head(17))
#%%
plt.figure(figsize=(12,8),dpi=150)
plot_tree(pruned_tree_1,filled=True,feature_names=X.columns);

#%%
pruned_tree_2 = DecisionTreeClassifier(max_leaf_nodes=2)
pruned_tree_2.fit(Scaled_Xtrain,y_train)
preds = pruned_tree_2.predict(Scaled_Xtest)
print(classification_report(y_test,base_pred))
#%%
plt.figure(figsize=(12,8),dpi=150)
plot_tree(pruned_tree_2,filled=True,feature_names=X.columns);
# %%
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

modelforrc = RandomForestClassifier(n_estimators=100,max_features='auto',random_state=101)
modelforrc.fit(Scaled_Xtrain,y_train)
preds = modelforrc.predict(Scaled_Xtest)
cff = confusion_matrix(y_test,preds)
sns.heatmap(cff, annot=True,cmap='Blues', fmt='g')
print(classification_report(y_test,preds))
#%%

# %%
#ROC for random forest
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_scores=modelforrc.predict_proba(Scaled_Xtest)
fpr,tpr,threshold =roc_curve(y_test,y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Random Forest')
plt.show()






#%%
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
svc = SVC(C= 1, class_weight='balanced',probability=True)
svc.fit(Scaled_Xtrain,y_train)
preds = svc.predict(Scaled_Xtest)
cff = confusion_matrix(y_test,preds)
sns.heatmap(cff, annot=True,cmap='Blues', fmt='g')
print(classification_report(y_test,preds))




#%%
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
svc = SVC(C= 0.01, class_weight='balanced', probability=True)
svc.fit(Scaled_Xtrain,y_train)
preds = svc.predict(Scaled_Xtest)
cff = confusion_matrix(y_test,preds)
sns.heatmap(cff, annot=True,cmap='Blues', fmt='g')
print(classification_report(y_test,preds))


#%%
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_scores=svc.predict_proba(Scaled_Xtest)
fpr,tpr,threshold =roc_curve(y_test,y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of svc')
plt.show()



# %%
