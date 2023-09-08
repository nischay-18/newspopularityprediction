# %%
# # Importing Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mlp
import rfit
import os


# Reading DataSet
#%%
OnlineNewsdf = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
print(OnlineNewsdf.head())


# Total No of observations 
# %%
print(len(OnlineNewsdf))
#There are total 39644 rows in the entire dataset

# Dropping Duplicates 
# %%
OnlineNewsdf=OnlineNewsdf.drop_duplicates()
print(OnlineNewsdf.isna().sum())

# Any duplicates values in the data set are removed, and there are no null values the data set.


# Describing the DataSet after duplicates dropping
# %%
OnlineNewsdf.describe()


#The n_tokens_content columns which contains the value 0 is removed
# %%
OnlineNewsdf = OnlineNewsdf[OnlineNewsdf['n_tokens_content']!=0]


# # Correlation Heatmap
# %%
plt.figure(figsize=(42, 15))
heatmap = sns.heatmap(OnlineNewsdf.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);



# Since URL is a non-numeric attribute and will not add value to our analysis so dropping it from the dataset
# Also timedelta is a non-predictive attribute and not a feature of the data set so we can drop it from the dataset
# We observe multicollinearity variables "n_non_stop_unique_tokens","n_non_stop_words" and "kw_avg_min", hence dropping these variables.
# %%
OnlineNewsdf = OnlineNewsdf.drop('url',axis=1)
OnlineNewsdf = OnlineNewsdf.drop('timedelta',axis=1)
OnlineNewsdf= OnlineNewsdf.drop(["n_non_stop_unique_tokens", "n_non_stop_words","kw_avg_min"],axis=1)

# %%
OnlineNewsdf.head()


# creating a grading criteria for the shares based on Mean
# With mean there it is forming biased dataset
# Hence, we are grading based on Median.
# %%
Threshold = OnlineNewsdf['shares'].median()
#print(Threshold)
OnlineNewsdf['popularity'] = np.where(OnlineNewsdf.shares>Threshold,'Popular','Unpopular')
#print(OnlineNewsdf.popularity.value_counts())


# %%
OnlineNewsdf.head()


# Plots 

# shares vs n_tokens_title
# %%
sns.set_theme(style="ticks")
palette = sns.color_palette("rocket_r")
sns.relplot(
    data = OnlineNewsdf,
    x = "n_tokens_title", y = "shares",
    hue = "popularity", kind = "line", palette = palette,
    height = 5, aspect = .75, facet_kws = dict(sharex = False))
plt.xlabel('Number of words in Title')
plt.ylabel('Number of Shares')
plt.show()





# %%
sns.set_theme(style="ticks")
palette = sns.color_palette("rocket_r")
sns.relplot(
    data = OnlineNewsdf,
    x = "n_tokens_title", y = "shares",
    hue = "Data_Channel", kind = "line", palette = palette,
    height = 5, aspect = .75, facet_kws = dict(sharex = False),
)



# %%
sns.set_style(style='whitegrid')
sns.scatterplot(
    data=OnlineNewsdf, 
    x='n_tokens_content', 
    y='shares', 
    hue='popularity',
    palette='Paired_r'
    )
plt.title('Analysing Popularity based on Shares')
plt.xlabel('No of Words in Content')
plt.ylabel('Shares')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,title="Key")
plt.show() 
 



# %%
sns.set_style(style='whitegrid')
sns.scatterplot(
    data=OnlineNewsdf, 
    x='n_tokens_content', 
    y='shares', 
    hue='Data_Channel',
    palette='Paired_r'
    )
plt.title('Analysing Popularity based on Shares')
plt.xlabel('No of Tokens Content')
plt.ylabel('Shares')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


# %%
sns.countplot(x ='Data_Channel', hue = "popularity", data = OnlineNewsdf)
plt.show()

# Discussing
# %%
sns.countplot( x= "Publish_DOW", hue="Data_Channel", data=OnlineNewsdf)
plt.show()

# %%
sns.countplot( x= "Publish_DOW", hue="popularity", data=OnlineNewsdf)
plt.show()

# %%
#df1 = OnlineNewsdf[OnlineNewsdf['shares'] > 1400]
#len(df1)


# %%
sns.set_theme(style="whitegrid")
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
g = sns.relplot(
    data=OnlineNewsdf,
    x="num_hrefs", y="shares",
    hue="Data_Channel",
    sizes=(10, 200),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)



# %%
sns.set_theme(style="white") 
sns.relplot(x="num_imgs", y="shares", hue="Publish_DOW", size="Data_Channel",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data = OnlineNewsdf)



#As Dummies for Data Channel is already available in dataset, I am dropping Data_Channel variable
# %%
OnlineNewsdf= OnlineNewsdf.drop("Data_Channel", axis=1)
OnlineNewsdf.shape


# Converting the values of Publish_DOW into Numerical variable
# Week starts on Monday and ends on Sunday.
# %%
Day_of_Week_Numbers = OnlineNewsdf['Publish_DOW']
OnlineNewsdf['Publish_DOW'].describe() 

# %%
OnlineNewsdf['Publish_DOW'] = OnlineNewsdf['Publish_DOW'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], [0, 1, 2, 3, 4, 5, 6]) 






# outlier Handling
# %%
OnlineNewsdf.head(10)
# %%
