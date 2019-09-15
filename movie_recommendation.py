# Spark hw2 Movie Recommendation
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import explode
import pyspark.sql.functions as f

import numpy as np
import pandas as pd
from pyspark.mllib.recommendation import ALS, Rating #this is different from the pyspark.ml.recommendation library
import math
#from pyspark import SparkConf
from pyspark.context import SparkContext
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# Part1: Data ETL and Data Exploration
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

movies = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)
movies.show(5)

display(movies.take(5))
display(ratings.take(5)) 
display(links.take(5)) 
display(tags.take(5)) 
movies.printSchema()
ratings.printSchema()
links.printSchema()
tags.printSchema()
#Part1.1: Exploratory Data Analysis(EDA)
ratingsPD = ratings.toPandas()
moviesPD = movies.toPandas()

df_opt1 = pd.merge(ratingsPD, moviesPD, on = 'movieId')
display(df_opt1.head(5))
df_opt1['rating'].dtype
df_opt1['rating'] = df_opt1['rating'].apply(pd.to_numeric) #convert to numeric value
df_opt1.groupby("title")["rating"].mean().sort_values(ascending = False).head(7) #groupBy("title"), apply aggregation function .mean() on "rating" column
df_opt1.groupby('title')['rating'].count().sort_values(ascending=False).head(5) #see which movies have the most rating
title_ratings = pd.DataFrame(df_opt1.groupby('title')['rating'].mean())
title_ratings.head(5)
#we are more interested in how many people rated one movie, calculate mean() accordingly
title_ratings['num of ratings'] = pd.DataFrame(df_opt1.groupby('title')['rating'].count())
title_ratings.head(5)
# Show histogram plot
display(title_ratings) #display num of ratings 
#Most movies have either zero or one rating
#which makes sense because most people only watch the famous or big-hit movies
display(title_ratings) #display ratings distribution
#Movie ratings distributed normally, around 3 stars to 3.5 stars
#outlier 1-star movies, bad movies that only few people watch it
sns.set_style('white')
fig, ax = plt.subplots()
sns.jointplot(x = 'rating', y = 'num of ratings', data = title_ratings, alpha = 0.5)
display(fig.show())
#As you get more ratings, you tend to have a higher rating of a movie
#the better the movie is, the more people are going to watch it, the more ratings you receive
#Print the results
tmp1 = ratings.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}.'.format(tmp1))
print('Minimum number of ratings per movie is {}.'.format(tmp2))
tmp1 = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user.'.format(tmp1, tmp2))
# Part2 Spark SQL and OLAP
num_of_users = ratings.select("userID").distinct().count()
print("The number of users is {}.".format(num_of_users))
num_of_movies = movies.select("movieID").distinct().count()
print("The number of movies is {}.".format(num_of_movies))
# How many movies are rated by users? List movies not rated before
ratings = ratings.dropna(subset=["rating"])
num_of_rated_movies = ratings.select("movieID").distinct().count()
print("There are {} movies that have been rated by users.".format(num_of_rated_movies))
# List movies not rated
movies_not_rated = []
movies_dict = {}
movies_df = movies.toPandas()
movies_title = movies_df['title']
movies_id = movies_df['movieId']
for i in range(len(movies_id)):
  movies_dict[movies_id[i]] = movies_title[i]
movies_rated = set(ratings.select('movieId').distinct().toPandas()['movieId'])
for movies_id in movies_dict.keys():
  if movies_id not in movies_rated:
    movies_not_rated.append(movies_dict[movies_id])
print(len(movies_not_rated))
print(movies_not_rated)
# List Movie Genres
@udf(ArrayType(StringType())) #employ python user-defined function
def splitUdf(x):
    splitted = x.split('|')
    return [i for i in splitted]

dfv = movies.withColumn('genres', splitUdf(col('genres'))) \
     .select(col('movieID'), col('title'), col('genres'))
display(dfv.take(5))
df_genre = dfv.withColumn("genres", explode("genres"))
distinct_df_genre = df_genre.select("genres").distinct()
display(distinct_df_genre.take(5))
genre_count = distinct_df_genre.count()
print("The total number of genres is {}.".format(genre_count))
# Movie for Each Category
df_genre_movies = df_genre.groupby("genres").agg(f.concat_ws(", ", f.collect_list(df_genre.title))) #aggregate function: in opposed to the explode() attribute
display(df_genre_movies.take(1))
#create a temp view for SQL statement
dfv.createOrReplaceTempView("dfv") 
df_genre.createOrReplaceTempView("df_genre") 
%sql
SELECT b.genres, COUNT(*) AS count
FROM dfv AS a
JOIN df_genre AS b
ON a.title = b.title
GROUP BY b.genres
ORDER BY count DESC
# We define 'most popular genres' based on frequency of the genre occurrence among all movies.
# Top-10 popular genres are drama, comedy, thriller, action, romance, adventure, crime, sci-fi, horror, and fantasy.
# Part3: Recommending Similar Movies
#Create a matrix that has the user ids on one axis and the movie title on another axis
#Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN(missing) values, because most people have not seen most of the movies
#Memory-based models are based on similarity between items or users, where we use cosine-similarity
moviemat = df_opt1.pivot_table(index = 'userId', columns = 'title', values = 'rating')
display(moviemat.head(5))
# Most rated movie
title_ratings.sort_values('num of ratings', ascending = False).head(5)
#choose two movies: Forrest Gump, a comedy-drama movie. And Shawshank Redemption, a crime fiction
forrestGump_user_ratings = moviemat['Forrest Gump (1994)']
shawshank_user_ratings = moviemat['Shawshank Redemption, The (1994)']
forrestGump_user_ratings.head(5)
#correlation method: get the correlation between two pandas series
similar_to_forrestGump = moviemat.corrwith(forrestGump_user_ratings)
similar_to_shawshank = moviemat.corrwith(shawshank_user_ratings)
#Get the correlations between movies
corr_forrestGump = pd.DataFrame(similar_to_forrestGump, columns=['Correlation'])
corr_forrestGump.dropna(inplace = True)
corr_forrestGump.head(5)
corr_shawshank = pd.DataFrame(similar_to_shawshank, columns = ['Correlation'])
corr_shawshank.dropna(inplace = True)
corr_shawshank.head(5)
#Sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense
#This is because there are a lot of movies only watched once by users who also watched Forrest Gump (it was the most popular movie)
corr_forrestGump.sort_values('Correlation', ascending = False).head(5)
# We can fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier)
corr_forrestGump = corr_forrestGump.join(title_ratings['num of ratings'])
corr_forrestGump .head(5)
corr_forrestGump[corr_forrestGump['num of ratings'] > 100].sort_values('Correlation', ascending = False).head(5)
#same philosophy applied to other movies
corr_shawshank = pd.DataFrame(similar_to_shawshank,columns = ['Correlation'])
corr_shawshank.dropna(inplace = True)
corr_shawshank = corr_shawshank.join(title_ratings['num of ratings'])
corr_shawshank[corr_shawshank['num of ratings'] > 100].sort_values('Correlation', ascending = False).head(5)
#Part4: Spark ALS based approach for training model
#Use an RDD-based API from pyspark.mllib to predict the ratings, so let's reload "ratings.csv" using sc.textFile and then convert it to the form of (user, item, rating) tuples
#Model-based CF is based on matrix factorization where we use Singular value decomposition (SVD) to factorize the matrix
#Reference Spark ML ALS model
#sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
movie_rating = sc.textFile("/FileStore/tables/ratings.csv")
header = movie_rating.take(1)[0]
#convert to tuples
rating_data = movie_rating.filter(lambda line: line!=header).map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
# check three rows
rating_data.take(3)
#Split the data into training/validation/testing sets using a 6/2/2 ratio
training_RDD, validation_RDD, test_RDD = rating_data.randomSplit([6, 2, 2], seed = 774)
# Prepare userID and movieID columns for the use of making predictions
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
training_RDD.cache()
validation_RDD.cache()
test_RDD.cache()
#ALS Model Selection and Evaluation
#With the ALS model, we can use a grid search to find the optimal hyperparameters
#Use an RDD-based API from pyspark.mllib to predict ratings
def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_params:
            #train ALS model using pyspark.mllib.recommendation - RDD-based API
            model = ALS.train(training_RDD, rank, seed = 774, iterations = num_iters,
                              lambda_ = reg)
            #make predictions
            predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
            #get the rating result
            rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
            #get the RMSE /(error) of cross validation data
            cv_error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            print ('{} latent factors and regularization = {}: validation RMSE is {:.4f}'.format(rank, reg, cv_error)) 
            if cv_error < min_error:
                min_error = cv_error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print ('\nThe best model was trained with {} which has {} latent factors. \nRegularization parameter = {}'.format(best_rank, best_rank, best_regularization))
    return best_model
num_iterations = 10   #How many times will be iterate over the data
ranks = [6, 8, 10, 12, 14]   #Creates a list of latent factors, experiement with each
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]   #Test with regularization parameter

start_time = time.time()
final_model = train_ALS(training_RDD, validation_for_predict_RDD, num_iterations, reg_params, ranks)

print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))
#Apply the final model and plot a learning curve based on errors
rating_data_for_predict_RDD = rating_data.map(lambda x: (x[0], x[1]))
predictions = final_model.predictAll(rating_data_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = rating_data.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
#get X: features
X = rates_and_preds.map(lambda r: (r[0][0], r[0][1], r[1][0]))  
 #get y: target
y = rates_and_preds.map(lambda r: r[1][1])  
X_arr = np.reshape(np.array(X.collect()), (-1, 3))
X_arr.shape
X_df = pd.DataFrame(X_arr, columns = ["userID", "movieID", "rating"])
#X_df = X_df.dropna()
display(X_df.head(5))
y_arr = np.reshape(np.array(y.collect()), (-1, 1))
y_arr.shape
y_df =  pd.DataFrame(y_arr, columns = ["predicted_rating"])
display(y_df.head(5))
X = X_df[["userID", "movieID", "rating"]] #features we train
y = y_df[["predicted_rating"]] #target
cv = 5 #k-fold cross validation
iter_array = [1, 100, 500, 1000] #train_sizes
#Plotting learning curve
def plot_learning_curve():
  lg = LinearRegression()
  lg.fit(X, y) #fit
  train_sizes, train_scores, validation_scores = learning_curve( \
                                                                estimator = lg, \
                                                                X = X, \
                                                                y = y, \
                                                                cv = cv, \
                                                                scoring = 'neg_mean_squared_error', \
                                                                train_sizes = iter_array \
                                                               )
    
  #calculate mean and std deviation for training and cross validation data
  train_scores_mean = -np.mean(train_scores, axis=1) #use the minus sign to offset negative mean squared error
  train_scores_std = np.std(train_scores, axis=1)
  validation_scores_mean = -np.mean(validation_scores, axis=1) 
  validation_scores_std = np.std(validation_scores, axis=1)
  
  #plot learning curve
  plt.clf() #clear the old figure before plotting a new one
  plt.figure() #create figure object
  plt.style.use('seaborn-whitegrid')
  plt.title('Learning Curves for a Linear Regression Model', fontsize = 18)
  plt.xlabel('Training set size', fontsize = 14)
  plt.ylabel('MSE', fontsize = 14)
  
  #plot the std deviation as a transparent range at each training set size
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
  plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
  plt.plot(train_sizes, train_scores_mean, color = "r", label = "Training error")
  plt.plot(train_sizes, validation_scores_mean, color = "g", label = "Cross-validation error")
  plt.legend(labelspacing=0.5, bbox_to_anchor=(1.00, 0.98), fontsize = 12)

  plt.ylim(0, 0.6)
  plt.show()
display(plot_learning_curve())
#take a look at those scores
lg = LinearRegression()
lg.fit(X, y) #fit
train_sizes, train_scores, validation_scores = learning_curve( \
                                                                estimator = lg, \
                                                                X = X, \
                                                                y = y, \
                                                                cv = cv, \
                                                                scoring = 'neg_mean_squared_error', \
                                                                train_sizes = iter_array \
                                                               )
    
#calculate mean and std deviation for training and cross validation data
train_scores_mean = -np.mean(train_scores, axis=1) #use the minus sign to offset negative mean squared error
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1) 
validation_scores_std = np.std(validation_scores, axis=1)
print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) 
print('\nStd deviation training scores\n\n', pd.Series(train_scores_std, index = train_sizes))
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
print('\n', '-' * 20) 
print('\nStd deviation validation scores\n\n',pd.Series(validation_scores_std, index = train_sizes))
def plot_learning_curve_DecisionTree():
  dt = DecisionTreeRegressor()
  dt.fit(X, y) #fit
  train_sizes, train_scores, validation_scores = learning_curve( \
                                                                estimator = dt, \
                                                                X = X, \
                                                                y = y, \
                                                                cv = cv, \
                                                                scoring = 'neg_mean_squared_error', \
                                                                train_sizes = iter_array \
                                                               )
    
  #calculate mean and std deviation for training and cross validation data
  train_scores_mean = -np.mean(train_scores, axis=1) #use the minus sign to offset negative mean squared error
  train_scores_std = np.std(train_scores, axis=1)
  validation_scores_mean = -np.mean(validation_scores, axis=1) 
  validation_scores_std = np.std(validation_scores, axis=1)
  
  #plot learning curve
  plt.clf() #clear the old figure before plotting a new one
  plt.figure() #create figure object
  plt.style.use('seaborn-whitegrid')
  plt.title('Learning Curves for a Decision Tree Model', fontsize = 18)
  plt.xlabel('Training set size', fontsize = 14)
  plt.ylabel('MSE', fontsize = 14)
  
  #plot the std deviation as a transparent range at each training set size
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
  plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
  plt.plot(train_sizes, train_scores_mean, color = "r", label = "Training error")
  plt.plot(train_sizes, validation_scores_mean, color = "g", label = "Cross-validation error")
  plt.legend(labelspacing=0.5, bbox_to_anchor=(1.00, 0.98), fontsize = 12)

  plt.ylim(-0.1, 1)
  plt.show()
display(plot_learning_curve_DecisionTree())
#Part5: Model testing on the test data
#Finally, make predictions and check the testing error
#Apply the final model to testing data
test_predictions = final_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
test_rates_and_preds = rating_data.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(test_predictions)
test_results_RDD = test_rates_and_preds.map(lambda r: (r[1][0], r[1][1]))   
arr_test = np.reshape(np.array(test_results_RDD.collect()), (-1, 2))
arr_test.shape
df_test =  pd.DataFrame(arr_test, columns = ["rating", "predicted_rating"])
display(df_test.head(5))
test_error = math.sqrt(test_rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print(test_error)
# Part 6 Key observations
#Part6: Key Observations
#Model Performance
#RMSE of test data is at 0.8912, close to the least validation RMSE we obtained with the best model. Model performed well.
#In this case, RMSE is used as one metric for model evaluation. We could experiment with other metrics such as F1 score, accuracy, etc.
#This project was initialized with a small movie rating dataset. Errors and learning curve would look different compared to that of the complete dataset.
#In the linear regression model, we see validation error is slightly lower than training error while training set size increases. Generally speaking, training error will almost always underestimate your validation error. However it is possible for the validation error to be less than the training:
#Training set had many 'hard' cases to learn.
#Validation set had mostly 'easy' cases to predict.
#That is why it is important that you really evaluate your model training methodology. If you don't split your data for training properly your results will lead to confusing, if not simply incorrect.
#Think of model evaluation in four different categories:
#Underfitting – Validation and training error high
#Overfitting – Validation error is high, training error low
#Good fit – Validation error low, slightly higher than the training error
#Unknown fit - Validation error low, training error 'high'
#Model trained with the small dataset using Linear Regression Estmator is somewhere between a good fit and a "unknown fit"
#Training error (red line) increases and plateau
#Indicates high bias scenario
#Cross-validation error (green line) stagnating almost throughout
#Unable to learn from data
#Low MSE (low errors)
#Should tweak model (suggestion: perhaps reduce model complexity)
#Model trained with the small dataset using Decision Tree Regression estmator is overfitting
#Training error (red line) is at its minimum regardless of training examples
#Shows severe overfitting
#Cross-validation error (green line) decreases over time
#Huge gap between cross-validation error and training error
#Indicates high variance scenario
#suggestions:
#Reduce complexity of the model or gather more data (use the complete dataset)
#Part7: Reference
#References of ALS
#Building a Movie Recommendation Service: https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw
#MLlib - Collaborative Filtering: https://spark.apache.org/docs/1.1.0/mllib-collaborative-filtering.html
#Collaborative Filtering - RDD-based API: https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
#Evaluation Metrics - RDD-based API: https://spark.apache.org/docs/2.2.0/mllib-evaluation-metrics.html
#ML - ALS: https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
#References of Learning Curve
#Learning Curve: https://www.ritchieng.com/machinelearning-learning-curve/
#Tutorial: Learning Curves: https://www.dataquest.io/blog/learning-curves-machine-learning/
