# Anime Recommendation System

<br>

![Naruto](https://wallpaperscave.com/images/thumbs/download/1280x768/18/02-21/anime-naruto-19895.jpg "Naruto")


## Introduction

From anime [dataset](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020?select=rating_complete.csv), We create recommendation system witch use cluster technique.<br>
Recommended anime were extracted from characteristic of cluster.<br>
User was segmented by user anime rating history.

[myanimelist site](https://myanimelist.net/)


### Steps

- Preprocessing
- Visualization
- K-Mean clustering
- Characteristic of each cluster


### Libraries

Import all libraries we need for data mining.



```python
# Basic libraries
from random import randint
from chernoff_faces import cface

# Import numpy, pandas and matplot libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from wordcloud import WordCloud
%matplotlib inline

# Machine learning libraries
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Finding spark in jupyter notebook
import findspark
findspark.init()

# Frequent pattern libraries
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import *

# Import HTML lib for changing direction of page
from IPython.display import HTML
from IPython.display import display
HTML('<style>.output{flex-direction:row;flex-wrap:wrap}</style>')

# Setup style of charts
plt.style.use('seaborn')
%config InlineBackend.figure_formats = {'png', 'retina'}

```

## Preprocessing

In this section, we are going to pre-process the data, so we will first clean the data and replace the missing values with valid values - The attribute mean for all samples belonging to the same class - and then remove the extra columns, and we will also combine the data sets. And sampling data to reduce data volume.


### Read anime dataset



```python
# Read anime dataset from csv file
anime = pd.read_csv("../data/anime.csv")
# Show the first 3 records
anime.head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>Spring 1998</td>
      <td>...</td>
      <td>229170.0</td>
      <td>182126.0</td>
      <td>131625.0</td>
      <td>62330.0</td>
      <td>20688.0</td>
      <td>8904.0</td>
      <td>3184.0</td>
      <td>1357.0</td>
      <td>741.0</td>
      <td>1580.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>8.39</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>Cowboy Bebop:The Movie</td>
      <td>カウボーイビバップ 天国の扉</td>
      <td>Movie</td>
      <td>1</td>
      <td>Sep 1, 2001</td>
      <td>Unknown</td>
      <td>...</td>
      <td>30043.0</td>
      <td>49201.0</td>
      <td>49505.0</td>
      <td>22632.0</td>
      <td>5805.0</td>
      <td>1877.0</td>
      <td>577.0</td>
      <td>221.0</td>
      <td>109.0</td>
      <td>379.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>8.24</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>Trigun</td>
      <td>トライガン</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>Spring 1998</td>
      <td>...</td>
      <td>50229.0</td>
      <td>75651.0</td>
      <td>86142.0</td>
      <td>49432.0</td>
      <td>15376.0</td>
      <td>5838.0</td>
      <td>1965.0</td>
      <td>664.0</td>
      <td>316.0</td>
      <td>533.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 35 columns</p>
</div>




```python
# Count of row and column
anime.shape

```




    (17562, 35)



### Anime dataset reduction



```python
# Remove extra columns from anime dateframe and then rename `MAL_ID` to `Anime_ID`,
# also set `Anime_ID` to index of dataframe
anime = (
    anime[
        [
            "MAL_ID",
            "Ranked",
            "Popularity",
            "Name",
            "Genres",
            "Type",
            "Source",
            "Rating",
            "Episodes",
            "Score",
            "Members",
            "Favorites",
        ]
    ]
    .rename(
        {
            "MAL_ID": "anime_id",
            "Ranked": "ranked",
            "Popularity": "popularity",
            "Name": "name",
            "Genres": "genres",
            "Type": "type",
            "Source": "source",
            "Rating": "rating",
            "Episodes": "episodes",
            "Score": "score",
            "Members": "members",
            "Favorites": "favorites",
        },
        axis=1,
    )
    .set_index("anime_id")
)

```


```python
# Remove invalid row with `Unknown` ranked
anime = anime[(anime["ranked"] != "Unknown") & (anime["ranked"] != "0.0")]

```

### Anime dataset transaction



```python
# Replace missing `Score` and `Episodes` with zero
anime["score"].replace("Unknown", 0.0, inplace=True)
anime["episodes"].replace("Unknown", 0, inplace=True)

```


```python
# Change the `Ranked`, `Episodes` and `Score` columns to numeric for math operations,
# as well as sort the table by `Ranked`
anime = (
    anime.astype({"ranked": "float"})
    .astype({"ranked": "int", "episodes": "int", "score": "float"})
    .sort_values("ranked")
)

```

### Calculate mean for each type



```python
# Calculate mean of `Score` and `Episodes` for each `Type`
group_by_type = anime.groupby("type")
print("✓ Mean of score for each type")
display(mean_scores := group_by_type["score"].mean().round(2))

print("\n✓ Mean of episodes for each type")
display(mean_episodes := group_by_type["episodes"].mean().round().astype(int))

```

    ✓ Mean of score for each type
    


    type
    Movie      4.42
    Music      2.93
    ONA        3.58
    OVA        4.27
    Special    5.14
    TV         5.50
    Name: score, dtype: float64


    
    ✓ Mean of episodes for each type
    


    type
    Movie       1
    Music       1
    ONA         9
    OVA         2
    Special     2
    TV         33
    Name: episodes, dtype: int32


### Handel missing values



```python
# Replace zeroes `Score` with its own category mean
for index in mean_scores.index:
    anime["score"].mask(
        (anime["type"] == index) & (anime["score"] == 0.0),
        mean_scores[index],
        inplace=True,
    )

```


```python
# Replace zeroes `Episodes` with its own category mean
for index in mean_episodes.index:
    anime["episodes"].mask(
        (anime["type"] == index) & (anime["episodes"] == 0),
        mean_episodes[index],
        inplace=True,
    )

```

### Write reduced anime dataset



```python
# Wright anime dataset to csv file
anime.to_csv("../data/anime_reduce.csv")

```

### Display information of anime dataset



```python
anime = pd.read_csv("../data/anime_reduce.csv")
anime.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anime_id</th>
      <th>ranked</th>
      <th>popularity</th>
      <th>name</th>
      <th>genres</th>
      <th>type</th>
      <th>source</th>
      <th>rating</th>
      <th>episodes</th>
      <th>score</th>
      <th>members</th>
      <th>favorites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40028</td>
      <td>2</td>
      <td>119</td>
      <td>Shingeki no Kyojin: The Final Season</td>
      <td>Action, Military, Mystery, Super Power, Drama,...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>16</td>
      <td>9.17</td>
      <td>733260</td>
      <td>44862</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>3</td>
      <td>9</td>
      <td>Steins;Gate</td>
      <td>Thriller, Sci-Fi</td>
      <td>TV</td>
      <td>Visual novel</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>24</td>
      <td>9.11</td>
      <td>1771162</td>
      <td>148452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38524</td>
      <td>4</td>
      <td>63</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>Action, Drama, Fantasy, Military, Mystery, Sho...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>10</td>
      <td>9.10</td>
      <td>1073626</td>
      <td>40985</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28977</td>
      <td>5</td>
      <td>329</td>
      <td>Gintama°</td>
      <td>Action, Comedy, Historical, Parody, Samurai, S...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>51</td>
      <td>9.10</td>
      <td>404121</td>
      <td>11868</td>
    </tr>
  </tbody>
</table>
</div>




```python
anime.shape

```




    (15798, 12)




```python
anime.describe().round(2)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anime_id</th>
      <th>ranked</th>
      <th>popularity</th>
      <th>episodes</th>
      <th>score</th>
      <th>members</th>
      <th>favorites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15798.00</td>
      <td>15798.00</td>
      <td>15798.00</td>
      <td>15798.00</td>
      <td>15798.00</td>
      <td>15798.00</td>
      <td>15798.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21601.96</td>
      <td>7896.21</td>
      <td>8884.46</td>
      <td>12.46</td>
      <td>5.88</td>
      <td>37752.95</td>
      <td>503.86</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14671.07</td>
      <td>4556.74</td>
      <td>5229.80</td>
      <td>49.12</td>
      <td>1.36</td>
      <td>131649.10</td>
      <td>4281.61</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.85</td>
      <td>25.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6514.00</td>
      <td>3946.50</td>
      <td>4084.00</td>
      <td>1.00</td>
      <td>5.14</td>
      <td>298.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23350.00</td>
      <td>7896.00</td>
      <td>9195.50</td>
      <td>2.00</td>
      <td>6.08</td>
      <td>1737.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35454.75</td>
      <td>11845.75</td>
      <td>13518.75</td>
      <td>12.00</td>
      <td>6.92</td>
      <td>15706.00</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48480.00</td>
      <td>15780.00</td>
      <td>17560.00</td>
      <td>3057.00</td>
      <td>9.19</td>
      <td>2589552.00</td>
      <td>183914.00</td>
    </tr>
  </tbody>
</table>
</div>



### Read rating dataset



```python
rating = pd.read_csv("../data/rating.csv").rename({"rating": "user_rating"}, axis=1)
rating.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>user_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>430</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1004</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3010</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>570</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2762</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>431</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>578</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>433</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1571</td>
      <td>10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>121</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.shape

```




    (57633278, 3)




```python
rating["user_rating"].describe().round(2)

```




    count    57633278.00
    mean            7.51
    std             1.70
    min             1.00
    25%             7.00
    50%             8.00
    75%             9.00
    max            10.00
    Name: user_rating, dtype: float64



### Definition of Like

Because of many users, many differences criteria for rating anime.
Then, We decide to find rating mean of each user.
Anime which got rating higher than user rating mean will assign as like

```
rating['user_rating'] > rating['mean_rating'] => User like this anime
```

![like](https://www.vhv.rs/dpng/f/561-5618483_instagram-likes-png.png "like")



```python
# User 922 has a low in rating mean
rating[rating["user_id"] == 922]["user_rating"].mean().round(3)

```




    1.0




```python
# User 99 has a middle in rating mean
rating[rating["user_id"] == 105]["user_rating"].mean().round(3)

```




    5.0




```python
# User 12 has a hight in rating mean
rating[rating["user_id"] == 12]["user_rating"].mean().round(3)

```




    10.0



### Calculate mean rating per user



```python
mean_rating_per_user = rating.groupby(["user_id"]).mean().reset_index()
mean_rating_per_user["mean_rating"] = mean_rating_per_user["user_rating"]

mean_rating_per_user.drop(["anime_id", "user_rating"], axis=1, inplace=True)

```


```python
mean_rating_per_user.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8.058252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7.603175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7.652542</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>8.162791</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7.073955</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7.908046</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>7.611111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>7.750000</td>
    </tr>
  </tbody>
</table>
</div>



### Merge mean rating to rating data frame



```python
rating = pd.merge(rating, mean_rating_per_user, on=["user_id", "user_id"])

```


```python
rating.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>user_rating</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>430</td>
      <td>9</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1004</td>
      <td>5</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3010</td>
      <td>7</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>570</td>
      <td>7</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2762</td>
      <td>9</td>
      <td>7.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop anime's user don't liked
rating = rating.drop(rating[rating["user_rating"] < rating["mean_rating"]].index)

```


```python
# user 922 favorite only one anime
rating[rating["user_id"] == 922].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>user_rating</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149818</th>
      <td>922</td>
      <td>16870</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# user 105 favorite only one anime
rating[rating["user_id"] == 105].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>user_rating</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13008</th>
      <td>105</td>
      <td>249</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating[rating["user_id"] == 12].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>user_rating</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1246</th>
      <td>12</td>
      <td>31964</td>
      <td>10</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>12</td>
      <td>16335</td>
      <td>10</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>12</td>
      <td>11021</td>
      <td>10</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>12</td>
      <td>35062</td>
      <td>10</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>12</td>
      <td>20785</td>
      <td>10</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.shape

```




    (30706635, 4)




```python
# Number of users
len(rating["user_id"].unique())

```




    310059



### Combine two datasets

In this section, we decide to reduce size of dataset, because of running time and memory



```python
# Merge anime and rating data frame
mergedata = pd.merge(anime, rating, on=["anime_id", "anime_id"])
# Choice record with User_ID lower than equal 25000, This is a random choice
mergedata = mergedata[mergedata["user_id"] <= 25000]
mergedata.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anime_id</th>
      <th>ranked</th>
      <th>popularity</th>
      <th>name</th>
      <th>genres</th>
      <th>type</th>
      <th>source</th>
      <th>rating</th>
      <th>episodes</th>
      <th>score</th>
      <th>members</th>
      <th>favorites</th>
      <th>user_id</th>
      <th>user_rating</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
      <td>1</td>
      <td>10</td>
      <td>8.058252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
      <td>6</td>
      <td>10</td>
      <td>7.073955</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
      <td>7</td>
      <td>10</td>
      <td>7.908046</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
      <td>11</td>
      <td>10</td>
      <td>8.503106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5114</td>
      <td>1</td>
      <td>3</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Military, Adventure, Comedy, Drama, Ma...</td>
      <td>TV</td>
      <td>Manga</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>64</td>
      <td>9.19</td>
      <td>2248456</td>
      <td>183914</td>
      <td>12</td>
      <td>10</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count of anime in mergedata
len(mergedata["anime_id"].unique())

```




    11411




```python
# Count of anime in actual dataset
len(anime["anime_id"].unique())

```




    15798



### Write merge dataset



```python
mergedata.to_csv('../data/mergedata.csv')

```

### Create Crosstable

Show detail of anime which each user like



```python
mergedata = pd.read_csv("../data/mergedata.csv")
user_anime = pd.crosstab(mergedata["user_id"], mergedata["name"])
user_anime.columns.name = None
user_anime.index.name = None

user_anime.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>"0"</th>
      <th>"Bungaku Shoujo" Kyou no Oyatsu: Hatsukoi</th>
      <th>"Bungaku Shoujo" Memoire</th>
      <th>"Bungaku Shoujo" Movie</th>
      <th>"Calpis" Hakkou Monogatari</th>
      <th>"Eiji"</th>
      <th>"Eiyuu" Kaitai</th>
      <th>"Kiss Dekiru Gyoza" x Mameshiba Movie</th>
      <th>"Parade" de Satie</th>
      <th>"R100" x Mameshiba Original Manners</th>
      <th>...</th>
      <th>s.CRY.ed Alteration I: Tao</th>
      <th>s.CRY.ed Alteration II: Quan</th>
      <th>the FLY BanD!</th>
      <th>xxxHOLiC</th>
      <th>xxxHOLiC Kei</th>
      <th>xxxHOLiC Movie: Manatsu no Yoru no Yume</th>
      <th>xxxHOLiC Rou</th>
      <th>xxxHOLiC Shunmuki</th>
      <th>ēlDLIVE</th>
      <th>◯</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 11409 columns</p>
</div>




```python
user_anime.shape

```




    (21804, 11409)



### Principal Component Analysis (PCA)

Principal Component Analysis converts our original variables to a new set of variables,
which are a linear combination of the original set of variables.
Main goal is to reduce dimension of data for clustering and visualize

[wikipedia](<https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.>)



```python
pca = PCA(n_components=3)
pca.fit(user_anime)
pca_samples = pca.transform(user_anime)

```


```python
ps = pd.DataFrame(pca_samples)
ps.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.461742</td>
      <td>0.477328</td>
      <td>-0.081578</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.083951</td>
      <td>-1.401628</td>
      <td>-1.014477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.693449</td>
      <td>-0.630825</td>
      <td>-0.298041</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.778695</td>
      <td>0.226784</td>
      <td>1.138751</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.802398</td>
      <td>1.464769</td>
      <td>0.436394</td>
    </tr>
  </tbody>
</table>
</div>




```python
ps.shape

```




    (21804, 3)



## Visualization

In this section, we use a variety of graphs to display the data so that we can better understanding


### Data points in 3D PCA axis



```python
to_cluster = pd.DataFrame(ps[[0, 1, 2]])

```


```python
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection="3d")
ax.scatter(to_cluster[0], to_cluster[2], to_cluster[1], s=24)

plt.title("Data points in 3D PCA axis", fontsize=18)
plt.savefig("../charts/Data_points_in_3D_PCA_axis.png")
plt.show()

```


    
![png](charts/Data_points_in_3D_PCA_axis.png)
    


### Data points in 2D PCA axis



```python
plt.scatter(to_cluster[1], to_cluster[0], s=24)
plt.xlabel("x_values")
plt.ylabel("y_values")
plt.title("Data points in 2D PCA axis", fontsize=18)
plt.savefig("../charts/Data_points_in_2D_PCA_axis.png")
plt.show()

```


    
![png](charts/Data_points_in_2D_PCA_axis.png)
    


### Chernoff faces

Display multivariate data in the shape of a human face. The individual parts, such as eyes, ears, mouth and nose represent values of the variables by their shape, size, placement and orientation [(wikipedia)](https://en.wikipedia.org/wiki/Chernoff_face)



```python
# Reduce dimension for face attributes
pca = PCA(n_components=17)
pca.fit(user_anime)
pca_samples = pca.transform(user_anime)
ps = pd.DataFrame(pca_samples)
ps.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.461742</td>
      <td>0.477327</td>
      <td>-0.081578</td>
      <td>0.064732</td>
      <td>0.634798</td>
      <td>0.142396</td>
      <td>-0.051592</td>
      <td>0.269147</td>
      <td>-0.324508</td>
      <td>-0.142589</td>
      <td>0.113462</td>
      <td>-0.245301</td>
      <td>-0.565451</td>
      <td>0.133959</td>
      <td>-0.538926</td>
      <td>-0.205612</td>
      <td>0.021235</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.083951</td>
      <td>-1.401628</td>
      <td>-1.014476</td>
      <td>0.235317</td>
      <td>-0.484953</td>
      <td>0.098983</td>
      <td>-1.004047</td>
      <td>0.046567</td>
      <td>0.459869</td>
      <td>0.146468</td>
      <td>0.718675</td>
      <td>0.356861</td>
      <td>0.807967</td>
      <td>-0.057999</td>
      <td>-0.275366</td>
      <td>0.364260</td>
      <td>0.360791</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.693449</td>
      <td>-0.630825</td>
      <td>-0.298040</td>
      <td>0.511194</td>
      <td>-0.784856</td>
      <td>0.568464</td>
      <td>0.385865</td>
      <td>-0.590062</td>
      <td>0.612352</td>
      <td>-0.411691</td>
      <td>-0.189001</td>
      <td>0.241942</td>
      <td>0.175324</td>
      <td>-0.141344</td>
      <td>0.211314</td>
      <td>0.080413</td>
      <td>0.156759</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.778695</td>
      <td>0.226784</td>
      <td>1.138750</td>
      <td>-1.648948</td>
      <td>-0.991642</td>
      <td>-1.601837</td>
      <td>1.068610</td>
      <td>-0.782450</td>
      <td>-1.796181</td>
      <td>-0.444752</td>
      <td>-0.452496</td>
      <td>-0.774258</td>
      <td>0.586723</td>
      <td>-0.709391</td>
      <td>1.154184</td>
      <td>-0.719425</td>
      <td>-0.100087</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.802398</td>
      <td>1.464768</td>
      <td>0.436394</td>
      <td>0.679526</td>
      <td>0.561298</td>
      <td>-0.313808</td>
      <td>-1.640556</td>
      <td>-0.554039</td>
      <td>-0.152285</td>
      <td>0.861153</td>
      <td>-0.331548</td>
      <td>-0.356287</td>
      <td>-0.692186</td>
      <td>0.140269</td>
      <td>0.834920</td>
      <td>-0.020511</td>
      <td>0.596688</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change size of figure
fig = plt.figure(figsize=(11, 11))

# Create sixteen face by random selection
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, aspect="equal")
    cface(ax, 0.9, *ps.iloc[randint(0, 21804), :])
    ax.axis([-1.2, 1.2, -1.2, 1.2])
    ax.set_xticks([])
    ax.set_yticks([])

fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("../charts/Chernoff_faces.png", bbox_inches="tight")
plt.show()

```


    
![png](charts/Chernoff_faces.png)
    


### Box plots



```python
fig, axes = plt.subplots(1, 3, figsize=(12, 7))
fig.suptitle("Box plots", fontsize=18)

anime["score"].plot.box(ax=axes[0])
anime["popularity"].plot.box(ax=axes[1])
rating["user_rating"].plot.box(ax=axes[2])

plt.savefig("../charts/Box_plots.png")
plt.show()

```


    
![png](charts/Box_plots.png)
    


### Score histogram



```python
data = anime["score"].values
bins = np.arange(1, 11)
plt.hist(data, bins, histtype="bar", rwidth=0.95)

plt.title("Score histogram", fontsize=18)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel("score")
plt.ylabel("value_count")

plt.savefig("../charts/Score_histogram.png")
plt.show()

```


    
![png](charts/Score_histogram.png)
    


### Pixel-Oriented Visualization

Pixel plots are the representation of a 2-dimension data set. In these plots, each pixel refers to a different value in a data set



```python
# Select first thousand data
head_anime = anime.head(1000)
columns = ["ranked", "popularity", "episodes", "score", "members", "favorites"]

# Creating a plot
fig, axes = plt.subplots(1, 6, figsize=(20, 8))

index = 0
for ax in axes:
    # plotting a plot
    data = head_anime[columns[index]].values.reshape((50, 20))
    ax.grid(False)
    ax.pcolor(data, cmap="Blues")

    # Customizing plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(columns[index], fontsize=18, labelpad=20)

    index += 1

# Save a plot
plt.savefig("../charts/Pixel_oriented.png")

# Show plot
plt.show()

```


    
![png](charts/Pixel_oriented.png)
    


### Score quantile plot

In statistics, a Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other.



```python
data = anime["score"]
fig, ax = plt.subplots(1, 1)

# Create quantile plot
stats.probplot(data, dist="norm", plot=ax)

# Calculate quantiles
median = data.median()
percent_25 = data.quantile(0.25)
percent_75 = data.quantile(0.75)

# Guide lines
plt.plot([-4, 4], [percent_25, percent_25], linestyle="dashed", label="Quartile1")
plt.plot([-4, 4], [median, median], linestyle="dashed", label="Median")
plt.plot([-4, 4], [percent_75, percent_75], linestyle="dashed", label="Quartile3")

# Customizing plot
plt.title("Score quantile plot", fontsize=18)
plt.legend()

plt.savefig("../charts/Score_quantile_plot.png")
plt.show()

```


    
![png](charts/Score_quantile_plot.png)
    


### Scatter plot matrices

Provides a first look at data to see clusters of points, outliers, etc



```python
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
fig.suptitle("Scatter plot", fontsize=18)
fig.supxlabel("Ranked")

group_setting = {
    # Type  marker   color   alpha
    "Movie": ["X", "#E72626", 0.4],
    "Music": ["^", "#48DA4B", 0.2],
    "ONA": [".", "#E820D7", 0.2],
    "OVA": ["*", "#FFD323", 0.2],
    "Special": [",", "#E4337A", 0.2],
    "TV": ["+", "#3719CC", 0.4],
}

# plotting a plot
for name in group_by_type.groups:
    data = group_by_type.get_group(name)
    color = group_setting[name][1]
    alpha = group_setting[name][2]

    axes[0, 0].scatter(data["ranked"], data["popularity"], marker=".", color=color, alpha=alpha, label=name)
    axes[0, 1].scatter(data["ranked"], data["episodes"], marker=".", color=color, alpha=alpha)
    axes[1, 0].scatter(data["ranked"], data["favorites"], marker=".", color=color, alpha=alpha)
    axes[1, 1].scatter(data["ranked"], data["members"], marker=".", color=color, alpha=alpha )

    # Customizing plot
    axes[0, 0].set_ylabel("Popularity")
    axes[0, 1].set_ylabel("Episodes")
    axes[1, 0].set_ylabel("Favorites")
    axes[1, 1].set_ylabel("Members")

fig.legend()
plt.savefig("../charts/Scatter_plot_matrices.png")
plt.show()

```


    
![png](charts/Scatter_plot_matrices.png)
    


### Anime types



```python
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Anime types", fontsize=18)

# Bar chart
data = anime["type"].value_counts()
for i in data.index:
    axes[0].bar(i, data[i])

axes[0].set_ylabel("Count")

# Pie chart
axes[1].pie(data, explode=(0, 0.1, 0, 0, 0, 0), autopct="%1.1f%%")

plt.savefig("../charts/Anime_types.png")
plt.show()

```


    
![png](charts/Anime_types.png)
    


### Anime sources



```python
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Anime sources", fontsize=18)

# Bar chart
data = anime["source"].value_counts()
for i in data.index:
    axes[0].bar(i, data[i])

axes[0].tick_params(axis="x", rotation=90)
axes[0].set_ylabel("Count")

# Pie chart
axes[1].pie(data, autopct="%1.1f%%")

plt.savefig("../charts/Anime_sources.png")
plt.show()

```


    
![png](charts/Anime_sources.png)
    


### Anime ratings



```python
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Anime ratings", fontsize=18)

# Bar chart
data = anime["rating"].value_counts()
for i in data.index:
    axes[0].bar(i, data[i])

axes[0].tick_params(axis="x", rotation=90)
axes[0].set_ylabel("Count")

# Pie chart
axes[1].pie(data, autopct="%1.1f%%")

plt.savefig("../charts/Anime_ratings.png")
plt.show()

```


    
![png](charts/Anime_ratings.png)
    


## Frequent Pattern Mining

Now, to find repetitive favorite anime's, we can use frequent patterns, find patterns and use them to predict what anime each user may watch.
Here we use the Spark library and FPGrowth algorithm and its guide, which is as follows:

- `minSupport`: the minimum support for an itemset to be identified as frequent. For example, if an item appears 3 out of 5 transactions, it has a support of 3/5=0.6.
- `minConfidence`: minimum confidence for generating Association Rule. Confidence is an indication of how often an association rule has been found to be true. For example, if in the transactions itemset X appears 4 times, X and Y co-occur only 2 times, the confidence for the rule `X => Y` is then 2/4 = 0.5. The parameter will not affect the mining for frequent itemsets, but specify the minimum confidence for generating.
- `association` rules from frequent itemsets.
  numPartitions: the number of partitions used to distribute the work. By default the param is not set, and number of partitions of the input dataset is used.


### Initialize Spark Library



```python
spark = (
    SparkSession.builder.master("local")
    .appName("anime_recommendation_system")
    .getOrCreate()
)
sc = spark.sparkContext

```


```python
# Create a map file
map_file = anime[["anime_id", "name"]]
map_file.set_index("anime_id", inplace=True)
map_file.to_csv("../data/frequent-pattern/map_file.csv")

```

### Prepare Data


We create a itemset for each user whose favorite anime is in these baskets.



```python
# Group rows with user_id and then convert anime_id to list
df = mergedata.groupby("user_id")["anime_id"].apply(list)
# Write to csv file
df.to_csv("../data/frequent-pattern/itemset.csv")
df

```




    user_id
    0        [199, 164, 431, 578, 2236, 121, 2034, 2762, 15...
    1        [5114, 9253, 11061, 28851, 32281, 199, 19, 232...
    2        [9253, 11061, 2904, 263, 1575, 1535, 30276, 32...
    3        [9253, 32281, 2904, 1, 17074, 23273, 1575, 103...
    4        [2904, 1575, 1535, 1698, 2685, 1142, 3091, 422...
                                   ...                        
    24996                                               [3470]
    24997    [2904, 199, 1, 1575, 164, 431, 1535, 32, 5, 30...
    24998    [5114, 38524, 11061, 32935, 37510, 263, 34599,...
    24999    [5114, 9253, 11061, 32281, 199, 1, 164, 245, 4...
    25000    [32281, 16782, 19111, 1689, 31953, 15051, 3537...
    Name: anime_id, Length: 21804, dtype: object



Then read itemset and remove double quotation from itemset



```python
# Read csv file
itemset = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("../data/frequent-pattern/itemset.csv")
)

# Remove double quotes, brackets and cast to integer array
itemset = itemset.withColumn("anime_id", regexp_replace(col("anime_id"), '"|\[|\]', ""))
itemset = itemset.withColumn("anime_id", split(col("anime_id"), ",").cast("array<int>"))

print(itemset.printSchema())
itemset.show(5)

```

    root
     |-- user_id: integer (nullable = true)
     |-- anime_id: array (nullable = true)
     |    |-- element: integer (containsNull = true)
    
    None
    +-------+--------------------+
    |user_id|            anime_id|
    +-------+--------------------+
    |      0|[199, 164, 431, 5...|
    |      1|[5114, 9253, 1106...|
    |      2|[9253, 11061, 290...|
    |      3|[9253, 32281, 290...|
    |      4|[2904, 1575, 1535...|
    +-------+--------------------+
    only showing top 5 rows
    
    

Read map_file from disk



```python
# Read csv file
map_file = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("../data/frequent-pattern/map_file.csv")
)
map_file.show(5)

```

    +--------+--------------------+
    |anime_id|                name|
    +--------+--------------------+
    |    5114|Fullmetal Alchemi...|
    |   40028|Shingeki no Kyoji...|
    |    9253|         Steins;Gate|
    |   38524|Shingeki no Kyoji...|
    |   28977|            Gintama°|
    +--------+--------------------+
    only showing top 5 rows
    
    

### FPGrowth Algorithm


Create FPGrowth model from itemset



```python
fpGrowth = FPGrowth(itemsCol="anime_id", minSupport=0.1, minConfidence=0.8)
model = fpGrowth.fit(itemset)

```

Display frequent itemsets



```python
freqItemsets = model.freqItemsets.withColumn("item_id", monotonically_increasing_id())
freqItemsets.show(5)

```

    +--------------+----+-------+
    |         items|freq|item_id|
    +--------------+----+-------+
    |         [223]|2679|      0|
    |       [21881]|2560|      1|
    |[21881, 11757]|2188|      2|
    |       [37510]|3319|      3|
    | [37510, 5114]|2350|      4|
    +--------------+----+-------+
    only showing top 5 rows
    
    


```python
print("Number of frequent item set :", freqItemsets.count())

```

    Number of frequent item set : 2477
    

Convert items id to anime name and save them



```python
# Merge freqItemsets and map_file to get name of anime's
freqItemsets = (
    freqItemsets.select("item_id", explode("items").alias("anime_id"))
    .join(map_file, "anime_id")
    .groupBy("item_id")
    .agg(collect_list(struct("name")).alias("items"))
    .join(freqItemsets.drop("items"), "item_id")
    .drop("item_id")
)
# Convert Spark DataFrame to Pandas DataFrame for saving in csv file
freqItemsets.toPandas().to_csv("../data/frequent-pattern/freqItemsets.csv")
freqItemsets.show(5)

```

    +--------------------+----+
    |               items|freq|
    +--------------------+----+
    |     [{Dragon Ball}]|2679|
    |[{Sword Art Onlin...|2560|
    |[{Sword Art Onlin...|2188|
    |[{Mob Psycho 100 ...|3319|
    |[{Mob Psycho 100 ...|2350|
    +--------------------+----+
    only showing top 5 rows
    
    

Display association rules



```python
# Display generated association rules.
associationRules = model.associationRules.withColumn("item_id", monotonically_increasing_id())
associationRules.show()

```

    +--------------------+----------+------------------+------------------+-------------------+-------+
    |          antecedent|consequent|        confidence|              lift|            support|item_id|
    +--------------------+----------+------------------+------------------+-------------------+-------+
    | [11061, 2904, 1575]|    [5114]|0.8343217197924389| 2.042846802734906| 0.1032379379930288|      0|
    |[2904, 30276, 511...|    [1575]|0.9611158072696534| 2.532467560327193| 0.1042927903137039|      1|
    |       [28851, 9253]|   [32281]|0.8306863301191152|2.3595993671075024|0.13433314988075581|      2|
    |[38524, 38000, 16...|   [35760]|0.8943056124539124| 4.506457031186758|0.10011924417538066|      3|
    |[30276, 1575, 16498]|    [1535]|0.8014911463187325| 1.643844695168248|0.11832691249312052|      4|
    |[30276, 1575, 16498]|    [2904]|0.8993476234855545|2.5873301995618196|0.13277380297193175|      5|
    |       [31240, 1535]|   [16498]|0.8017391304347826|1.8257044386422978|0.12685745734727574|      6|
    | [6547, 1575, 16498]|    [2904]|0.9199406968124537| 2.646574344016195|0.11383232434415703|      7|
    |       [2904, 16498]|    [1575]|0.9522244137628753|2.5090394099922335|0.19927536231884058|      8|
    | [13601, 2904, 9253]|    [1575]|0.9561328790459966|2.5193379208119526|0.10296275912676574|      9|
    |               [523]|     [199]|0.8210137275607181| 2.363218919568831|0.14263437901302514|     10|
    |      [22535, 31240]|   [30276]|0.8005865102639296|2.1534651208727755|0.10016510731975785|     11|
    |      [22535, 31240]|   [16498]|0.8203812316715543| 1.868155861657083|0.10264171711612548|     12|
    |       [10087, 5114]|   [11741]|0.8808364365511315| 3.964855008786307| 0.1410291689598239|     13|
    |       [32937, 1535]|   [30831]| 0.887374749498998| 4.030899799599198| 0.1015410016510732|     14|
    | [11741, 5114, 1535]|   [10087]|0.9165668662674651| 3.914754936747465|0.10530177949000183|     15|
    |      [32935, 20583]|   [28891]| 0.963907284768212| 5.529343445694842|0.13350761328196661|     16|
    |       [19815, 1575]|    [2904]|0.9044759825327511|2.6020839587206894|0.15199046046596953|     17|
    |       [31964, 2904]|   [30276]|0.8052631578947368|2.1660446452919864|0.10525591634562466|     18|
    |       [31964, 2904]|    [1575]|0.9540350877192982|2.5138103991095564|0.12470188956154835|     19|
    +--------------------+----------+------------------+------------------+-------------------+-------+
    only showing top 20 rows
    
    


```python
print("Number of association rules :", associationRules.count())

```

    Number of association rules : 903
    

Convert items id to anime name and save them



```python
# Merge associationRules and map_file based on antecedent column to get name of anime's
associationRules = (
    associationRules.select("item_id", explode("antecedent").alias("anime_id"))
    .join(map_file, "anime_id")
    .groupBy("item_id")
    .agg(collect_list(struct("name")).alias("antecedent"))
    .join(associationRules.drop("antecedent"), "item_id")
)
# Merge associationRules and map_file based on consequent column to get name of anime's
associationRules = (
    associationRules.select("item_id", explode("consequent").alias("anime_id"))
    .join(map_file, "anime_id")
    .groupBy("item_id")
    .agg(collect_list(struct("name")).alias("consequent"))
    .join(associationRules.drop("consequent"), "item_id")
    .drop("item_id")
)
associationRules.toPandas().to_csv("../data/frequent-pattern/associationRules.csv")
associationRules.show()

```

    +--------------------+--------------------+------------------+------------------+-------------------+
    |          consequent|          antecedent|        confidence|              lift|            support|
    +--------------------+--------------------+------------------+------------------+-------------------+
    |[{Fullmetal Alche...|[{Hunter x Hunter...|0.8343217197924389| 2.042846802734906| 0.1032379379930288|
    |[{Code Geass: Han...|[{Code Geass: Han...|0.9611158072696534| 2.532467560327193| 0.1042927903137039|
    |  [{Kimi no Na wa.}]|[{Koe no Katachi}...|0.8306863301191152|2.3595993671075024|0.13433314988075581|
    |[{Shingeki no Kyo...|[{Shingeki no Kyo...|0.8943056124539124| 4.506457031186758|0.10011924417538066|
    |      [{Death Note}]|[{One Punch Man},...|0.8014911463187325| 1.643844695168248|0.11832691249312052|
    |[{Code Geass: Han...|[{One Punch Man},...|0.8993476234855545|2.5873301995618196|0.13277380297193175|
    |[{Shingeki no Kyo...|[{Re:Zero kara Ha...|0.8017391304347826|1.8257044386422978|0.12685745734727574|
    |[{Code Geass: Han...|[{Angel Beats!}, ...|0.9199406968124537| 2.646574344016195|0.11383232434415703|
    |[{Code Geass: Han...|[{Code Geass: Han...|0.9522244137628753|2.5090394099922335|0.19927536231884058|
    |[{Code Geass: Han...|[{Psycho-Pass}, {...|0.9561328790459966|2.5193379208119526|0.10296275912676574|
    |[{Sen to Chihiro ...|[{Tonari no Totoro}]|0.8210137275607181| 2.363218919568831|0.14263437901302514|
    |   [{One Punch Man}]|[{Kiseijuu: Sei n...|0.8005865102639296|2.1534651208727755|0.10016510731975785|
    |[{Shingeki no Kyo...|[{Kiseijuu: Sei n...|0.8203812316715543| 1.868155861657083|0.10264171711612548|
    |[{Fate/Zero 2nd S...|[{Fate/Zero}, {Fu...|0.8808364365511315| 3.964855008786307| 0.1410291689598239|
    |[{Kono Subarashii...|[{Kono Subarashii...| 0.887374749498998| 4.030899799599198| 0.1015410016510732|
    |       [{Fate/Zero}]|[{Fate/Zero 2nd S...|0.9165668662674651| 3.914754936747465|0.10530177949000183|
    |[{Haikyuu!! Secon...|[{Haikyuu!!: Kara...| 0.963907284768212| 5.529343445694842|0.13350761328196661|
    |[{Code Geass: Han...|[{No Game No Life...|0.9044759825327511|2.6020839587206894|0.15199046046596953|
    |   [{One Punch Man}]|[{Boku no Hero Ac...|0.8052631578947368|2.1660446452919864|0.10525591634562466|
    |[{Code Geass: Han...|[{Boku no Hero Ac...|0.9540350877192982|2.5138103991095564|0.12470188956154835|
    +--------------------+--------------------+------------------+------------------+-------------------+
    only showing top 20 rows
    
    

Display transform



```python
# transform examines the input items against all the association rules and summarize the
# consequents as prediction
transform = model.transform(itemset)
transform.show(10)

```

    +-------+--------------------+--------------------+
    |user_id|            anime_id|          prediction|
    +-------+--------------------+--------------------+
    |      0|[199, 164, 431, 5...|                  []|
    |      1|[5114, 9253, 1106...|[38524, 2904, 302...|
    |      2|[9253, 11061, 290...|       [5114, 31964]|
    |      3|[9253, 32281, 290...|[16498, 4181, 153...|
    |      4|[2904, 1575, 1535...|                  []|
    |      5|[199, 877, 4224, ...|                  []|
    |      6|[5114, 4181, 2904...|                  []|
    |      7|[5114, 4181, 199,...|                  []|
    |      8|[4181, 578, 10408...|                  []|
    |     10|   [1889, 934, 3652]|                  []|
    +-------+--------------------+--------------------+
    only showing top 10 rows
    
    


```python
print("Number of transform :", transform.count())

```

    Number of transform : 21804
    

Convert items id to anime name and save them



```python
# Merge transform and map_file based on prediction column to get name of anime's
transform = (
    transform.select("user_id", explode("prediction").alias("anime_id"))
    .join(map_file, "anime_id")
    .groupBy("user_id")
    .agg(collect_list(struct("name")).alias("prediction"))
    .join(transform.drop("prediction"), "user_id")
)
transform.toPandas().to_csv("../data/frequent-pattern/transform.csv")
transform.show(10)

```

    +-------+--------------------+--------------------+
    |user_id|          prediction|            anime_id|
    +-------+--------------------+--------------------+
    |      1|[{Shingeki no Kyo...|[5114, 9253, 1106...|
    |      2|[{Fullmetal Alche...|[9253, 11061, 290...|
    |      3|[{Shingeki no Kyo...|[9253, 32281, 290...|
    |     12|[{Code Geass: Han...|[5114, 199, 1575,...|
    |     13|[{Code Geass: Han...|[1575, 486, 30, 2...|
    |     14|[{Fullmetal Alche...|[9253, 38524, 110...|
    |     16|[{JoJo no Kimyou ...|[5114, 9253, 3228...|
    |     17|[{One Punch Man},...|[5114, 9253, 2897...|
    |     19|[{Death Note}, {B...|[5114, 9253, 3852...|
    |     21|[{Clannad: After ...|[9253, 28851, 322...|
    +-------+--------------------+--------------------+
    only showing top 10 rows
    
    

## Clustering

Now we want to categorize users who are similar in interests anime into the same clusters.

![clusters](https://miro.medium.com/max/1200/0*W4LYzCfTzYjMGgYz)


### Selecting number of k

We Guess number of clusters with silhouette score. Silhouette refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object has been classified. The silhouette value is a measure of how similar an object is to its own cluster compared to other clusters. [Wikipedia](<https://en.wikipedia.org/wiki/Silhouette_(clustering)>)



```python
scores = []
inertia_list = np.empty(10)
K = range(2, 10)

for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(to_cluster)
    inertia_list[k] = k_means.inertia_
    scores.append(silhouette_score(to_cluster, k_means.labels_))

```

Elbow Method



```python
plt.plot(range(0, 10), inertia_list, "-X")

plt.title("Elbow Method")
plt.xticks(np.arange(10))
plt.xlabel("Number of cluster")
plt.ylabel("Inertia")
# Draw vertical line in ax equal's 4
plt.axvline(x=4, color="blue", linestyle="--")

plt.savefig("../charts/Elbow_Method.png")
plt.show()

```


    
![png](charts/Elbow_Method.png)
    


Results KMeans



```python
plt.plot(K, scores)

plt.title("Results KMeans")
plt.xticks(np.arange(10))
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
plt.axvline(x=4, color="blue", linestyle="--")

plt.savefig("../charts/Results_KMeans.png")
plt.show()

```


    
![png](charts/Results_KMeans.png)
    


### K means clustering

Now that we have the number of clusters, we can use the KMeans algorithm.



```python
clusterer = KMeans(n_clusters=4, random_state=30).fit(to_cluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(to_cluster)

centers

```




    array([[ 1.6859081 , -1.67711621, -0.39729844],
           [-1.72324023,  0.12476707,  0.15343777],
           [ 8.06274357,  0.13311179,  1.11656004],
           [ 1.87270589,  2.52512675, -0.56430143]])



Data points in 3D PCA axis - clustered



```python
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection="3d")
ax.scatter(
    to_cluster[0],
    to_cluster[2],
    to_cluster[1],
    c=c_preds, # clusters
    cmap="viridis", # change color
    alpha=0.7, # Capacity
    s=24,
)

plt.title("Data points in 3D PCA axis - clustered", fontsize=18)
plt.savefig("../charts/Data_points_in_3D_PCA_axis_clustered.png")
plt.show()

```


    
![png](charts/Data_points_in_3D_PCA_axis_clustered.png)
    


Data points in 2D PCA axis - clustered



```python
plt.scatter(to_cluster[1], to_cluster[0], c=c_preds, cmap="viridis", alpha=0.7, s=24)

for ci, c in enumerate(centers):
    plt.plot(c[1], c[0], "X", markersize=8, color="red", alpha=1)

plt.title("Data points in 2D PCA axis - clustered", fontsize=18)
plt.xlabel("x_values")
plt.ylabel("y_values")

plt.savefig("../charts/Data_points_in_2D_PCA_axis_clustered.png")
plt.show()

```


    
![png](charts/Data_points_in_2D_PCA_axis_clustered.png)
    


Add cluster to user_anime DataFrame



```python
user_anime['cluster'] = c_preds
user_anime.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>"0"</th>
      <th>"Bungaku Shoujo" Kyou no Oyatsu: Hatsukoi</th>
      <th>"Bungaku Shoujo" Memoire</th>
      <th>"Bungaku Shoujo" Movie</th>
      <th>"Calpis" Hakkou Monogatari</th>
      <th>"Eiji"</th>
      <th>"Eiyuu" Kaitai</th>
      <th>"Kiss Dekiru Gyoza" x Mameshiba Movie</th>
      <th>"Parade" de Satie</th>
      <th>"R100" x Mameshiba Original Manners</th>
      <th>...</th>
      <th>s.CRY.ed Alteration II: Quan</th>
      <th>the FLY BanD!</th>
      <th>xxxHOLiC</th>
      <th>xxxHOLiC Kei</th>
      <th>xxxHOLiC Movie: Manatsu no Yoru no Yume</th>
      <th>xxxHOLiC Rou</th>
      <th>xxxHOLiC Shunmuki</th>
      <th>ēlDLIVE</th>
      <th>◯</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 11410 columns</p>
</div>




```python
user_anime.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21804 entries, 0 to 25000
    Columns: 11410 entries, "0" to cluster
    dtypes: int32(1), int64(11409)
    memory usage: 1.9 GB
    

## Characteristic of Each Cluster
In this section, we will calculate information for each cluster and print statistical data.



```python
c0 = user_anime[user_anime["cluster"] == 0].drop("cluster", axis=1).mean()
c1 = user_anime[user_anime["cluster"] == 1].drop("cluster", axis=1).mean()
c2 = user_anime[user_anime["cluster"] == 2].drop("cluster", axis=1).mean()
c3 = user_anime[user_anime["cluster"] == 3].drop("cluster", axis=1).mean()

```

### Functions


Create anime information list



```python
def createAnimeInfoList(animelist):
    genre_list = list()
    episode_list = list()
    score_list = list()
    member_list = list()
    popularity_list = list()
    favorites_list = list()

    for x in anime["name"]:
        if x in animelist:
            for y in anime[anime["name"] == x].genres.values:
                genre_list.append(y)

            episode_list.append(anime[anime["name"] == x].episodes.values.astype(int))
            score_list.append(anime[anime["name"] == x].score.values.astype(float))
            member_list.append(anime[anime["name"] == x].members.values.astype(int))
            popularity_list.append(anime[anime["name"] == x].popularity.values.astype(int))
            favorites_list.append(anime[anime["name"] == x].favorites.values.astype(int))

    # Return data in pandas series form to prevent
    # "the Length of Values error does not match the length of the index"
    return (
        pd.Series(genre_list),
        pd.Series(episode_list),
        pd.Series(score_list),
        pd.Series(member_list),
        pd.Series(popularity_list),
        pd.Series(favorites_list),
    )

```

Count word



```python
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste:
        keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split(","):
        if type(liste_keywords) == float and pd.isnull(liste_keywords):
            continue
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s):
                keyword_count[s] += 1
    # ______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key=lambda x: x[1], reverse=True)
    return keyword_occurences, keyword_count

```

Make cloud graph



```python
def makeCloud(Dict, name, color, isSave=True):
    words = dict()

    for s in Dict:
        words[s[0]] = s[1]

        wordcloud = WordCloud(
            width=1500,
            height=500,
            background_color=color,
            max_words=20,
            max_font_size=500,
            normalize_plurals=False,
        )
        wordcloud.generate_from_frequencies(words)

    fig = plt.figure(figsize=(20, 8))
    plt.title(name, fontsize=18)
    plt.imshow(wordcloud)
    plt.axis("off")

    if isSave:
        plt.savefig(f"../charts/{name}.png")
    plt.show()

```

### Calculate all genre keywords



```python
animelist = list(c0.index)
data = pd.DataFrame()
data["genre"] = createAnimeInfoList(animelist)[0]

```


```python
set_keywords = set()
for liste_keywords in data["genre"].str.split(",").values:
    if isinstance(liste_keywords, float):
        continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)

```

### Cluster 0


Top 15 anime which will explain characteristic of this cluster



```python
c0.sort_values(ascending=False)[0:15]

```




    Shingeki no Kyojin                       0.751535
    One Punch Man                            0.732706
    Kimi no Na wa.                           0.707532
    Death Note                               0.640196
    Koe no Katachi                           0.634056
    Boku no Hero Academia 2nd Season         0.623823
    No Game No Life                          0.622391
    Re:Zero kara Hajimeru Isekai Seikatsu    0.615227
    Fullmetal Alchemist: Brotherhood         0.614409
    Boku no Hero Academia                    0.609906
    Shingeki no Kyojin Season 2              0.609087
    Steins;Gate                              0.591281
    Shigatsu wa Kimi no Uso                  0.544003
    Kimetsu no Yaiba                         0.543185
    Boku dake ga Inai Machi                  0.528244
    dtype: float64



Favorite genre for this cluster



```python
c0_animelist = list(c0.sort_values(ascending=False)[0:30].index)
c0_data = pd.DataFrame()
(
    c0_data["genre"],
    c0_data["episode"],
    c0_data["score"],
    c0_data["member"],
    c0_data["popularity"],
    c0_data["favorites"],
) = createAnimeInfoList(c0_animelist)

keyword_occurences, dum = count_word(c0_data, "genre", set_keywords)
makeCloud(keyword_occurences[0:10], "Cluster_0", "lemonchiffon")

```


    
![png](charts/Cluster_0.png)
    



```python
keyword_occurences[0:5]

```




    [['Action', 17],
     [' Shounen', 16],
     [' Comedy', 12],
     [' Drama', 11],
     [' Super Power', 11]]



Average of each information for anime which user in this cluster like



```python
avg_episodes = int(c0_data["episode"].mean()[0].round())
avg_score = c0_data["score"].mean()[0].round(2)
avg_popularity = int(c0_data["popularity"].mean()[0].round())
avg_member = int(c0_data["member"].mean()[0].round())
avg_favorites = int(c0_data["favorites"].mean()[0].round())

print(f"Cluster 0\nAVG episode : {avg_episodes}\nAVG score : {avg_score}\nAVG popularity : {avg_popularity}\nAVG member : {avg_member}\nAVG favorites : {avg_favorites}\n")

```

    Cluster 0
    AVG episode : 23
    AVG score : 8.55
    AVG popularity : 25
    AVG member : 1562339
    AVG favorites : 59740
    
    

### Cluster 1


Top 15 anime which will explain characteristic of this cluster



```python
c1.sort_values(ascending=False)[0:15]

```




    Death Note                            0.371123
    Shingeki no Kyojin                    0.265868
    Fullmetal Alchemist: Brotherhood      0.251081
    Sen to Chihiro no Kamikakushi         0.243270
    Code Geass: Hangyaku no Lelouch       0.229013
    Code Geass: Hangyaku no Lelouch R2    0.202017
    Steins;Gate                           0.196557
    Toradora!                             0.192386
    One Punch Man                         0.187912
    Kimi no Na wa.                        0.184652
    Angel Beats!                          0.184197
    Howl no Ugoku Shiro                   0.165314
    Fullmetal Alchemist                   0.160158
    Sword Art Online                      0.158641
    Elfen Lied                            0.151513
    dtype: float64



Favorite genre for this cluster



```python
c1_animelist = list(c1.sort_values(ascending=False)[0:30].index)
c1_data = pd.DataFrame()
(
    c1_data["genre"],
    c1_data["episode"],
    c1_data["score"],
    c1_data["member"],
    c1_data["popularity"],
    c1_data["favorites"],
) = createAnimeInfoList(c1_animelist)

keyword_occurences, dum = count_word(c1_data, "genre", set_keywords)
makeCloud(keyword_occurences[0:10], "Cluster_1", "lemonchiffon")

```


    
![png](charts/Cluster_1.png)
    



```python
keyword_occurences[0:5]

```




    [[' Drama', 17],
     ['Action', 16],
     [' Supernatural', 12],
     [' Comedy', 10],
     [' Adventure', 8]]



Average of each information for anime which user in this cluster like



```python
avg_episodes = int(c1_data["episode"].mean()[0].round())
avg_score = c1_data["score"].mean()[0].round(2)
avg_popularity = int(c1_data["popularity"].mean()[0].round())
avg_member = int(c1_data["member"].mean()[0].round())
avg_favorites = int(c1_data["favorites"].mean()[0].round())

print(f"Cluster 1\nAVG episode : {avg_episodes}\nAVG score : {avg_score}\nAVG popularity : {avg_popularity}\nAVG member : {avg_member}\nAVG favorites : {avg_favorites}\n")

```

    Cluster 1
    AVG episode : 27
    AVG score : 8.44
    AVG popularity : 35
    AVG member : 1498743
    AVG favorites : 62681
    
    

### Cluster 2


Top 15 anime which will explain characteristic of this cluster



```python
c2.sort_values(ascending=False)[0:15]

```




    No Game No Life                          0.863184
    Shingeki no Kyojin                       0.846600
    One Punch Man                            0.844942
    Steins;Gate                              0.843284
    Angel Beats!                             0.825871
    Toradora!                                0.812604
    Re:Zero kara Hajimeru Isekai Seikatsu    0.796849
    Code Geass: Hangyaku no Lelouch          0.792703
    Fullmetal Alchemist: Brotherhood         0.786070
    Code Geass: Hangyaku no Lelouch R2       0.770315
    Kimi no Na wa.                           0.764511
    Death Note                               0.762852
    Hataraku Maou-sama!                      0.762023
    Boku dake ga Inai Machi                  0.758706
    Shokugeki no Souma                       0.753731
    dtype: float64



Favorite genre for this cluster



```python
c2_animelist = list(c2.sort_values(ascending=False)[0:30].index)
c2_data = pd.DataFrame()
(
    c2_data["genre"],
    c2_data["episode"],
    c2_data["score"],
    c2_data["member"],
    c2_data["popularity"],
    c2_data["favorites"],
) = createAnimeInfoList(c2_animelist)

keyword_occurences, dum = count_word(c2_data, "genre", set_keywords)
makeCloud(keyword_occurences[0:10], "Cluster_2", "lemonchiffon")

```


    
![png](charts/Cluster_2.png)
    



```python
keyword_occurences[0:5]

```




    [[' Supernatural', 14],
     ['Action', 13],
     [' Comedy', 12],
     [' Drama', 11],
     [' School', 11]]



Average of each information for anime which user in this cluster like



```python
avg_episodes = int(c2_data["episode"].mean()[0].round())
avg_score = c2_data["score"].mean()[0].round(2)
avg_popularity = int(c2_data["popularity"].mean()[0].round())
avg_member = int(c2_data["member"].mean()[0].round())
avg_favorites = int(c2_data["favorites"].mean()[0].round())

print(f"Cluster 2\nAVG episode : {avg_episodes}\nAVG score : {avg_score}\nAVG popularity : {avg_popularity}\nAVG member : {avg_member}\nAVG favorites : {avg_favorites}\n")

```

    Cluster 2
    AVG episode : 19
    AVG score : 8.44
    AVG popularity : 33
    AVG member : 1481909
    AVG favorites : 55440
    
    

### Cluster 3


Top 15 anime which will explain characteristic of this cluster



```python
c3.sort_values(ascending=False)[0:15]

```




    Code Geass: Hangyaku no Lelouch       0.692277
    Death Note                            0.668911
    Sen to Chihiro no Kamikakushi         0.659406
    Fullmetal Alchemist: Brotherhood      0.651089
    Code Geass: Hangyaku no Lelouch R2    0.629703
    Steins;Gate                           0.627723
    Tengen Toppa Gurren Lagann            0.584158
    Toradora!                             0.580198
    Bakemonogatari                        0.567129
    Mononoke Hime                         0.560396
    Shingeki no Kyojin                    0.544950
    Suzumiya Haruhi no Yuuutsu            0.539802
    Cowboy Bebop                          0.539406
    Mahou Shoujo Madoka★Magica            0.538218
    Toki wo Kakeru Shoujo                 0.535842
    dtype: float64



Favorite genre for this cluster



```python
c3_animelist = list(c3.sort_values(ascending=False)[0:30].index)
c3_data = pd.DataFrame()
(
    c3_data["genre"],
    c3_data["episode"],
    c3_data["score"],
    c3_data["member"],
    c3_data["popularity"],
    c3_data["favorites"],
) = createAnimeInfoList(c3_animelist)

keyword_occurences, dum = count_word(c3_data, "genre", set_keywords)
makeCloud(keyword_occurences[0:10], "Cluster_3", "lemonchiffon")

```


    
![png](charts/Cluster_3.png)
    



```python
keyword_occurences[0:5]

```




    [[' Drama', 16],
     ['Action', 16],
     [' Sci-Fi', 10],
     [' Supernatural', 10],
     [' Comedy', 9]]



Average of each information for anime which user in this cluster like



```python
avg_episodes = int(c3_data["episode"].mean()[0].round())
avg_score = c3_data["score"].mean()[0].round(2)
avg_popularity = int(c3_data["popularity"].mean()[0].round())
avg_member = int(c3_data["member"].mean()[0].round())
avg_favorites = int(c3_data["favorites"].mean()[0].round())

print(f"Cluster 3\nAVG episode : {avg_episodes}\nAVG score : {avg_score}\nAVG popularity : {avg_popularity}\nAVG member : {avg_member}\nAVG favorites : {avg_favorites}\n")

```

    Cluster 3
    AVG episode : 21
    AVG score : 8.48
    AVG popularity : 66
    AVG member : 1214717
    AVG favorites : 52597
    
    

![The End](https://www.gladstonebrookes.co.uk/wp-content/uploads/2019/08/News-Gladstone-Brookes-PPI-%E2%80%93-The-Beginning-of-the-End-29th-August-2019.png)

