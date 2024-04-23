## Simple Linear Regression
```

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("salary_data.csv")
dataset.head()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test,y_test, color = "red")
plt.plot(X_test,y_pred, color = "blue")
plt.title("salary vs experience")
plt.xlabel("Years of exp")
plt.ylabel("salary")
plt.show()

```

## Multiple Linear regression

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Student_Performance.csv")

dataset.head()

X = dataset.drop(columns=['Performance Index'])
y = dataset['Performance Index']
X_encoded = pd.get_dummies(X, columns=['Extracurricular Activities', 'Sample Question Papers Practiced'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test['Hours Studied'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Hours Studied'], y_pred, color='red', label='Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Scatterplot for Hours Studied vs Performance Index')
plt.legend()
plt.show()

plt.plot(X_test['Previous Scores'], y_test, 'o', color='blue', label='Actual')
plt.plot(X_test['Previous Scores'], y_pred, 'o', color='red', label='Predicted')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.title('Line chart for Previous Scores vs Performance Index')
plt.legend()
plt.show()

plt.hist(y_test, color='blue', alpha=0.5, label='Actual' )
plt.hist(y_pred, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Performance Index')
plt.ylabel('Frequency')
plt.title('Histogram of Performance Index')
plt.legend()
plt.show()

```


## Time series analaysis

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# Loading dataset in pandas and creating sample dataset
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
data = np.random.randn(len(date_range))
df = pd.DataFrame(data, index=date_range, columns=['Value'])
# Resampling
monthly_data = df.resample('M').mean()
# Calculating Rolling mean
rolling_mean = df.rolling(window=7).mean()

#  Seasonal Decomposition
decomposition = seasonal_decompose(df['Value'], model='additive', period=30)
plt.figure(figsize=(12, 6))

# Original Data
plt.subplot(2, 2, 1)
plt.plot(df.index, df['Value'], label='Original Data', color='blue')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# Trend Component
plt.subplot(2, 2, 2)
plt.plot(decomposition.trend.index, decomposition.trend, label='Trend', color='green')
plt.title('Trend Component')
plt.xlabel('Date')
plt.ylabel('Trend')
plt.legend()

# Seasonal Component
plt.subplot(2, 2, 3)
plt.plot(decomposition.seasonal.index, decomposition.seasonal, label='Seasonal', color='red')
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Seasonal')
plt.legend()

# Residual Component
plt.subplot(2, 2, 4)
plt.plot(decomposition.resid.index, decomposition.resid, label='Residual', color='purple')
plt.title('Residual Component')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()

plt.tight_layout()
plt.show()

```

## Arima Model

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Generate Random Time Series Data
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
values = np.random.randint(0, 100, size=len(date_range))
data = pd.DataFrame({'date': date_range, 'value': values})

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['value'], color='blue')
plt.title('Random Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Step 2: Check for Stationarity
result = adfuller(data['value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')
if result[1] <= 0.05:
    print('Data is stationary (reject null hypothesis)')
else:
    print('Data is non-stationary (fail to reject null hypothesis)')

# Step 3: Plot Correlation and Auto-Correlation Charts
plt.figure(figsize=(10, 6))
plot_acf(data['value'], lags=20)
plt.title('Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['value'], lags=20)
plt.title('Partial Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

# Step 4: Construct ARIMA Model
# Assuming you have determined the order parameters (p, d, q) for ARIMA
p = 1
d = 1
q = 1

# Fit the ARIMA model
model = ARIMA(data['value'], order=(p, d, q))
fit_model = model.fit()

# Print model summary
print(fit_model.summary())

```



## Sentiment Analysis 

```
import re

def clean_text(text):
    # Remove punctuation and numbers using regular expressions
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    return cleaned_text

# Read the text file
with open('Modi_2019.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Clean the text
cleaned_text = clean_text(text)
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize

# Tokenize at sentence level
sent_tokens = sent_tokenize(cleaned_text)

# Tokenize at word level
word_tokens = word_tokenize(cleaned_text)
!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')

# POS Tagging
pos_tags = pos_tag(word_tokens)
from nltk.corpus import stopwords
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
porter_stemmer = PorterStemmer()

# Stemming
stemmed_words = [porter_stemmer.stem(word) for word in filtered_tokens]

import matplotlib.pyplot as plt

# Visualize original text vs cleaned text length
plt.figure(figsize=(10, 6))
plt.bar(['Original Text', 'Cleaned Text'], [len(text), len(cleaned_text)], color=['blue', 'green'])
plt.title('Length of Original Text vs. Cleaned Text')
plt.ylabel('Number of Characters')
plt.show()
from nltk.probability import FreqDist

# Frequency Distribution of Word Tokens
word_freq = FreqDist(word_tokens)
plt.figure(figsize=(10, 6))
word_freq.plot(30, title='Top 30 Most Common Words')
plt.show()


```



## Data Visual R

```
# Create data frame of students' information
students <- data.frame(
  Name = c("John", "Emily", "Michael", "Sophia", "Daniel", "Olivia", "William", "Ava", "James", "Isabella"),
  Subject = rep(c("Math", "Science", "English"), each = 10),
  Marks = sample(60:100, 30, replace = TRUE)  # Generating random marks for demonstration
)


# Plot boxplot for subjects
boxplot(Marks ~ Subject, data = students)
# View the data frame
View(students)

# Read the CSV file into a data frame
startup_data <- read.csv("startup_funding.csv")

# Display the top 5 rows of the data frame in table format
print(head(startup_data, n = 5))
# Load required libraries
library(ggplot2)

# Create a subset of data with top 5 categories
top_5_categories <- c("E-Tech", "Transportation", "E-commerce", "FinTech", "Fashion and Apparel")
subset_data <- subset(startup_data, Industry.Vertical %in% top_5_categories)

# Create a bar plot
ggplot(subset_data, aes(x = Industry.Vertical)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Count of Startups by Industry (Top 5 Categories)",
       x = "Industry",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Load required libraries
library(ggplot2)

# Create a subset of data with top 5 categories
top_5_categories <- c("E-Tech", "Transportation", "E-commerce", "FinTech", "Fashion and Apparel")
subset_data <- subset(startup_data, Industry.Vertical %in% top_5_categories)

# Calculate frequency of startups in each category
category_counts <- table(subset_data$Industry.Vertical)

# Create a pie chart
pie(category_counts, labels = names(category_counts), col = rainbow(length(category_counts)),
    main = "Distribution of Startups by Industry (Top 5 Categories)")

# Load required libraries
library(ggplot2)

# Create a subset of data with top 5 categories
top_5_categories <- c("E-Tech", "Transportation", "E-commerce", "FinTech", "Fashion and Apparel")
subset_data <- subset(startup_data, Industry.Vertical %in% top_5_categories)

# Create a box plot
ggplot(subset_data, aes(x = Industry.Vertical, y = as.numeric(gsub(",", "", Amount.in.USD)))) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Startups by Industry (Top 5 Categories)",
       x = "Industry",
       y = "Amount in USD") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Load required libraries
library(ggplot2)

# Select the top 10 data points
top_10_data <- head(startup_data, 10)

# Create a histogram for funding amounts across industries
ggplot(top_10_data, aes(x = as.numeric(gsub(",", "", Amount.in.USD)))) +
  geom_histogram(binwidth = 1000000, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Funding Amounts (Top 10 Data)",
       x = "Funding Amount (USD)",
       y = "Frequency") +
  theme_minimal()

# Load required libraries
library(ggplot2)

# Select the top 10 data points
top_10_data <- head(startup_data, 10)

# Create a scatter plot for location and industry
ggplot(top_10_data, aes(x = City..Location, y = Industry.Vertical)) +
  geom_point(color = "skyblue") +
  labs(title = "Top 10 Startups: Location vs Industry",
       x = "Location",
       y = "Industry") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability


# Load required libraries
library(ggplot2)
library(dplyr)

# Select the top 10 data points
top_10_data <- head(startup_data, 10)

# Create a dataframe for heatmap
heatmap_data <- top_10_data %>%
  group_by(City..Location, Industry.Vertical) %>%
  summarise(count = n())

# Create a heatmap for location and industry
ggplot(heatmap_data, aes(x = City..Location, y = Industry.Vertical, fill = count)) +
  geom_tile() +
  labs(title = "Heatmap of Top 10 Startups: Location vs Industry",
       x = "Location",
       y = "Industry",
       fill = "Count") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

```

## Data Visual Python

```

from sklearn import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df = pd.read_csv("startup_funding2021.csv")
# Bar Plot
top_5_sectors = df['Sector'].value_counts().nlargest(5)
plt.figure(figsize=(10, 6))
top_5_sectors.plot(kind='bar')
plt.title('Count of Startups by Sector (Top 5)')
plt.xlabel('Sector')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


#  Pie Chart of the stages of startups
plt.figure(figsize=(8, 8))
df['Stage'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Startup Stages')
plt.ylabel('')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Stage', y='Founded', data=df)
plt.title('Boxplot of Year Founded by Stage')
plt.xlabel('Stage')
plt.ylabel('Year Founded')
plt.xticks(rotation=45)
plt.show()
# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Founded'], bins=20, kde=True)
plt.title('Histogram of Year Founded')
plt.xlabel('Year Founded')
plt.ylabel('Frequency')
plt.show()

#Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HeadQuarter', y='Sector', data=df.head(10))
plt.title('Scatter Plot of HeadQuarter vs Sector')
plt.xlabel('HeadQuarters Location')
plt.ylabel('Sector')
plt.show()

# Heatmap

top_sectors = df['Sector'].value_counts().nlargest(10).index
top_hq = df['HeadQuarter'].value_counts().nlargest(10).index

# Filter the data
df_filtered = df[df['Sector'].isin(top_sectors) & df['HeadQuarter'].isin(top_hq)]

plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(df_filtered['HeadQuarter'], df_filtered['Sector']), annot=True, fmt=".0f", cmap="Blues")
plt.title('Heatmap of HeadQuarter vs Sector')
plt.xlabel('HeadQuarters Location')
plt.ylabel('Sector')
plt.show()



# Line chart
plt.figure(figsize=(10, 6))
df['Founded'].value_counts().sort_index().plot()
plt.title('Number of Startups Founded Over Years')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# CDF
# Extract 'Founded' data
founded_data = df['Founded'].dropna()  # Drop NaN values if any

sorted_data = np.sort(founded_data)

cumulative_prob = np.arange(len(sorted_data)) / len(sorted_data)

# Plot the CDF
plt.figure(figsize=(8, 6))
sns.lineplot(x=sorted_data, y=cumulative_prob)
plt.title('Cumulative Density Function (CDF) of Year Founded')
plt.xlabel('Year Founded')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()


```


 



