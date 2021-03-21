###Importing libraries needed

#ML tools
import numpy as np
import pandas as pd

#ploting tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

#Natural Language processing tools
import nltk
nltk.download('vader_lexicon')
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Other tools
import datetime
import time
import math
import csv
import re
import string

#Loading and cleaning dataset of tweets
vaccine_tweets = []

with open("vaccineTweetsSAJan21+Feb21.csv",encoding='utf-8') as vaccineTweets:
    lineReader = csv.reader(vaccineTweets, delimiter=',', quotechar="\"")
    for row in lineReader:
        try:
            #read tweet
            tweet = row[1]
            #remove html links
            tweet = re.sub('http[s]?://\S+', '', tweet)
            #remove punctuation
            tweet = tweet.translate(str.maketrans('', '', string.punctuation))
            #change to lower case
            tweet = tweet.lower()
            vaccine_tweets.append(tweet)
        except:
            print("error reading tweet from csv")

#print(len(vaccine_tweets))

###Calculating sentiments

positiveSent = 0
negativeSent = 0
neutralSent = 0
polarity = 0
tweets = []
positive_tweets = []
negative_tweets = []
neutral_tweets = []

begin_time = datetime.datetime.now()

for tweet in vaccine_tweets:
    
    tweets.append(tweet)
    analysis = TextBlob(tweet)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity
    
    if neg > pos:
        negative_tweets.append(tweet)
        negativeSent += 1
    elif pos > neg:
        positive_tweets.append(tweet)
        positiveSent += 1

    elif pos == neg:
        neutral_tweets.append(tweet)
        neutralSent += 1

end_time = datetime.datetime.now()

actualTweets = len(tweets)
positiveSent = 100 * (float(positiveSent)/float(actualTweets))
negativeSent = 100 * (float(negativeSent)/float(actualTweets))
neutralSent = 100 * (float(neutralSent)/float(actualTweets))
polarity = 100 * (float(polarity)/float(actualTweets))
positiveSent = format(positiveSent, ".1f")
negativeSent = format(negativeSent, ".1f")
neutralSent = format(neutralSent, ".1f")

#print(end_time - begin_time)
#print(positiveSent, "\n", negativeSent, "\n", neutralSent, "\n", polarity)

###Making pie chart
labels = ['Positive ('+str(positiveSent)+'%)', 'Neutral ('+str(neutralSent)+'%)', 'Negative ('+str(negativeSent)+'%)']
sizes = [positiveSent, neutralSent, negativeSent]
colors = ['lightgreen', 'blue', 'red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title('Sentiment analysis results for vaccine tweets')
plt.axis('equal')
plt.savefig("pie.png")
plt.show()

###Making the word clouds
icon = Image.open("plus.png") 
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)

rgb_array_plus = np.array(image_mask) # converts the image object to an array

icon = Image.open("minus.png")
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)

rgb_array_minus = np.array(image_mask) # converts the image object to an array

stopwords = set(STOPWORDS)

word_cloud = WordCloud(mask=rgb_array_plus, background_color='white', stopwords = stopwords,
                      max_words=1000, colormap='winter').generate(''.join(positive_tweets))
plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#word_cloud.to_file("pos.png")

word_cloud = WordCloud(mask=rgb_array_minus, background_color='white', stopwords = stopwords,
                      max_words=1000, colormap='gist_heat').generate(''.join(negative_tweets))
plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#word_cloud.to_file("neg.png")