# Sentiment Analysis
This is the first big project that I've worked on during the course of machine learning. The dataset(Amazon fine food reviews) that I used was download from kaggel.

There are 3 files here 
1. Data Analysis
2. Sentiment Analysis using Summary and review
3. Sentiment Analysis using just the summary 


## Data Analysis
Data Analysis is a very crucial part before building a model since it helps us understand the data before hand. Data analysis helps us in doing feature engineering which in turn helps us building a better model.

Before the actual data analysis I did some preliminary checks which I like to do always. I checked for missing values and duplicates. There were some missing values but it wouldn't hamper our model so I left it like that. However there were many duplicated which I removed. From around 500k datapoints I had now around 396k datapoints.

After this I went on with my normal data analysis, there I found some interesting discoveries. Please go through the file to understand more about it.

## Sentiment Analysis using Summary and review
Since we would be doing sentiment analyis on the summanry and text part of the review, we would be needing only text summary and score of the review. 
The score would be modified to binary values since we are doing a binary classification. 
We will append the summary and text.

After this we would do various text processing tasks like removing special charecters, most of the punctuation stemming etc.

Then I had used 4 different types of feature engineering
* Bag of words 
* tf idf
* Mean weighted W2V
* Tf idf weighted W2V

[Nb: For W2V I had used Google's pre-trained W2V]

I had used a bunch of classifiers, almost all that I had learned. Best that I got was with Logistic Regression and tfidf feature engineering.

## Sentiment Analysis using just the summary 
Everything is same as the previous file except the fact that I didn't append the summary with the text. I just used the summary to determine the sentiment of the review.

Again here the best I got was Logistic Regression using tfidf 


I wanted to train my own Word to Vec model. However I couldn't do it since I had very little amounts of data. In the future if I get a chance, I would love to train my own W2V model.
