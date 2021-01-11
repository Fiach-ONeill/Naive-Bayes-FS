import pandas as pd
import numpy as np
import re
import itertools
from itertools import product
from sklearn import model_selection
from sklearn import metrics

file_name = 'movie_reviews.xlsx'
df = pd.read_excel(file_name, index_col=False)

#These undeclared variables with hold the the number of positive/negative reviews in the training/test dataset
test_sent_count = None
train_sent_count = None
train_df = None
test_df = None
alpha =  1

word_occurnces = input("Please enter the minimum number of word occurnces for features extraction: ")
word_occurnces = int(word_occurnces)
minimum_word_length = input("Please enter the minimum length for a word to be extracted: ")
minimum_word_length = int(minimum_word_length)


#This function includes pre-proccessig methods as listed in TASK2
def clean_data():
    df['Review'] = df['Review'].replace('[^a-zA-Z0-9 ]', '', regex=True).str.lower()#Remove non-alphanumeric characters and convert to lower-case
    df["Review"] = df["Review"].apply(lambda s: s.split())#Tokenize colum values

print("Cleaning data..")
clean_data()
print("Data cleaned")
#This function splits the data into respective test and train sets
def split_data():
    global train_sent_count
    global test_sent_count
    global train_df
    global test_df

    print("Splitting data..")
    train_df = df[df["Split"]=="train"]
    test_df = df[df["Split"]=="test"]
    X_train = train_df["Review"]
    y_train = train_df["Sentiment"]
    X_test = test_df["Review"]
    y_test = test_df["Sentiment"]
    print("Data Split")


    #Below prints out num of negative and positive reviews in both train and test datasets
    print("\nData for the test set: ")
    test_sent_count = test_df["Sentiment"].value_counts()
    print(test_sent_count)
    print("\nData for the train set: ")
    train_sent_count = train_df["Sentiment"].value_counts()
    print(train_sent_count)

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test = split_data()

#This function creates a vocabulary with words removed that do not meet the minimum requirements
#This function is from task2
def create_vocab(ds,min_word_len,min_word_occ):
    vocab = []
    words = {}#To be used for word occurence counts
    for review in ds:
        for word in review:
            if word not in words:#Build  word occurnce mapping
                words[word] = 1
            else:
                words[word] +=1
    for key in words:
        if len(key)>min_word_len and words[key]>min_word_occ:
            vocab.append(key)#Checks if fits minimum requirements for length and occurnce and adds to list
    return vocab

#This function is from task3
#This function counts the occurnces of words in both positive/negative reviews
def create_word_mappings(word_list):
    positive_word_mapping = {}#will count the amount of times a word comes up in positive review
    negative_word_mapping = {}#will count the amount of times a word comes up in a negative review
    y_train_mod = y_train.values
    X_train_mod = list(map(set,X_train))
    review_compare = zip(X_train_mod,y_train_mod)#Create a list of tuples to compare
    for w,review in itertools.product(word_list,review_compare):
        if review[1] == ("positive"):
            if w not in positive_word_mapping:
                positive_word_mapping[w]=0
            if w in review[0]:
                positive_word_mapping[w]+=1
        else:
            if w not in negative_word_mapping:
                negative_word_mapping[w]=0
            if w in review[0]:
                negative_word_mapping[w]+=1
    return positive_word_mapping, negative_word_mapping

#This function i in task 4
#function to calculate likelihood with laplace smoothing for both negative and positive
def likelihood(word_mapping,vocabulary):
    likelihood_mapping = {}
    word_count_review = sum(word_mapping.values())
    vocabulary = np.asarray(vocabulary)

    for w in vocabulary:
        w_count = word_mapping[w]
        len_vocab = len(vocabulary)
        lik = (w_count+alpha)/(word_count_review*alpha)#compute likelihood with laplace smoothing
        likelihood_mapping[w] = lik
    return likelihood_mapping

#calculates priors
def priors(y):
    count = y.value_counts()
    pos = count["positive"]
    neg = count["negative"]
    total=pos+neg
    neg_prior = neg/total
    pos_prior = pos/total
    return pos_prior,neg_prior

#This function applies bayesian formula for conditonal probability
def classify(review,positive_likelihood_mapping,negative_likelihood_mapping,pos_prior,neg_prior):
    review_mod = review
    #review_mod = re.sub("[^0-9a-zA-Z]+", " ", review)
    #review_mod = review_mod.lower().split()
    pos_score = pos_prior
    neg_score = neg_prior
    for w in review_mod:
        if w in positive_likelihood_mapping:
            pos_score*=positive_likelihood_mapping[w]
        if w in negative_likelihood_mapping:
            neg_score *= negative_likelihood_mapping[w]
    if pos_score>neg_score:
        return "positive"
    elif neg_score>pos_score:
        return "negative"
    else:
        print("Undetermined")

#This function is used for predicting the test data
def predict(ds,pwm,nwm,p_prior,n_prior):
    res = []
    for r in ds:
        pred = classify(r,pwm,nwm,p_prior,n_prior)
        res.append(pred)
    return res

#This function fit the data to the model for each iteration
def fit(data,target):
    vocab = create_vocab(data,minimum_word_length,word_occurnces)
    positive_word_mapping,negative_word_mapping = create_word_mappings(vocab)
    positive_likelihood_mapping = likelihood(positive_word_mapping,vocab)
    negative_likelihood_mapping = likelihood(negative_word_mapping,vocab)
    pos_prior,neg_prior = priors(target)
    return positive_word_mapping,negative_word_mapping,pos_prior,neg_prior


def train():
    print("Training, this might take a while..")
    data = df["Review"]
    target = df["Sentiment"]
    positives = len(data[target=="positive"])
    negatives = len(data[target=="negative"])
    #kf = model_selection.StratifiedKFold(n_splits=min(positives,negatives), shuffle=True)
    kf = model_selection.KFold(n_splits=5,shuffle=False)
    TP = 0
    FP =0
    TN = 0
    FN = 0
    for k in range(1,3):
        for train_index, test_index in kf.split(data):
            current_i_train = data.iloc[train_index]
            current_i_test = data.iloc[test_index]
            ver = target[test_index]
            print(len(ver))
            positive_word_mapping,negative_word_mapping,pos_prior,neg_prior = fit(current_i_train,target[train_index])
            predicted_y = predict(current_i_test,positive_word_mapping,negative_word_mapping,pos_prior,neg_prior)
            print(len(predicted_y))
            for i in range(len(predicted_y)):
                if ver.iloc[i]==predicted_y[i]=="positive":
                    TP+=1
                if predicted_y[i]=="positive" and ver.iloc[i]!=predicted_y[i]:
                    FP+=1
                if ver.iloc[i]==predicted_y[i]=="negative":
                    TN+=1
                if predicted_y[i]=="negative" and ver.iloc[i]!=predicted_y[i]:
                    FN+=1
        print("K:=",k)

    total = TP+TN+FP+FN

    TP_score = TP/total
    TN_score = TN/total
    FP_score = FP/total
    FN_score = FN/total

    T = TN+TP
    F = FN+FP

    T_score = T/total*100
    F_score = F/total*100

    print("True Positive Score: ",TP_score)
    print("True Negaitve Score: ",TN_score)
    print("False Positive Score: ",FP_score)
    print("False Negative Score:",FN_score)
    print()
    print("Accuracy: "+str(T_score)+"%")

train()
