from django.shortcuts import render


import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import swifter
from sklearn.model_selection import train_test_split

tokenizer = RegexpTokenizer(r'\w+')

STOP_WORDS = stopwords.words('english') # stop words
lemmatizer = WordNetLemmatizer()
df_business_restaurants=pd.read_csv('df_business_restaurants.csv')
df_review = pd.read_csv("review_data_state.csv")
# replace missing reviews with an empty string 
df_review["text"].fillna('', inplace=True)
print(df_review.head())
df_review = df_review[['user_id', 'business_id', 'stars', 'text']]
dict_items = {key: val for val, key in enumerate(df_review['business_id'].unique())}
inv_dict_items = {val: key for key, val in dict_items.items()}

dict_users = {key: val for val, key in enumerate(df_review['user_id'].unique())}
inv_dict_users = {val: key for key, val in dict_users.items()}
df_review['user_id'] = df_review['user_id'].swifter.apply(lambda x: dict_users[x])
df_review['business_id'] = df_review['business_id'].swifter.apply(lambda x: dict_items[x])
    
# separating the data into training and test
train_ratings_df, test_ratings_df = train_test_split(df_review, test_size = 0.2)

def text_preprocess(text):
    text = tokenizer.tokenize(text.lower()) # we convert the text to lowercase and split it into tokens by spaces and punctuation marks
    text = [re.sub('[^a-z\s]', '', w) for w in text]# get rid of numbers and non-Latin characters
    text = [lemmatizer.lemmatize(w) for w in text if w not in STOP_WORDS]# perform lemmatization and get rid of stop words
    return ' '.join(text)

def get_train_whole_text(df, name_id):
    return df[[name_id,'text']].groupby(name_id).agg({'text': ' '.join})

    
user_id_review_df = pd.DataFrame(index=inv_dict_users.keys(), columns=["text"])
user_id_review_df.index.name = 'user_id'
user_id_review_df['text'] =  get_train_whole_text(train_ratings_df, 'user_id')["text"]
user_id_review_df["text"].fillna('', inplace=True)
business_id_review_df= pd.DataFrame(index=inv_dict_items.keys(), columns=["text"])
business_id_review_df.index.name = 'business_id'
business_id_review_df['text'] =  get_train_whole_text(train_ratings_df, 'business_id')["text"]
business_id_review_df["text"].fillna('', inplace=True)

user_id_vectorizer = TfidfVectorizer(tokenizer = tokenizer.tokenize, max_features=5000)
user_id_vectors = user_id_vectorizer.fit_transform(user_id_review_df['text'])

business_id_vectorizer = TfidfVectorizer(tokenizer = tokenizer.tokenize, max_features=5000)
business_id_vectors = business_id_vectorizer.fit_transform(business_id_review_df['text'])

I = pd.DataFrame(business_id_vectors.toarray(), index=business_id_review_df.index, columns=business_id_vectorizer.get_feature_names())
I.sort_index(inplace=True)


# Create your views here.
def home(request):
    
    if request.method == 'POST':
        words = str(request.POST['words'])
        query = pd.DataFrame([text_preprocess(words)], columns=['text']) # create a dataframe with a given query
        query_vector = user_id_vectorizer.transform(query['text']) # selection of query features
        query = pd.DataFrame(query_vector.toarray(), index=query.index, columns=user_id_vectorizer.get_feature_names())

        predictRating=pd.DataFrame(np.dot(query.loc[0],I.T), index=I.index, columns=['rating'])
        topRecommendations=pd.DataFrame.sort_values(predictRating,['rating'],ascending=[0])[:6]

        number = 0
        name=[]
        city=[]
        state=[]
        category=[]
        avg_star=[]
        review_count=[]

        for i in topRecommendations.index:
            number+=1
            # print(f"\n№ {number}")
            name.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['name'].iloc[0])
            # print(f"Name: {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['name'].iloc[0]}")
            city.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['city'].iloc[0])
            state.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['state'].iloc[0])
            # print(f"City, state: {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['city'].iloc[0]}, {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['state'].iloc[0]}")
            category.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['categories'].iloc[0])
            # print(f"Category: {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['categories'].iloc[0]}")
            avg_star.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['stars'].iloc[0])
            # print(f"Average star: {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['stars'].iloc[0]}")
            review_count.append(df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['review_count'].iloc[0])
            # print(f"Review count: {df_business_restaurants[df_business_restaurants['business_id']==inv_dict_items[i]]['review_count'].iloc[0]}")

            #dictr={'name':name,'city':city,'state':state,'category':category,'avg_star':avg_star,'review_count':review_count}
            
            #print(dictr.name[0])
            #print(type(dictr))
        dictr={'name':name,'city':city,'state':state,'category':category,'avg_star':avg_star,'review_count':review_count}
        print(dictr)
        print(f"name:{dictr['name'][0]}")
        index=['1','2','3','4','5','6']

        mylist = zip(name, city,state,category,avg_star,review_count,index)
        context = {
            'mylist': mylist,
        }


        return render(request,'main.html',context)
    else:
        return render(request,'main.html')
    
def main(request):
    return render(request,'main.html')