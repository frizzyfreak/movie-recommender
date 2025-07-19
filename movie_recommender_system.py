#!/usr/bin/env python
# coding: utf-8

# In[113]:


#importing liberay that we need in this project
import numpy as np
import pandas as pd
import ast


# In[114]:


#assign the varible to each data set to its easy to handle
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[115]:


#looking in the movies heading
movies.head(1)


# In[116]:


#it show us the heading of credits dataset
credits.head(1)


# In[117]:


#merging both dataset on bases of titles
#reassign the dataset name by movies and adding data in movies varible that we created already on start
movies = movies.merge(credits,on='title')


# In[118]:


#seeing the merge dataset head
movies.head(1)


# In[119]:


#Thing that we are only taking in data set and except we are deleting so over data will short and valuable as per need
#we use same movies varible that we assign earlier
#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[120]:


#seeing the dataset after take usefull things only 
movies.head()


# In[121]:


# from this onward we are creating new dataset with 3 table only
#movie_id/title/tags
#movies_id are same // title are same // but for tags we are merging (overview + geners + keywords + cast + Crew)
#the tags data are in weird format so we first simplyfy the data


# In[122]:


#we are finding any missing data from the set
movies.isnull().sum()


# In[123]:


#we have 3 missing data in overview 
#we are droping the missing data that we found inshort deleting the 3 missing data movies
movies.dropna(inplace=True)


# In[124]:


#now review that the missing data is avilable or not now afte 
movies.isnull().sum()


# In[125]:


#now we are finding any duplicate data from this set
movies.duplicated().sum()


# In[126]:


#seeing inside the genres colum so we can know how to filter that 
movies.iloc[0].genres


# In[127]:


#sorting the genres so we can simply read that by doing helper function by only read genres name 
#now the genres are string so we have to use we have to convert to integer 
#so in python we have ast module that we can use
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[128]:


movies['genres'].apply(convert)


# In[129]:


movies['genres'] = movies['genres'].apply(convert)


# In[130]:


movies.head()


# In[131]:


#same filtering system that we use in genres we add in keywords
#we are using same def function that we created on genres so we dont have to writ all that again
movies['keywords'] = movies['keywords'].apply(convert)


# In[132]:


movies.head()


# In[133]:


#now we are sorting the cast so we have to pick only top 3 actror and rest we have to delete 
#by the same function that we use on top both genres or keyword but adding some modification using counter we use that to shown top 3 actor on movies
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[134]:


movies['cast'].apply(convert3)


# In[135]:


#we are adding the new filter cast in the same movie function that we created
movies['cast'] = movies['cast'].apply(convert3)


# In[136]:


#cheking the new filter cast are showing in data 
movies.head()


# In[137]:


#chekcing the crew column to know how to filter
movies['crew'][0]


# In[138]:


#now we are only adding director in crew
#because most of fans watch movie by whom is directed 
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[139]:


movies['crew'].apply(fetch_director)


# In[140]:


#we are adding the new filter crew in the same movie function that we created
movies['crew'] = movies['crew'].apply(fetch_director)


# In[141]:


#cheking the new filter crew are appearing in data 
movies.head()


# In[142]:


#now we are cheking overview column so we can filter it by according to over need
movies['overview'][0]


# In[143]:


#overview colum  is string so we are converting in list because we can concatenate with other list
#now we are using lambda function to create a string into list 
#if we find x then it will split x
movies['overview'].apply(lambda x:x.split())


# In[144]:


#we are adding the new filter crew in the same movie function that we created
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[145]:


#cheking the new filter overview are appearing in data 
movies.head()


# In[146]:


# now we are removing the spaces between the two names becasue the suppse there is two person in the same staring name but ending name is differnet so
#if we remove the space the two same name sam is become indival because of its surname 
#and it will not confused or model


# In[147]:


#now we first applying on genres 
# now we are using lambda function
#we are use list comprehension i dot replace any places with nothing for i in x
#simmilarly the same we adding for other colum that we need to remove spaces 
movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[148]:


#now we remove space in genres,cast,crew,keywords so we can return update this on data with same function movie
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[149]:


#cheking the updates genres in dataset 
movies.head()


# In[150]:


#now the main task is we are creating the new tags and this colum we add this 4-5 pre existed colum
#we are concatenate the column and create the new column knwo as "TAG"
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[151]:


#now we check the new column that we created meriging 5 columns
movies.head()


# In[152]:


#now we get the tag column but we dont need the other column that we add in tags so we remove that
#for that we create a new data frame 
new_df = movies[['movie_id','title','tags']]


# In[153]:


#checking the new dataframe that we created new_df
new_df


# In[154]:


#now the tag column are list because we have to merge we created it list already but now we dont need list so we convert into string
#now we are doing to remove space is use lambda function
#x we have to join x in each space 
new_df['tags'].apply(lambda x:" ".join(x))


# In[155]:


#now we created a new tags so we have to add this tag in old tags function so it can appear in dataset
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[156]:


#cheking the new tag funtion that appear as string or not
new_df.head()


# In[157]:


#now lets see the tags function deeply so what we get for now
new_df['tags'][0]


# In[158]:


#now the tags are in some part is uppar case so are converting into lower case as per onlin suggestion
#now we are using lamda to lower the case 
#if we find x then x will lower case
new_df['tags'].apply(lambda x:x.lower())


# In[159]:


#now return we update this tags function to preexisted tags so it will appear on the data set
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[160]:


#now check the return data that the new tags are appear or not
new_df.head()


# In[161]:


#-------------------------------------------------------------------------------------------#
# we are doing this for because of multiple text with same meaning we have to gave them one name so it didnt confuse model after


# In[162]:


import nltk


# In[163]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[164]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[165]:


new_df['tags'].apply(stem)


# In[166]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[167]:


#-------------------------------------------------------------------------------------------#


# In[168]:


#now we are doing text vectorization
#now we recommend the movie on bases of tags 


# In[169]:


#eg1:
new_df['tags'][0]


# In[170]:


#eg2:
new_df['tags'][1]


# In[171]:


#so according to the eg1 and eg2 we have to find a way to comapare the text similarity of both the example and recommend it to user
#so our strategy is to convert the text into vector t
#and while doing text vectorization we didnt conside stop words for do this we use scikit-learn liberay we use countvectorizer


# In[172]:


#now we are extracting scikit learn liberay 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[173]:


#converting tags into array
cv.fit_transform(new_df['tags']).toarray()


# In[174]:


#after converting array we saw the size of array
cv.fit_transform(new_df['tags']).toarray().shape


# In[175]:


#and after that we are assign the varible to this function so we easily call and save the data 
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[176]:


#now we are checking the varible vector so it can show proper data or not 
vectors


# In[178]:


#now checking the most frequent 5000 words 
cv.get_feature_names_out()


# In[ ]:


#so now main problem with array is there are so many word with same meaning so we have to merge that and fix that issue 
#but that issue we fix in upar code using nltk libery and using stem method


# In[179]:


#now we are measuring the thita between vectore angle to recommendation
#we have libery for that in sikitlearn we are using that
from sklearn.metrics.pairwise import cosine_similarity


# In[180]:


cosine_similarity(vectors)


# In[181]:


cosine_similarity(vectors).shape


# In[182]:


#assign the varible to function so its easy to call
similarity = cosine_similarity(vectors)


# In[183]:


similarity[1]


# In[184]:


#now we creating a function to recommend the 5 movies from index
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] #its function that find index value from table
    distances = similarity[movie_index] #measuring distance of array
    # For next step we sort the list with emurate function so it convert with tuple 
    # Then it will reverse the list and then its will give 5 simmilar cordinate suggestion using lambda function
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6] 
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return movies_list


# In[192]:


output_movie= recommend('Orphan')


# In[ ]:


#to send movies list from here to frontend
#import pickle


# In[ ]:


#pickle.dump(new_df,open('movies.pkl','wb'))


# In[ ]:


#but because of silt problem streamlit doesnt supports pandas file so we have to convert our file into dictory to import to frontemd 
#pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:


#we dont have simialriy so get that we use simialrity pickle
#pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:


# # Add this cell in your notebook
# import pickle
# with open('output_movie.pkl', 'wb') as f:
#     pickle.dump(output_movie, f)
# print("output_movie saved successfully!")

