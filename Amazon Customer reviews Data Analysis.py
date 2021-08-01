#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import nltk
import string


# In[34]:


con = sqlite3.connect('D:\Data Analysis Projects Files\Amazon/database.sqlite')


# In[35]:


type(con)


# #### reading data from Sqlite database

# In[15]:


pd.read_sql_query("SELECT * FROM Reviews", con)


# In[ ]:





# #### reading some n number of rows, use LIMIT over ther

# In[13]:


pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)


# In[ ]:





# #### or we can also Load the dataset using pandas

# In[10]:


df = pd.read_csv('D:\Data Analysis Projects Files\Amazon/Reviews.csv')

print(df.shape)
df.head()


# In[8]:


df.shape


# 

# In[16]:


get_ipython().system('pip install TextBlob')


# In[17]:


from textblob import TextBlob


# In[18]:


TextBlob(df['Summary'][0]).sentiment.polarity


# In[ ]:


polarity=[]

for i in df['Summary']:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)   
    except:
        polarity.append(0)


# In[ ]:


len(polarity)


# In[24]:


data=df.copy()


# In[ ]:


data['polarity']=polarity


# In[9]:


data.head()


# In[ ]:


data['polarity'].nunique()


# ### EDA for the Positve sentences

# In[ ]:


data_positive = data[data['polarity']>0]


# In[37]:


data_positive.shape


# In[36]:


get_ipython().system('pip install wordcloud')


# In[38]:


from wordcloud import WordCloud, STOPWORDS


# In[39]:


stopwords=set(STOPWORDS)


# In[40]:


positive=data_positive[0:200000]


# In[41]:



total_text= (' '.join(data_positive['Summary']))


# In[42]:


len(total_text)


# In[43]:


total_text[0:10000]


# In[44]:


import re
total_text=re.sub('[^a-zA-Z]',' ',total_text)


# In[46]:


total_text[0:20000]


# In[47]:


total_text=re.sub(' +',' ',total_text)


# In[48]:


total_text[0:20000]


# In[49]:


len(total_text)


# In[50]:


wordcloud = WordCloud(width = 1000, height = 500,stopwords=stopwords).generate(total_text)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# In[ ]:





# ## EDA for the Negative sentences

# In[51]:


data_negative = data[data['polarity']<0]
data_negative.shape


# In[52]:


data_negative.head()


# In[53]:


total_negative= (' '.join(data_negative['Summary']))


# In[69]:


total_negative


# In[54]:


import re
total_negative=re.sub('[^a-zA-Z]',' ',total_negative)


# In[55]:


len(total_negative)


# In[56]:


total_negative


# In[57]:


total_negative=re.sub(' +',' ',total_negative)


# In[58]:


len(total_negative)


# In[59]:



wordcloud = WordCloud(width = 1000, height = 500,stopwords=stopwords).generate(total_negative)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Analyse Amazon recommendations

# #### Amazon can recommend more products to only those who are going to buy more or to one who has a better conversion rate.

# In[ ]:





# In[60]:


df['UserId'].shape


# In[61]:


df['UserId'].nunique()


# In[62]:


df.head()


# In[63]:


raw=df.groupby(['UserId']).agg({'Summary':'count', 'Text':'count','Score':'mean','ProductId':'count'}).sort_values(by='Text',ascending=False)
raw


# In[64]:


raw.columns=['Number_of_summaries','num_text','Avg_score','Number_of_products_purchased']
raw


# In[65]:


user_10=raw.index[0:10]
number_10=raw['Number_of_products_purchased'][0:10]

plt.bar(user_10, number_10, label='java developer')
plt.xlabel('User_Id')
plt.ylabel('Number of Products Purchased')
plt.xticks(rotation='vertical')


# #### These 10 users are the target audience to recommend more products to them.

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[66]:


df.head()


# In[67]:


final=df.sample(n=2000)


# In[68]:


final=df[0:2000]


# #### check missing values in dataset

# In[69]:


final.isna().sum()


# In[ ]:





# #### Removing the Duplicates if any

# In[70]:


final.duplicated().sum()


# In[ ]:





# ### Analyse Length of Comments 

# In[72]:


final.head()


# In[73]:


len(final['Text'][0].split(' '))


# In[74]:


final['Text'][0]


# In[75]:



def calc_len(text):
    return (len(text.split(' ')))


# In[76]:


final['Text_length']=final['Text'].apply(calc_len)


# In[78]:


get_ipython().system('pip install plotly')


# In[79]:





# In[80]:


import plotly.express as px
px.box(final, y="Text_length")


# #### Conclusion-->>
#     Seems to have Almost 50 percent users are going to give their Feedback limited to 50 words whereas there are only few users who are going give Lengthy Feedbacks

# In[ ]:





# In[ ]:





# #### Analyze Score

# In[81]:


sns.countplot(final['Score'], palette="plasma")


# In[ ]:





# ### Text Pre-Processsing

# In[ ]:





# In[82]:


final['Text'] =final['Text'].str.lower()
final.head(10)


# In[ ]:





# In[83]:


final['Text'][164]


# In[84]:


import re
re.sub('[^a-zA-Z]',' ',final['Text'][164])


# In[ ]:





# #### drawback of this re.sub - it will remove some numerical data too & may be that numerical values matters alot

# In[ ]:





# #### logic to remove punctuations or all the special characters

# In[85]:


# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

data= final['Text'][164]

# remove punctuation from the string
no_punct = ""
for char in data:
    if char not in punctuations:
        no_punct = no_punct + char

# display the unpunctuated string
no_punct


# In[ ]:





# #### Create function to remove punctuations in your review

# In[86]:


def remove_punc(review):
    import string
    punctuations =string.punctuation
    # remove punctuation from the string
    no_punct = ""
    for char in review:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


# In[87]:


final['Text'] =final['Text'].apply(remove_punc)


# In[199]:


final.head()


# In[88]:


final['Text'][164]


# In[97]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install stopwords')


# In[106]:


get_ipython().system('pip install stopwords')


# In[107]:


import nltk
nltk.download('stopwords')


# In[ ]:





# #### Removal of Stopwords

# In[108]:


import nltk
from nltk.corpus import stopwords


# In[103]:


review='seriously this product was as tasteless as they come there are much better tasting products out there but at 100 calories its better than a special k bar or cookie snack pack you just have to season it or combine it with something else to share the flavor'


# In[109]:


re=[word for word in review.split(' ') if word not in set(stopwords.words('english'))]
str=''
for wd in re:
    str=str+wd
    str=str+' '
str


# #### using join to convert list into string

# In[110]:


re=[word for word in review.split(' ') if word not in set(stopwords.words('english'))]
' '.join(re)


# In[ ]:





# 

# In[111]:


def remove_stopwords(review):
    return ' '.join([word for word in review.split(' ') if word not in set(stopwords.words('english'))])


# In[112]:


remove_stopwords(review)


# In[113]:


final.shape


# In[114]:


final.columns


# In[115]:


final['Text'] = final['Text'].apply(remove_stopwords)


# In[116]:


final.head()


# In[ ]:





# ### Preprocessing data

# #### check if urls is present in Text column or not

# In[117]:


final['Text'].str.contains('http?').sum()


# In[118]:


final['Text'].str.contains('http').sum()


# In[119]:


pd.set_option('display.max_rows',2000)
final['Text'].str.contains('http',regex=True)


# In[120]:


final['Text'][21]


# In[ ]:





# 

# ####  Removal of urls

# In[121]:


final['Text'][21]


# In[122]:


review=final['Text'][21]
review


# In[123]:


import re


# In[124]:


url_pattern = re.compile(r'href|http.\w+')
url_pattern.sub(r'', review)


# In[125]:


import re
def remove_urls(review):
    url_pattern = re.compile(r'href|http.\w+')
    return url_pattern.sub(r'', review)


# In[126]:


final['Text'] = final['Text'].apply(remove_urls)


# In[127]:


final.head()


# In[128]:


final['Text'].str.contains('http').sum()


# In[129]:


final['Text'][34]


# ##### Removing br

# In[130]:


final['Text'][34].replace('br','')


# In[131]:


for i in range(len(final['Text'])):
    final['Text'][i]=final['Text'][i].replace('br','')


# In[132]:


data2=final.copy()


# In[133]:


data2['Text'][34]


# In[134]:


data2.shape


# In[135]:


data2.dtypes


# In[ ]:




Advantages of Word Clouds :
Analyzing customer and employee feedback.
Identifying new SEO keywords to target.
# In[136]:


from wordcloud import WordCloud, STOPWORDS 


# In[137]:


stopwords = set(STOPWORDS)


# In[138]:


data2.head()


# In[139]:


comment_words = '' 
for val in data2['Text']:
    # typecaste each val to string
    
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words=comment_words+ " ".join(tokens)+" "
    


# In[141]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)


# In[142]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud) 
plt.axis("off")


# In[ ]:





# In[ ]:




