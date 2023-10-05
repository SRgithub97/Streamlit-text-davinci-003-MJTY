import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime
#from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image
import PIL


@st.cache_data
def load_data(path):
    df=pd.read_excel(path)
    return df

def sum_npn(df):
    sums=df[['Negative','Positive','Neual']].sum()
    piechart_df=pd.DataFrame(sums.values,columns=['value'])
    piechart_df['label']=sums.index.values
    return piechart_df


def week_sentiment(week,sentiment_timesdf):
    init_date=sentiment_timesdf.date.min()
    startdate=init_date+pd.Timedelta(days=7*(week-1))
    enddate=startdate+pd.Timedelta(days=7*week)
    sentiment_week=sentiment_timesdf.loc[(sentiment_timesdf['date'] >= startdate) & (sentiment_timesdf['date'] <= enddate)]
    return sentiment_week

def week_keywords(week,keywordslistdf):
    init_date=keywordslistdf.date.min()
    startdate=init_date+pd.Timedelta(days=7*(week-1))
    enddate=startdate+pd.Timedelta(days=7*week)
    keywords_week=keywordslistdf.loc[(keywordslistdf['date'] >= startdate) & (keywordslistdf['date'] <= enddate)]
    return keywords_week

@st.cache_data
def keywordsrawprocessing(keyworddf):
    keywordslist=[]
    for i in range(0,len(keyworddf)):
        k=keyworddf.loc[i,'keywords']
        words=re.split(',',k)
        keywordslist.append(words)   
    keywordslistdf=pd.DataFrame(keywordslist)
    keywordslistdf['date']=keyworddf['date']
    return keywordslistdf

def wordsprocessing(keywordslistdf):
    for i in range(0,keywordslistdf.shape[0]):
        for j in range(0,keywordslistdf.shape[1]):
            text=keywordslistdf.iloc[i,j]
            if text!=None:
                text=text.replace("‘",'')
                text=text.replace("’",'')
                text=text.replace(" ",'')
                text=text.replace("”",'')
                keywordslistdf.iloc[i,j]=text.strip()
            else:
                keywordslistdf.iloc[i,j]=''
    keywordsdict=keywordslistdf.melt().value.unique()
    return keywordslistdf,keywordsdict

def wordstimes(keywordslistdf,keywordsdict):
    times=[]
    for k in keywordsdict:
        time=0
        for i in range(0,keywordslistdf.shape[0]):
            for j in range(0,keywordslistdf.shape[1]):
                if keywordslistdf.iloc[i,j]==k:
                    time+=1
        times.append(time)

    keywords_timesdf=pd.DataFrame([keywordsdict,times]).T
    keywords_timesdf.columns=['keywords','times']
    index_names = keywords_timesdf[keywords_timesdf['keywords'] ==''].index
    keywords_timesdf.drop(index_names, inplace = True)
    return keywords_timesdf



## sentiment_classifier = pipeline('zero-shot-classification',model="facebook/bart-large-mnli")

st.set_page_config(layout='wide', page_title="云之羽 SMA")

sentiment_timesdf=load_data('./data/sentiment_analysis.xlsx')
keyworddf=load_data('./data/keywords_text-davinci-003.xlsx')
keywordslistdf=keywordsrawprocessing(keyworddf)


with st.sidebar:
    st.image("./img/logo.jpg",width=400)
    gpt_api_key=st.text_input(label="Please input your GPT API KEY:")
    st.write("Do you want to start keywords analysis?")
    st.button(label="Start")
    st.write("Do you want to start sentiment analysis?")
    st.button(label="Start")
    st.markdown("#### Supported by")
    cols1,cols2=st.columns(2)
    with cols1:
        st.markdown('##### Douban')
        st.image('./img/douban_logo.png')
    with cols2:
        st.markdown("##### OpenAI")
        st.image("./img/OpenAIlogo.png")































