import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime
#from transformers import pipeline
import plotly.express as px



@st.cache_data
def load_data(path):
    df=pd.read_excel(path)
    return df

@st.cache_data
def sentimentprocess(sentiment_timesdf):
    sentiment_timesdf['sum']=sentiment_timesdf[['Neutral','Negative','Positive']].sum(axis=1)
    sentiment_timesdf['Neutral_per']=sentiment_timesdf['Neutral']/sentiment_timesdf['sum']
    sentiment_timesdf['Negative_per']=sentiment_timesdf['Negative']/sentiment_timesdf['sum']
    sentiment_timesdf['Positive_per']=sentiment_timesdf['Positive']/sentiment_timesdf['sum']
    return sentiment_timesdf

@st.cache_data
def keywordsrawprocessing(keyworddf):
    keyworddf=keyworddf.dropna()
    keywordslist=[]
    for i in range(0,len(keyworddf)):
        k=keyworddf.loc[i,'keywords']
        words=re.split(',',k)
        keywordslist.append(words)   
    keywordslistdf=pd.DataFrame(keywordslist)
    keywordslistdf['date']=keyworddf['date']
    return keywordslistdf

@st.cache_data
def wordsprocessing(keywordslistdf):
    for i in range(0,keywordslistdf.shape[0]):
        for j in range(0,keywordslistdf.shape[1]):
            text=keywordslistdf.iloc[i,j]
            if text!=None:
                text=text.replace("'",'')
                text=text.replace(" ",'')
                text=text.replace("《",'')
                text=text.replace("》",'')
                keywordslistdf.iloc[i,j]=text.strip()
            else:
                keywordslistdf.iloc[i,j]=''
    keywordsdict=keywordslistdf.melt().value.unique()
    return keywordslistdf,keywordsdict

@st.cache_data
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

@st.cache_data
def contribution(list,keywords_timesdf_final):
    df=pd.DataFrame([])
    for i in list:
        tempdf=keywords_timesdf_final[keywords_timesdf_final.keywords.str.contains(i)]
        df=pd.concat([df,tempdf],axis=0)
    df_clean=df.drop_duplicates()
    contri=df_clean.times.sum()
    return contri


def contribu_bar(keywords_timesdf_final):
        directorlist=['郭','敬明','导','gjm','四','GJM','顾晓声','GXS','顾小声']
        actresslist=['虞','书','欣','女主','兰花','苍','ysx','🐟']
        actorlist=['男主','张凌赫','公子羽','0d','张']
        Supporting=['上官','丞','浅','女二','宫尚','角','尚','男三','男三','三','二','宫二','宫三','浅之角','夜色','金靖']
        director_contribu=contribution(directorlist,keywords_timesdf_final)
        actress_contribu=contribution(actresslist,keywords_timesdf_final)
        actor_contribu=contribution(actorlist,keywords_timesdf_final)
        Supporting_contribu=contribution(Supporting,keywords_timesdf_final)
        labels=['director','actor','actress','supporting']
        contribu=[director_contribu,actor_contribu,actress_contribu,Supporting_contribu]
        contribu_bardf=pd.DataFrame(np.array([labels,contribu])).T
        contribu_bardf.columns=['label','contribu']
        return contribu_bardf


## sentiment_classifier = pipeline('zero-shot-classification',model="facebook/bart-large-mnli")

st.set_page_config(layout='wide', page_title="云之羽 SMA")

sentiment_timesdf=load_data('./data/sentiment_analysis.xlsx')
sentiment_timesdf=sentimentprocess(sentiment_timesdf)
keyworddf=load_data('./data/keywords_text-davinci-003.xlsx')
keywordslistdf=keywordsrawprocessing(keyworddf)
keywordslistdf_final,keywordsdict_final=wordsprocessing(keywordslistdf.iloc[:,:-1])
keywords_timesdf_final=wordstimes(keywordslistdf_final,keywordsdict_final)

with st.sidebar:
    st.image("./img/logo.jpg",use_column_width=True)
    st.header("Review analysis",divider='rainbow')
    gpt_api_key=st.text_input(label="Please input your GPT API KEY:",type='password')
    st.write("Do you want to start analysis?")
    start=st.button(label="Start",key='analysisstart',use_container_width=True)

    st.markdown("#### Supported by")
    cols1,cols2=st.columns(2)
    with cols1:
        st.markdown('##### Douban')
        st.image('./img/douban_logo.png',use_column_width=True)
    with cols2:
        st.markdown("##### OpenAI")
        st.image("./img/OpenAIlogo.jpg",use_column_width=True)
    st.text('Data updated at 22/09/2023')


col1,col2=st.columns(2)
with col1:
    st.subheader("Hot value analysis")
    placeholderforhot=st.empty()
    placeholderforhot.image('./img/empty.png',use_column_width=True)

    st.subheader("Sentiment analysis")
    placeholderforsentiment=st.empty()
    placeholderforsentiment.image('./img/empty.png',use_column_width=True)
    if start:
        fig_sentiment=px.bar(sentiment_timesdf,x='date',y=['Negative_per','Positive_per','Neutral_per'],title='Sentiment VS Date')
        fig_sentiment.update_layout(height=300)
        placeholderforsentiment.plotly_chart(fig_sentiment,theme="streamlit",use_container_width=True,height=300)
with col2:
    st.subheader("Keywords analysis")
    placeholderforkeywords=st.empty()
    placeholderforkeywords.image('./img/empty.png',use_column_width=True)
    if start:
        keywords_top10_final=keywords_timesdf_final.sort_values(by='times',ascending=False).head(10)
        fig_keywords_10_final=px.bar(keywords_top10_final,x='keywords',y='times',title="Keywords VS Date")
        fig_keywords_10_final.update_layout(height=300)
        placeholderforkeywords.plotly_chart(fig_keywords_10_final,theme="streamlit",use_container_width=True,height=300)

    st.subheader("Contribution analysis")
    placeholderforcontribu=st.empty()
    placeholderforcontribu.image('./img/empty.png',use_column_width=True)
    if start:
        contribu_bardf=contribu_bar(keywords_timesdf_final)
        fig_contribu=px.bar(contribu_bardf,x='label',y='contribu',title='Contribution')
        fig_contribu.update_layout(height=300)
        placeholderforcontribu.plotly_chart(fig_contribu,theme="streamlit",use_container_width=True,height=300)


























