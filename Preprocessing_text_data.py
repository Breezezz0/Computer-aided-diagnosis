import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import re
import json
def preprocess_data_and_get_index (df_data_path, df_abbr_path) :
    df_data = pd.read_csv(df_data_path, sep='\t')
    df_abbr = pd.read_excel(df_abbr_path)
    df_data = df_data.fillna(0)
    index_chinese = []
    for index,emr in enumerate(df_data.iloc[:,8]) :
        emr = str(emr)
        for word in emr :
            if  (u'\u4e00' <= word <= u'\u9fff') : #找尋是否含有中文字
                index_chinese.append(df_data.at[index,'病歷號'])
                break
    index_chinese = list(OrderedDict.fromkeys(index_chinese))
    print(len(index_chinese))
    
    index_eng = []
    for index, patient in enumerate(df_data.iloc[:,1]) :
        date = df_data.at[index,'就診日']
        if patient not in index_chinese :
            index_eng.append([date,patient])
    index_eng_sorted = []
    for i in index_eng :
        if i not in index_eng_sorted :
            index_eng_sorted.append(i)

    # load abbr information
    abbr_dict = {}
    for index , abbr in enumerate(df_abbr.iloc[:,0]) :
        abbr_dict[abbr] = df_abbr.at[index,'Word']
    print(abbr_dict)

    #replace abbreviation
    df_emr = df_data['病歷紀錄.1'].copy()
    print(df_emr.head())
    for index, text in enumerate(df_emr) :
        if type(text) != str :
            df_emr[index] = str(text)
        for abbr in abbr_dict.keys() :
            text = re.sub(abbr,abbr_dict[abbr],str(text))
        text = text.replace('_x000D_', '\r')
        df_emr[index] = str(text)

    df_inf_copy = df_data.copy()
    df_inf_copy['病歷紀錄.1'] = df_emr
    return df_inf_copy, index_eng_sorted 
    

#抓取病人資料並可以透過函式返回dict格式的資料
class patient_inf ():
    def __init__(self , df , med_rec_no) :
        self.df = df 
        self.num = med_rec_no
    def get_inf (self) :
        inf = {}
        df_patient = self.df[self.df['病歷號']==self.num]
        inf['性別'] = df_patient.at[df_patient.index[0],'性別']
        inf['診斷碼'] = []
        inf['診斷碼'].append(df_patient.at[df_patient.index[0],'診斷碼'])
        if type(df_patient.at[df_patient.index[0],'診斷碼.1']) != str :
            inf['診斷碼'].append(df_patient.at[df_patient.index[0],'診斷碼.1'])
        if type(df_patient.at[df_patient.index[0],'診斷碼.2']) != str :
            inf['診斷碼'].append(df_patient.at[df_patient.index[0],'診斷碼.2'])
        for i in range(len(df_patient.index)) :
            if df_patient.iloc[i,9] == '徵侯' :
                inf['徵侯'] = df_patient.iloc[i,10]
            elif df_patient.iloc[i,9] == '病史' :
                inf['病史'] = df_patient.iloc[i,10]
            else :
                inf['處置'] = df_patient.iloc[i,10]
        return inf

class patient_inf_v2 ():
    def __init__(self , df , med_rec_no, date) :
        self.df = df 
        self.num = med_rec_no
        self.date = date
    def get_inf (self) :
        inf = {}
        df_patient = self.df[(self.df['病歷號']==self.num) & (self.df['就診日']==self.date)]
        inf['性別'] = '男' if df_patient.at[df_patient.index[0],'性別']== 1 else '女'
        diagnosis = df_patient.at[df_patient.index[0],'診斷碼'].split(' ')
        while (diagnosis.count('')) :
            diagnosis.remove('')
        # for i, value in enumerate(diagnosis) :
        #     diagnosis[i] = float(value)
        inf['診斷碼'] = diagnosis
        for i in range(len(df_patient.index)) :
            if df_patient.iloc[i,7] == '徵侯' :
                inf['徵侯'] = df_patient.iloc[i,8]
            elif df_patient.iloc[i,7] == '病史' :
                inf['病史'] = df_patient.iloc[i,8]
            else :
                inf['處置'] = df_patient.iloc[i,8]
        return inf


if __name__ == '__main__':
    df_path = '111 門診'
    df_abbr_path = 'abbreviation.xlsx'
    df_preprocessed , index_eng = preprocess_data_and_get_index(df_path, df_abbr_path)
    dataset_inf = {}
    for date, patient in index_eng :
        key = str(date)+str(patient)
        dataset_inf[key] = patient_inf_v2(df_preprocessed, patient, date).get_inf()
    print(dataset_inf)