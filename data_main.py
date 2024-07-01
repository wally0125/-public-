import data as dt
import pandas as pd
import numpy as np

#讀資料
weight=dt.read_weight('dataset/weight.csv')
water=dt.read_water('dataset/water.csv')
water.to_csv('dataset/water_clean.csv', index=False) 
tank=dt.read_position('dataset/tank.csv')
grown=dt.read_grown('dataset/grown.csv')
#%%
#計算全部資料脫殼周期平均
mean_average,mean_before_average,mean_after_average=dt.calculate_shelling_mean_average(grown)

#t = np.where(tank.Eyetag =='W999')[0][0]

data_set = pd.DataFrame()
Eyetags = set(grown['Eyetag'])

#依據每隻蝦的眼標，把每隻蝦每天的資料合併
for eyetag in Eyetags:
    S_water= dt.merge_info(eyetag,grown,tank,weight,water,mean_average)
    data_set = pd.concat([data_set, S_water], axis = 0)
    
    #檢查是某有0的紀錄
    for i in range(len(S_water.index)):
        if S_water['pH'][i]==0:
            print(S_water.Eyetag[i],S_water.index[i])
#%%

#dt.uniform_date_format(data_set)       #統一日期格式

data_set=dt.add_lunar(data_set)        #加入農曆時間資訊
dt.add_age(data_set)                    #加入年齡
dt.add_last_shelling(data_set)          #加入一個特徵紀錄距離上次脫殼過了多久(天)
dt.add_next_shelling(data_set)         #加入一個特徵紀錄距離下次脫殼還有多久(天)
data_set=dt.life_one_hot(data_set)      #Life one hot encoding
dt.life_label_encode(data_set)          #Life label encoding
dt.family_label_encode(data_set)          #family label encoding
data_set=dt.family_one_hot(data_set) 

#將有中文名稱的特徵改成英文
data_set=data_set.rename(columns={'人工授精':'artificial insemination',
                                  '淘汰':'died',
                                  '生產':'yielding',
                                  '脫殼':'shelling',
                                  '脫殼受精':'molting',
                                  'TCBS 綠菌 (CFU/ml)':'TCBS_green (CFU/ml)',
                                  '剪眼':'eye_cut'
                                  })
'''
#更改特徵順序,並選取需要特徵
data_set = data_set.reindex(columns=['Eyetag','Date','Family','Family_encode','Weight','age','days','Tank ID','NH4-N(mg/l)', 'NO2 (mg/l)', 'NO3 (mg/l)', 'Alkalinity (ppm)','TCBS (CFU/ml)', 'TCBS_green (CFU/ml)',
                                     'Salinity', 'pH', 'O2', 'ORP','temp','Birth_cut', 'Shelling_mean','spring','summer','fall','winter',
                                     'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','moon_1','moon_2','moon_3','moon_4','moon_5','moon_6',
                                     'last_shelling','life', '0', '0-I', 'I','I-II', 'II', 'II-III','III', 'died','yielding', 'shelling', 'moulting','G52','G992','W1213CF7f3'
                                     ,'W910','W914','B636','B88','F0210','F0310','F0710','F0810','F0910','F1210','F1310'])
'''
'''
data_set = data_set.reindex(columns=['Eyetag','Date','Family','Family_encode','Weight','age','life','life_encode','days','Tank ID','NH4-N(mg/l)', 'NO2 (mg/l)', 'NO3 (mg/l)', 'Alkalinity (ppm)','TCBS (CFU/ml)', 'TCBS_green (CFU/ml)',
                                     'Salinity', 'pH', 'O2', 'ORP','temp', 'Shelling_mean','spring','summer','fall','winter',
                                     'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','moon_1','moon_2','moon_3','moon_4','moon_5','moon_6',
                                     'last_shelling', '0', '0-I', 'I','I-II', 'II', 'II-III','III', 'eye_cut','died','yielding', 'shelling', 'moulting','artificial insemination'
                                     ])
'''

data_set=data_set.drop(["life","last_shelling_date","Tank ID","Birthday","Lunar","Moon","Season"], axis=1)
data_set.to_csv("dataset_1.csv", header=True, index=False)

