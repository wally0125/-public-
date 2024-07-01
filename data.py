import pandas as pd
import numpy as np
import sxtwl
from datetime import datetime,timedelta
from sklearn.preprocessing import LabelEncoder
import statistics

'''
讀重量資訊
'''
def read_weight(path):
    #weight = pd.read_excel(path, engine='openpyxl')
    #weight.to_csv("weight.csv", header=True, index=False)
    weight = pd.read_csv(path)
    weight.dropna(axis=1,how='all',inplace=True)  #把沒有值的行刪掉
    return weight

'''
讀餌料資訊
'''
def read_food(path):
    #food = pd.read_excel(path, engine='openpyxl')
    #food.to_csv("food.csv", header=True, index=False)
    food = pd.read_csv(path)
    food['Date']=[datetime.date(datetime.strptime(str(x), "%Y/%m/%d")) for x in list(food['Date'])]  #統一日期格式
    food.dropna(axis=1,how='all',inplace=True)    #把沒有值的行刪掉
    return food

'''
讀水質資訊
'''
def read_water(path):
    #water = pd.read_excel(path, engine='openpyxl')
    #water.to_csv("water.csv", header=True, index=False)
    water = pd.read_csv(path)      
    
    water['Date']=[datetime.date(datetime.strptime(str(x), "%Y%m%d")) for x in list(water['Date'])] #統一日期格式
    water = water.drop(['Marine(CFU/ml)', '螢光菌TCBS (CFU/ml)',  '螢光菌 Marine (CFU/ml)','Note'], axis = 1) #把缺失值多的資料去掉
    water.dropna(axis=1,how='all',inplace=True)   #把沒有值的行刪掉
     
    #處理缺失值
    #water = water.interpolate() #差值法(前一項和下一項的平均)
    water=water.sort_values(by=['Tank ID','Date'])
    water = water.fillna(method = 'ffill' ,axis = 0)#用前一項填NaN值
    water = water.fillna(method = 'backfill' ,axis = 0)#用下一項填NaN值
    water = water.dropna(axis = 'rows')#若還有缺失值就刪掉整列
    
    
    #把有分成早中晚的資料合併成一個(取最大值)
    water['Salinity'] = [max(i, j, k) for i, j, k in zip(water['Salinity_1'].astype(np.float32),  water['Salinity_2'].astype(np.float32),water['Salinity_3'].astype(np.float32))]
    water['pH'] = [max(i, j, k) for i, j, k in zip(water['pH_1'].astype(np.float32), water['pH_2'].astype(np.float32), water['pH_3'].astype(np.float32))]
    water['O2'] = [max(i, j, k) for i, j, k in zip(water['O2 (mg/l)_1'].astype(np.float32), water['O2 (mg/l)_2'].astype(np.float32), water['O2 (mg/l)_3'].astype(np.float32))]
    water['ORP'] = [max(i, j, k) for i, j, k in zip(water['ORP (mV)_1'].astype(np.float32), water['ORP (mV)_2'].astype(np.float32), water['ORP (mV)_3'].astype(np.float32))]
    water['temp'] = [max(i, j, k) for i, j, k in zip(water['W Temp_1'].astype(np.float32), water['W Temp_2'].astype(np.float32),  water['W Temp_3'].astype(np.float32))]
    #刪掉分成早中晚的資料
    water = water.drop(['Salinity_1', 'Salinity_2', 'Salinity_3', 'pH_1', 'pH_2', 'pH_3', 'O2 (mg/l)_1', 'O2 (mg/l)_2',
                        'O2 (mg/l)_3', 'ORP (mV)_1', 'ORP (mV)_2', 'ORP (mV)_3','W Temp_1', 'W Temp_2', 'W Temp_3'], axis = 1)
    

    for i in range(len(water.index)):
        if water['Salinity'][i]>50 or water['Salinity'][i]<20:
            print('Salinity',water.index[i],water['Salinity'][i])
        if water['pH'][i]>10 or water['pH'][i]<6 :
            print('pH',water.index[i],water['pH'][i])
        if water['temp'][i]>35 or water['temp'][i]<20 :
            print('temp',water.index[i],water['temp'][i])
        if water['ORP'][i]>800 or water['ORP'][i]<0 :
            print('ORP',water.index[i],water['ORP'][i])
        if water['O2'][i]>20 or water['O2'][i]<0 :
            print('O2',water.index[i],water['O2'][i])
        if water['Alkalinity (ppm)'][i]>300 or water['Alkalinity (ppm)'][i]<0 :
            print('Alkalinity (ppm)',water.index[i],water['Alkalinity (ppm)'][i])
        if water['NO3 (mg/l)'][i]>1500 or water['NO3 (mg/l)'][i]<0 :
            print('NO3 (mg/l)',water.index[i],water['NO3 (mg/l)'][i])
        if water['NO2 (mg/l)'][i]>100 or water['NO2 (mg/l)'][i]<0 :
            print('NO2 (mg/l)',water.index[i],water['NO2 (mg/l)'][i])
        if water['NH4-N(mg/l)'][i]>10 or water['NH4-N(mg/l)'][i]<0 :
            print('NH4-N(mg/l)',water.index[i],water['NH4-N(mg/l)'][i])
        #if water['TCBS (CFU/ml)'][i]>5000 or water['TCBS (CFU/ml)'][i]<0 :
            #print('TCBS (CFU/ml)',water.index[i],water['TCBS (CFU/ml)'][i])
        #if water['TCBS 綠菌 (CFU/ml)'][i]>5000 or water['TCBS 綠菌 (CFU/ml)'][i]<0 :
            #print('TCBS 綠菌 (CFU/ml)',water.index[i],water['TCBS 綠菌 (CFU/ml)'][i])
        
    return water
'''
讀蝦隻所處的水缸位置

加入剪眼時的年齡(birth_cut):剪眼日-出生日
'''
def read_position(path):
    #tank = pd.read_excel(path, engine='openpyxl')
    #tank.to_csv("tank.csv", header=True, index=False)
    tank = pd.read_csv(path)
    tank.dropna(axis=1,how='all',inplace=True)   #把沒有值的行刪掉

    #統一日期格式
    
    col_list = list(tank.columns)
    col_time = [datetime.date(datetime.strptime(str(x), "%Y/%m/%d")) for x in col_list[4:]]
    col_list[4:] = col_time
    tank.columns = col_list
    
    
    #加入"birth_cut":剪眼時的年齡
    tank['出生日'] = [datetime.date(datetime.strptime(str(x), "%Y/%m/%d")) for x in tank['出生日']]
    #tank['剪眼日'] = [datetime.date(datetime.strptime(str(x), "%Y/%m/%d")) for x in tank['剪眼日']]
    #tank['birth_cut'] = tank['剪眼日'] - tank['出生日']
    #只取birth_cut的數字部分
    #tank['birth_cut'] = (tank['birth_cut'] / np.timedelta64(1, 'D')).astype(int)
    
    return tank

'''
讀蝦隻每天的卵巢成熟階段
'''
def read_grown(path):
    #grown = pd.read_excel(path, engine='openpyxl')
    #grown.to_csv("grown.csv", header=True, index=False)
    grown = pd.read_csv(path)
    grown = grown.drop(['出生日','剪眼日'],axis = 1)
    grown.dropna(axis=1,how='all',inplace=True)    #把沒有值的行刪掉
    
    #統一日期格式
    col_list = list(grown.columns)
    col_time = [datetime.date(datetime.strptime(str(x), "%Y/%m/%d")) for x in col_list[2:]]
    col_list[2:] = col_time
    grown.columns = col_list
    
    #統一資料表示方式
    col = np.where(grown == '剪眼')[0]
    row = np.where(grown == '剪眼')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '剪眼'
        
    col = np.where(grown == '死亡')[0]
    row = np.where(grown == '死亡')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '淘汰'        
        
    col = np.where(grown == 0.0)[0]
    row = np.where(grown == 0.0)[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '0'
    
    col = np.where(grown == '0-Ⅰ')[0]
    row = np.where(grown == '0-Ⅰ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '0-I'
    
    col = np.where(grown == 'Ⅰ-Ⅱ')[0]
    row = np.where(grown == 'Ⅰ-Ⅱ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'I-II'
        
    col = np.where(grown == 'Ⅰ')[0]
    row = np.where(grown == 'Ⅰ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'I'
    
    col = np.where(grown == 'Ⅱ')[0]
    row = np.where(grown == 'Ⅱ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'II'
    
    col = np.where(grown == 'Ⅱ-Ⅲ')[0]
    row = np.where(grown == 'Ⅱ-Ⅲ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'II-III'
    
    col = np.where(grown == 'Ⅲ')[0]
    row = np.where(grown == 'Ⅲ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'III'
    
    col = np.where(grown == 'II - III')[0]
    row = np.where(grown == 'II - III')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'II-III'
    
    col = np.where(grown == '脫殼有受精')[0]
    row = np.where(grown == '脫殼有受精')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '脫殼受精'
    
    col = np.where(grown == '脫0')[0]
    row = np.where(grown == '脫0')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '脫殼'
        
    col = np.where(grown == '脫殼死亡')[0]
    row = np.where(grown == '脫殼死亡')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = '脫殼'
    
    col = np.where(grown == 'I-II ')[0]
    row = np.where(grown == 'I-II ')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'I-II'
    
    col = np.where(grown == 'II - III')[0]
    row = np.where(grown == 'II - III')[1]
    for i, j in zip(col, row):
        grown.iloc[i, j] = 'II-III'
    
    return grown

"""
計算全部脫殼周期平均
脫殼周期計算方法:後一次脫殼時間 - 前一次脫殼時間

all_shelling_daycount_mean:個別蝦隻脫殼周期平均(一隻蝦會有一個值)
all_shelling_after_mean:個別蝦隻脫殼前卵巢成熟階段為0的天數平均(一隻蝦會有一個值)
all_shelling_before_mean:個別蝦隻脫殼後卵巢成熟階段為0的天數平均(一隻蝦會有一個值)

mean_average:全部蝦隻脫殼周期平均
mean_after_average:全部蝦隻脫殼前卵巢成熟階段為0的天數平均
mean_before:全部蝦隻脫殼後卵巢成熟階段為0的天數平均
"""
def calculate_shelling_mean_average(grown): #放入卵巢成熟階段資料
    #print('{:<4}'.format('眼標'),'{:<35}'.format('脫殼週期'),'{:>6}'.format('平均'),'{:<33}'.format('脫殼前'),'{:<33}'.format('脫殼後'))
    all_shelling_daycount_mean=[]
    all_shelling_after_mean=[]
    all_shelling_before_mean=[]
    Eyetags = set(grown['Eyetag'])
    for eyetag in Eyetags:          #一隻一隻蝦計算
        num=np.where(grown.Eyetag == eyetag)[0][0]
        life = grown.iloc[num, :]
        life = life.iloc[2:]
        life = pd.DataFrame(life[life.notnull()]) 

        count_shell=0
        count_after=0
        mean=0
        mean_before=0
        mean_after=0
        first_shelling_key=0     #用來判斷一隻蝦是不是第一次脫殼
        key_after=0              #用來控制脫殼後卵巢成熟階段為0的天數
        all_shelling_daycount=[]  #用來存脫殼周期
        all_shelling_after=[]     #用來存脫殼後卵巢成熟階段為0的天數
        for i in range(len(life)):
            if(life.iat[i,0]=='脫殼')|(life.iat[i,0]=='脫殼受精'):   #判斷是否遇到脫殼
               if(first_shelling_key==0):         #如果是第一次遇到就開始計算天數
                    first_shelling_key=1
               elif(first_shelling_key==1):       #之後再遇到就把計算好的天數存下來，並歸零重新一輪
                    all_shelling_daycount.append(count_shell)
                    all_shelling_after.append(count_after)
                    count_shell=0
                    count_after=0
                    key_after=0
            if(first_shelling_key==1):   #從第一次脫殼後開始計算
               count_shell+=1        #計算距離下一次脫殼所需天數
               if(key_after==0):
                    if(life.iat[i,0]=='脫殼')|(life.iat[i,0]=='脫殼受精'):
                        count_after=0
                    elif(life.iat[i,0]=='0'):         #計算脫殼後卵巢成熟階段為0的天數
                        count_after+=1
                    else:                             #一有中斷就停止紀錄 key_after設為1
                        key_after=1      
        all_shelling_after.append(count_after)

        count_before=0
        first_shelling_key=0
        key_before=0
        all_shelling_before=[]     #用來存脫殼前卵巢成熟階段為0的天數
        for i in reversed(range(len(life))):   #倒過來計算脫殼前卵巢成熟階段為0的天數
            if(life.iat[i,0]=='脫殼')|(life.iat[i,0]=='脫殼受精'):
               if(first_shelling_key==0):
                    first_shelling_key=1
               elif(first_shelling_key==1):
                    all_shelling_before.append(count_before)
                    count_before=0
                    key_before=0
            if(first_shelling_key==1):
               if(key_before==0):
                    if(life.iat[i,0]=='脫殼')|(life.iat[i,0]=='脫殼受精'):
                        count_before=0
                    elif(life.iat[i,0]=='0'):
                        count_before+=1
                    else:
                        key_before=1                          
        all_shelling_before.append(count_before)
        
    
        if(all_shelling_daycount!=[]):
            mean = statistics.mean(all_shelling_daycount)
        if(mean!=0):
            all_shelling_daycount_mean.append(mean)
        if(all_shelling_after!=[]):
            mean_after = statistics.mean(all_shelling_after)
        if(mean_after!=0):
            all_shelling_after_mean.append(mean_after)
        if(all_shelling_before!=[]):
            mean_before = statistics.mean(all_shelling_before)
        if(mean_before!=0):
            all_shelling_before_mean.append(mean_before)
        all_shelling_before=all_shelling_before[::-1]
        
        #print('{:<5}'.format(eyetag),'{:<40}'.format(str(all_shelling_daycount)),'[{:>5.2f}]'.format(mean),'{:<35}'.format(str(all_shelling_before)),'{:<35}'.format(str(all_shelling_after)))
    mean_average=statistics.mean(all_shelling_daycount_mean)
    mean_after_average=statistics.mean(all_shelling_after_mean)
    mean_before_average=statistics.mean(all_shelling_before_mean)
    #print('脫殼周期平均: ',mean_average) 
    #print('脫殼前天數平均: ',mean_before_average)  
    #print('脫殼後天數平均: ',mean_after_average)
    
    return mean_average,mean_before_average,mean_after_average


'''
計算一隻蝦脫殼周期
脫殼周期計算方法:後一次脫殼時間 - 前一次脫殼時間
'''
def calculate_shelling_mean(grown):      
    life = grown
    life = life.iloc[2:]
    life = pd.DataFrame(life[life.notnull()]) 
    count_s=0
    mean=0
    first_shelling_key=0
    all_shelling_daycount=[]
    for i in range(len(life)):
        if(life.iat[i,0]=='脫殼')|(life.iat[i,0]=='脫殼受精'):
           if(first_shelling_key==0):
               first_shelling_key=1
           elif(first_shelling_key==1):
               all_shelling_daycount.append(count_s)
               count_s=0
        if(first_shelling_key==1):
            count_s+=1
    if(all_shelling_daycount!=[]):
        mean = statistics.mean(all_shelling_daycount)

    return mean    

'''
依據眼標結合資料
加入'days'特徵，表示一隻蝦當天的記錄天數
加入'Shelling_date'特徵，表示前一次脫殼日期
'''
def merge_info(eyetag,grown,tank,weight,water,mean_average):
    #依據眼標找資料位置
    g = np.where(grown.Eyetag == eyetag)[0][0]   
    t = np.where(tank.Eyetag == eyetag)[0][0]
    w = np.where(weight.Eyetag == eyetag)[0][0]
    
    grown_s = grown.iloc[g, :]
    tank_s = tank.iloc[t, :]
    grown_s = grown_s.iloc[2:]
    tank_s = tank_s.iloc[4:]
    
    #結合卵巢成熟階段和水缸位置資料
    grown_s = pd.DataFrame(grown_s[grown_s.notnull()])
    tank_s = pd.DataFrame(tank_s[tank_s.notnull()])
    S = pd.merge(grown_s, tank_s, how='inner', left_index=True, right_index=True)
    S.columns = ['life', 'Tank ID']
    
    if len(grown_s)!=len(S):
        print(eyetag,':',len(grown_s),': ',len(S))
    #取出出生日，重量資訊
    #birthcut_s=tank.iloc[t, -1]
    birthday_s=tank.iloc[t, 2]
    family_s=tank.iloc[t, 1]
    weight_s=weight.iloc[w,1]
    
    #脫殼週期
    mean = calculate_shelling_mean(grown_s)
    
    S['Eyetag']=eyetag
    S['Family']=family_s
    S['Weight']=weight_s
    #S['Birth_cut']=birthcut_s
    S['Birthday']=birthday_s
    #S['Shellng_daycount']=len(S)
    if(mean!=0):
        S['Shelling_mean']=mean
    else:
        S['Shelling_mean']=mean_average
        
    S = S.reset_index()
    S=S.rename(columns={'index':'Date'})
    
    #依據水缸位置資訊和水質資料結合
    S_water = pd.merge(S, water, how='left', on = ['Date', 'Tank ID'])
    #S_water = pd.merge(S_water, food, how='left', on = ['Date', 'Tank ID'])
    S_water = S_water.fillna(0)
    
    if (S_water.empty):        # 判斷是否為空
        #days =timedelta(days = 0)
        print('Empty:',eyetag)
    else:
        #加入'days'特徵，表示一隻蝦當天的記錄天數
        delta = []
        first_day = S_water['Date'].iloc[0]
        for idx in S_water['Date']:
            delta.append(idx - first_day)
    
        delta = pd.DataFrame(delta)
        delta = (delta[0] / np.timedelta64(1, 'D')).astype(int)
        S_water['days'] = delta.values
    
        #加入'last_shelling_date'特徵，表示前一次脫殼日期
        Shell_date=[]
        Shell_temp=S_water['Date'].iloc[0]
        count_eyetag=0
        for idx in S_water['Date']:
            Shell_date.append(Shell_temp)
            if(S_water['life'].iloc[count_eyetag]=='脫殼')|(S_water['life'].iloc[count_eyetag]=='脫殼受精'):
                Shell_temp=idx
            #Shell_date.append(Shell_temp)
            count_eyetag+=1
        Shell_date = pd.DataFrame(Shell_date)
        S_water['last_shelling_date'] = Shell_date
        
        #加入'last_shelling_date'特徵，表示下一次脫殼日期
        #要預測脫殼時間時用
        Shell_date=[]
        Shell_temp=S_water['Date'].iloc[len(S_water)-1]
        count_eyetag=len(S_water)-1
        for idx in reversed(S_water['Date']):
            if(S_water['life'].iloc[count_eyetag]=='脫殼')|(S_water['life'].iloc[count_eyetag]=='脫殼受精'):
                Shell_temp=idx
            #Shell_date.append(Shell_temp)
            Shell_date.append(Shell_temp)
            count_eyetag-=1
        Shell_date = pd.DataFrame(reversed(Shell_date))
        S_water['next_shelling_date'] = Shell_date
        
        #加入'eyecut_or_not'特徵，表示有無經過檢眼
        eye_cut=[]
        eyecut_temp=0
        #這些蝦沒有剪眼的日期的紀錄，但都是剪過眼的，所以'Eyecut_or_not'都設為1
        name=["W999","W928","B638","W4","W5","W8","W9","W11","W12","W16","W21","W21","W24","W25","W26","W27","W28","W31","W32","W34","W35","P60","W7","W17","W22"]
        count_eyetag=0
        for idx in S_water['Date']:
            if(S_water['Eyetag'].iloc[count_eyetag] in name):
                eyecut_temp=1
            if(S_water['life'].iloc[count_eyetag]=='剪眼'):
                eyecut_temp=1
            eye_cut.append(eyecut_temp)
            count_eyetag+=1
        eye_cut = pd.DataFrame(eye_cut)
        S_water['Eyecut_or_not'] = eye_cut
            
    return S_water
    
"""
加入農曆月份、月相、季節
月份
1: 'Jan',2: 'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'

季節
1,2,3:'Spring' ;  4,5,6:'Summer' ;  7,8,9:'Fall' ; 10,11,12:'Winter'

月相
29~3:moon_1 ; 4~8:moon_2 ; 9~13:moon_3 ; 14~18:moon_4 ; 19~23:moon_5 ; 24~28:moon_6
"""
def add_lunar(data):
    Lunar_month = []
    moon=[]
    season=[]
    for x in data['Date']:
        day = sxtwl.fromSolar(x.year, x.month, x.day) 
        M = day.getLunarMonth()
        D = day.getLunarDay()
       
        #農曆月份判斷
        if M==1:
            Lunar_month.append('Jan')
        elif M==2:
            Lunar_month.append('Feb')
        elif M==3:
            Lunar_month.append('Mar')
        elif M==4:
            Lunar_month.append('Apr')
        elif M==5:
            Lunar_month.append('May')
        elif M==6:
            Lunar_month.append('Jun')
        elif M==7:
            Lunar_month.append('Jul')
        elif M==8:
            Lunar_month.append('Aug')
        elif M==9:
            Lunar_month.append('Sep')
        elif M==10:
            Lunar_month.append('Oct')
        elif M==11:
            Lunar_month.append('Nov')
        elif M==12:
            Lunar_month.append('Dec')
        
        
        #月相判斷
        if(D>=29)or(D<=3):
            moon.append('moon_1')
        elif(4<=D<=8):
            moon.append('moon_2')
        elif(9<=D<=13):
            moon.append('moon_3')
        elif(14<=D<=18):
            moon.append('moon_4')
        elif(19<=D<=23):
            moon.append('moon_5')
        elif(24<=D<=28):
            moon.append('moon_6')
        
        #季節判斷
        if M in [1,2,3]:
            season.append('spring')
        elif M in [4,5,6]:
            season.append('summer') 
        elif M in [7,8,9]:
            season.append('fall') 
        elif M in [10,11,12]:
            season.append('winter') 

    data['Lunar'] = Lunar_month
    data['Moon'] = moon
    data['Season'] = season    
    
    data = pd.concat([data,pd.get_dummies(data['Lunar'])], axis = 1) # Lumar one hot encode
    data = pd.concat([data,pd.get_dummies(data['Moon'])], axis = 1)  # Moon one hot encode
    data = pd.concat([data,pd.get_dummies(data['Season'])], axis = 1)  # Season one hot encode

    return data

'''
統一日期格式
'''
def uniform_date_format(data):
    data['Date'] = [datetime.strptime(str(x), "%Y-%m-%d") for x in data['Date']]
    data['last_shelling_date'] = [datetime.strptime(str(x), "%Y-%m-%d") for x in data['last_shelling_date']]
    data['next_shelling_date'] = [datetime.strptime(str(x), "%Y-%m-%d") for x in data['next_shelling_date']]
    data['Birthday'] = [datetime.strptime(str(x), "%Y-%m-%d") for x in data['Birthday']]   
    
'''
加入年齡(天)
計算方式:當天日期(date)-出生日期(Birthday)
'''
def add_age(data):
    data['age'] = data['Date'] - data['Birthday']
    data['age'] = (data['age']  / np.timedelta64(1, 'D')).astype(int)#只取數字部分
    
'''
加入一個特徵紀錄距離上次脫殼過了多久(天)
'''
def add_last_shelling(data):
    data['last_molting']=data['Date']-data['last_shelling_date']
    data['last_molting'] = (data['last_molting'] / np.timedelta64(1, 'D')).astype(int)
'''
加入一個特徵紀錄距離下次脫殼還有多久(天)
'''
def add_next_shelling(data):
    data['next_molting']=data['next_shelling_date']-data['Date']
    data['next_molting'] = (data['next_molting'] / np.timedelta64(1, 'D')).astype(int)

'''
卵巢成熟階段one hot encoding
'''
def life_one_hot(data):
    Label = pd.get_dummies(data['life'])
    data = pd.concat([data, Label], axis = 1)
    return data

'''
對life用labelEncoder
0:0 ,1:0-I ,2:I ,3:I-II ,4:II ,5:II-III ,6:III ,7:人工授精 ,8:死亡 ,9:生產 ,10:脫殼 ,11:脫殼受精   (沒有剪眼時)
0:0 ,1:0-I ,2:I ,3:I-II ,4:II ,5:II-III ,6:III ,7:人工授精 ,8:剪眼 ,9:死亡 ,10:生產 ,11:脫殼 ,12:脫殼受精 
'''
def life_label_encode(data):
    le = LabelEncoder()
    data['life_encode'] = le.fit_transform(data['life'])

def family_label_encode(data):
    le = LabelEncoder()
    data['Family_encode'] = le.fit_transform(data['Family'])
    
def family_one_hot(data):
    Label = pd.get_dummies(data['Family'])
    data = pd.concat([data, Label], axis = 1)
    return data




