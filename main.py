import method 
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression,RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import random
from collections import Counter
import time
#%%
data_set = pd.read_csv('dataset.csv')

#建立一個用來預測的label
data_set=method.label_3(data_set)

#將"脫殼""脫殼受精"的卵巢成熟階段設為0
data_set.loc[data_set.life_encode==10,'0']=0   #生產
data_set.loc[data_set.life_encode==11,'0']=1  #脫殼
data_set.loc[data_set.life_encode==12,'0']=1  #脫殼受精
data_set.loc[data_set.life_encode==12,'shelling']=1  #脫殼受精
#將"淘汰"刪掉
data_set=data_set.drop(data_set[data_set.life_encode==9].index)

#篩選要使用的資料
#全部
df=data_set.drop(['Family_encode','Shelling_mean'], axis=1)

#%%
#將資料清理成只有第一次脫殼後的
Eyetag=set(df['Eyetag'])
df_after_shelling=pd.DataFrame()
for eyetag in Eyetag:
    temp=df[df['Eyetag']==eyetag]
    count=0
    for i in range(len(temp.index)):
        if temp['shelling'].iloc[i]==1 or temp['moulting'].iloc[i]==1:
            count=i
            break
    df_after_shelling=df_after_shelling.append(temp[count+1:])
    
#%%
all_shelling_daycount,eyetag_drop=method.caculate_shelling_days(df_after_shelling,Eyetag)   

#刪掉脫殼時間一 常的蝦隻
for eyetag in eyetag_drop:
    df_after_shelling = df_after_shelling.drop(df_after_shelling[df_after_shelling['Eyetag'] == eyetag].index)  

#%%
pastDay_list=[7]
futureDay=7
Max_pastDay=np.max(pastDay_list)
'''
預測label數量統計
'''
print('Original shrimp counts:',len(set(df['Eyetag'])))
print('shrimp counts:',len(set(df_after_shelling['Eyetag'])))
#將記錄天數不夠用來預測的蝦隻資料刪掉，只取紀錄天數大於 pastday+futureDay 的
df_final=method.drop_not_enough_data(df_after_shelling,Max_pastDay,futureDay)
print('shrimp counts:',len(set(df_final['Eyetag'])))
print('label count:',len(df_final[df_final['days']>=Max_pastDay]))
print('label count:',len(df_final))
print('Label 0 count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['0']==1]))
print('Label 0-I count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['0-I']==1]))
print('Label I count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['I']==1]))
print('Label I-II count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['I-II']==1]))
print('Label II count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['II']==1]))
print('Label II-III count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['II-III']==1]))
print('Label III count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['III']==1]))
print('yielding count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['yielding']==1]))
print('shelling count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['shelling']==1]))
print('molting count:',len(df_final[df_final['days']>=Max_pastDay][df_final[df_final['days']>=Max_pastDay]['moulting']==1]))

#%%
# 獲取的家族列表
family_list = df['Family'].unique()
family_df = pd.DataFrame(columns=['Family', 'shrimp_count'])

# 輸出每個家族的成員數量
for family in family_list:
    number_count = df[df['Family'] == family]['Eyetag'].nunique()
    family_df = family_df.append({'Family': family, 'shrimp_count': number_count}, ignore_index=True)
#family_df.to_excel('family.xlsx', index=False)

#%%
'''
建立訓練資料
'''
def buildTrain(df_final, pastDay, futureDay,Max_pastDay):
   X_train=pd.DataFrame()
   Eyetags = set(df_final['Eyetag'])
   #53個
   family_list = set(df_final['Family'])
   for eyetag in Eyetags:
       train_temp=df_final[df_final['Eyetag']==eyetag]
       #train_temp = train_temp.drop(["Date","Eyetag"], axis=1)
     
       if train_temp.shape[0]>=pastDay+futureDay:
           for i in range(train_temp.shape[0]-futureDay-pastDay):
             if i>=Max_pastDay-pastDay:  #為了讓使用不同天數做預測時的資料數量一樣，以最大的pastDay作為基準產生資料
                 #if(np.array(train_temp.iloc[i+pastDay:i+pastDay+futureDay]["eye_cut"])==0).all() and (np.array(train_temp.iloc[i+pastDay:i+pastDay+futureDay]["artificial insemination"])==0).all()and (np.array(train_temp.iloc[i+pastDay:i+pastDay+futureDay]["yielding"])==0).all()and (np.array(train_temp.iloc[i+pastDay:i+pastDay+futureDay]["shelling"])==0).all()and (np.array(train_temp.iloc[i+pastDay:i+pastDay+futureDay]["moulting"])==0).all():
                X_train_temp=pd.DataFrame()
                X_train_temp['Date']=pd.Series([train_temp["Date"].iloc[i+pastDay-1]])
                X_train_temp['Eyetag']=pd.Series([train_temp["Eyetag"].iloc[i+pastDay-1]])
                X_train_temp['Weight']=pd.Series([train_temp["Weight"].iloc[i+pastDay-1]])
                X_train_temp['age']=pd.Series([train_temp["age"].iloc[i+pastDay-1]])
                X_train_temp['days']=pd.Series([train_temp["days"].iloc[i+pastDay-1]])
                X_train_temp['last_shelling']=pd.Series([train_temp["last_shelling"].iloc[i+pastDay-1]])
                X_train_temp['next_shelling']=pd.Series([train_temp["next_shelling"].iloc[i+pastDay-1]])
                X_train_temp['Eyecut_or_not']=pd.Series([train_temp["Eyecut_or_not"].iloc[i+pastDay-1]])
                water_list=['NH4-N(mg/l)', 'NO2 (mg/l)', 'NO3 (mg/l)', 'Alkalinity (ppm)','TCBS (CFU/ml)', 'TCBS_green (CFU/ml)','Salinity', 'pH', 'O2', 'ORP','temp']
                for w in water_list:
                    temp=df[w].iloc[i:i+pastDay].mean()
                    X_train_temp[w]=pd.Series([temp])
                for family in family_list:
                    X_train_temp[family]=pd.Series([train_temp[family].iloc[i]])
                for p in range(pastDay):
                    X_train_temp = pd.concat([X_train_temp, pd.Series(train_temp["life_label"].iloc[i+p]).rename("day_"+str(p+1)+"_life_label")], axis=1)
                    X_train_temp = pd.concat([X_train_temp, pd.Series(train_temp["shelling"].iloc[i+p]).rename("day_"+str(p+1)+"_shelling")], axis=1)
                    #X_train_temp = pd.concat([X_train_temp, pd.Series(train_temp["life_encode"].iloc[i+p]).rename("day_"+str(p+1)+"_life_encode")], axis=1)
                for f in range(futureDay):
                    X_train_temp = pd.concat([X_train_temp, pd.Series(train_temp["life_label"].iloc[i+pastDay+f]).rename("predict_"+str(f+1))], axis=1)

                X_train = pd.concat([X_train, X_train_temp], ignore_index=True)
                 
   return X_train
#data_predict_mix= buildTrain(df_final,pastDay=7, futureDay=7,Max_pastDay=7)
#data_predict_mix.to_csv('data_predict_mix_7_7.csv', index=False) 

#%%
'''
rfe_feature_selection
'''
data_predict_mix = pd.read_csv('data_predict_mix_7_7.csv')
X=data_predict_mix.drop(["Date","Eyetag","next_shelling"], axis=1)
Y=pd.DataFrame()
k_list=[20,30,40,50,60,X.shape[1]]
k_list=[X.shape[1]]
for f in range(futureDay):
    X=X.drop(["predict_"+str(f+1)], axis=1)
    Y = pd.concat([Y, data_predict_mix["predict_"+str(f+1)]], axis=1)
selected_feature_future=method.rfe_feature_selection_future(X, Y,Max_pastDay,futureDay,Max_pastDay,k_list)

X=data_predict_mix.drop(["Date","Eyetag","next_shelling"], axis=1)
for f in range(futureDay):
    X=X.drop(["predict_"+str(f+1)], axis=1)
Y=data_predict_mix.loc[:,["next_shelling"]]
selected_feature_shelling=method.rfe_feature_selection_shelling(X, Y,Max_pastDay,1,Max_pastDay,k_list)

#%%
execution_times=1#模型總共要跑幾次
metric_list=['accuracy','precision','sensitivity','specificity','f1_score','fpr','tpr','roc_auc','time']
#選擇要跑的模型
model_list=['LSTM','CNN','RF','DT','KNN','SVM','ADA','XGB']
model_list=['RF']
for metric in metric_list: 
    for model in model_list:
        for k in k_list:           
            globals()[model+'_'+metric+'_list_k_'+str(k)]=[[0]*futureDay for i in range(execution_times)]
            
for model in model_list:
    for k in k_list:
        globals()[model+'_mse_list_k_'+str(k)]=[[0]*futureDay for i in range(execution_times)]
        globals()[model+'_mae_list_k_'+str(k)]=[[0]*futureDay for i in range(execution_times)]            


for times in range(execution_times):
    #依據眼標將資料分成train_data,val_data,test_data，確保同一隻蝦的資料不會出現在不同資料集中
    train_data,test_data=method.split_by_eyetags_train_test(data_predict_mix,0.2)  
    for pastDay in pastDay_list:      
        print('----------- pastDay=%d  futureDay=%d -----------'%(pastDay,futureDay))      
        count=0
        for k in k_list:
            start = time.process_time()
            '''
            預測脫殼時間
            '''
            print('---------------------- k=%d --------------------'%(k))
            print('---------------- RF predict_shelling(training) -----------------')
            X_train=train_data.drop(["Date","Eyetag","next_shelling"], axis=1)
            Y_train=train_data.loc[:,["next_shelling"]]   
            for f in range(futureDay):
                X_train=X_train.drop(["predict_"+str(f+1)], axis=1)
                
            X_train_select=X_train.loc[:,selected_feature_shelling[count]]
            Y_train_select=Y_train 
            X_train_select=np.array(X_train_select)
            Y_train_select=np.array(Y_train_select)

            # 將資料進行正規化
            X_train_select,X_train_scaler=method.normalize(X_train_select) 
            
            #將資料打散
            X_train_select, Y_train_select = method.shuffle(X_train_select, Y_train_select)

            #建立model
            Y_train_select = Y_train_select.ravel()
            RF_model_predict_shelling = RandomForestRegressor(max_depth=None, min_samples_leaf=1, n_estimators=200)
            RF_model_predict_shelling.fit(X_train_select, Y_train_select)
            
            print('---------------- RF predict_shelling(testing) -------------')
            X_test=test_data.drop(["Date","Eyetag","next_shelling"], axis=1)
            for f in range(futureDay):
                X_test=X_test.drop(["predict_"+str(f+1)], axis=1)
            Y_test_shelling=test_data.loc[:,["next_shelling"]]
            
            X_test_select=X_test.loc[:,selected_feature_shelling[count]]
            X_test_select=np.array(X_test_select)
            X_test_select,X_test_scaler=method.normalize(X_test_select)
            
            Y_test_shelling=np.array(Y_test_shelling)
            Y_test_shelling = Y_test_shelling.ravel()
            
            Y_pred_shelling = RF_model_predict_shelling.predict(X_test_select)    
            
            test_data['next_shelling_pred']=Y_pred_shelling
            
            Y_pred_shelling = Y_pred_shelling.reshape(-1, 1)
            Y_test_shelling = Y_test_shelling.reshape(-1, 1)

            
            mse = np.mean((Y_test_shelling - Y_pred_shelling)**2)
            mae = np.mean(np.abs(Y_test_shelling - Y_pred_shelling))
            print('mse:', mse)
            print('mae:', mae)      
            globals()['RF_mse_list_k_'+str(k)][times] = mse
            globals()['RF_mae_list_k_'+str(k)][times] = mae
            
            train_data_future,val_data_future=method.split_by_eyetags_train_test(train_data,0.2)  
            for m in model_list:
                '''
                預測卵巢成熟
                '''
                print('---------------- '+m+' predict_future(add next shelling)(seperate) -------------')
                for future_count in range(futureDay):                
                    X_train=train_data_future.drop(["Date","Eyetag","next_shelling"], axis=1)
                    X_val=val_data_future.drop(["Date","Eyetag","next_shelling"], axis=1)
                    Y_train=pd.DataFrame()
                    Y_val=pd.DataFrame()
                    for f in range(futureDay):
                        X_train=X_train.drop(["predict_"+str(f+1)], axis=1)
                        X_val=X_val.drop(["predict_"+str(f+1)], axis=1)   
                    Y_train = pd.concat([Y_train, train_data_future["predict_"+str(future_count+1)]], axis=1)
                    Y_val = pd.concat([Y_val, val_data_future["predict_"+str(future_count+1)]], axis=1)       
                    
                    X_train_select=X_train.loc[:,selected_feature_future[count]]
                    X_train_select = pd.concat([X_train_select, train_data_future["next_shelling"]], axis=1)
                    X_val_select=X_val.loc[:,selected_feature_future[count]]
                    X_val_select = pd.concat([X_val_select, val_data_future["next_shelling"]], axis=1)
                    Y_train_select=Y_train
                    Y_val_select=Y_val
              
                    X_train_select=np.array(X_train_select)
                    Y_train_select=np.array(Y_train_select)
                    X_val_select=np.array(X_val_select)
                    Y_val_select=np.array(Y_val_select)
        
                    # 將資料進行正規化
                    X_train_select,X_train_scaler=method.normalize(X_train_select) 
                    Y_train_select,Y_train_scaler=method.normalize(Y_train_select)
                    X_val_select,X_val_scaler=method.normalize(X_val_select) 
                    Y_val_select,Y_val_scaler=method.normalize(Y_val_select) 
                    
                    #將資料打散
                    X_train_select, Y_train_select = method.shuffle(X_train_select, Y_train_select)
                    X_val_select, Y_val_select = method.shuffle(X_val_select, Y_val_select)
                    
                    Y_train_select = Y_train_select.ravel()
                    Y_val_select = Y_val_select.ravel()
                    
                    X_test=test_data.drop(["Date","Eyetag","next_shelling"], axis=1)
                    Y_test_future=pd.DataFrame()
                    for f in range(futureDay):
                        X_test=X_test.drop(["predict_"+str(f+1)], axis=1)
                    Y_test_future = pd.concat([Y_test_future, test_data["predict_"+str(future_count+1)]], axis=1)
                    
                    X_test_select=X_test.loc[:,selected_feature_future[count]]
                    X_test_select = pd.concat([X_test_select, test_data["next_shelling_pred"]], axis=1)
                    
                    Y_test_future=Y_test_future
                    X_test_select=np.array(X_test_select)
                    Y_test_future=np.array(Y_test_future)
                    X_test_select,X_test_scaler=method.normalize(X_test_select)
                    Y_test_future,Y_test_scaler=method.normalize(Y_test_future) 
                    X_test_select, Y_test_future = method.shuffle(X_test_select, Y_test_future)
                    
                    
                    if m=='RF':
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                        
                        model = RandomForestClassifier()

                        #'n_estimators': [50,100,200,500],
                        #'min_samples_leaf': [1,2,4],
                        #'max_depth': [None, 5, 10]
                        param_grid = {
                            'n_estimators': [100,200,400,600],
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future=RandomForestClassifier(**grid_search.best_params_)
                        predict_future.fit(X_train_select, Y_train_select)
                        #predict_future = RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=200,class_weight='balanced')                                
                    
                    elif m=='DT': 
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                        '''
                        model = DecisionTreeClassifier()
                        
                        #'criterion': ['gini', 'entropy'],
                        #'splitter': ['best', 'random'],
                        #'max_depth': [None, 5, 10],
                        #'min_samples_leaf': [1, 2, 4]
                        param_grid = {
                            'criterion': ['gini'],
                            'splitter': ['best'],
                            'max_depth': [None],
                            'min_samples_leaf': [1]
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future=DecisionTreeClassifier(**grid_search.best_params_)
                        predict_future.fit(X_train_select, Y_train_select)
                        '''
                        predict_future = DecisionTreeClassifier()
                        predict_future.fit(X_train_select, Y_train_select)
                        
                    elif m=='KNN': 
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                        
                        '''
                        model = KNeighborsClassifier()
                        #'n_neighbors': [3,4,5],
                        #'weights': ['uniform', 'distance']  
                        param_grid = {
                            'n_neighbors': [5],
                            'weights': ['uniform', 'distance']    
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future=KNeighborsClassifier(**grid_search.best_params_)
                        predict_future.fit(X_train_select, Y_train_select)
                        '''
                        predict_future = KNeighborsClassifier(n_neighbors=5)
                        predict_future.fit(X_train_select, Y_train_select)
                        
                    elif m=='SVM':
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                        '''
                        model= SVC()
                        #'C': [0.1, 1, 10, 100],
                        #'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                        #'gamma': [0.1, 0.01, 0.001, 0.0001],
                        #'class_weight':[None,'balanced']
                        param_grid = {
                            'C': [10, 100],
                            'kernel': ['rbf'],
                            'gamma': [0.1],
                            'class_weight':['balanced']
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future=SVC(**grid_search.best_params_)
                        '''
                        predict_future=SVC(C=10,kernel='rbf',gamma=0.1,probability=True)
                        predict_future.fit(X_train_select, Y_train_select)
                        
                    elif m=='ADA':
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                       
                        '''
                        model = AdaBoostClassifier()
                        #'n_estimators': [50, 100, 200, 500],
                        #'learning_rate': [0.01, 0.1, 0.5, 1.0]
                        param_grid = {
                            'n_estimators': [200],
                            'learning_rate': [0.1, 1.0]
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future=AdaBoostClassifier(**grid_search.best_params_)
                        '''
                        predict_future=AdaBoostClassifier(n_estimators=200,learning_rate=0.1)
                        predict_future.fit(X_train_select, Y_train_select)
    
                    elif m=='XGB':
                        X_train_select = np.concatenate((X_train_select, X_val_select), axis=0)
                        Y_train_select = np.concatenate((Y_train_select, Y_val_select), axis=0)
                       
                        '''
                        model=XGBClassifier()
                        #'n_estimators': [50, 100, 200, 500],
                        #'learning_rate': [0.01, 0.1, 0.5, 1.0],
                        #'max_depth': [3, 5, 7]
                        param_grid = {
                            'n_estimators': [200],
                            'learning_rate': [0.1],
                            'max_depth': [5]
                        }
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                        grid_search.fit(X_train_select, Y_train_select)
                        predict_future = XGBClassifier(**grid_search.best_params_)
                        '''
                        predict_future = XGBClassifier(n_estimators=200,learning_rate=0.1)
                        predict_future.fit(X_train_select, Y_train_select)
                    
                    elif m=='LSTM':
                        X_train_select = X_train_select.reshape(X_train_select.shape[0],1, X_train_select.shape[1])
                        X_val_select = X_val_select.reshape(X_val_select.shape[0],1, X_val_select.shape[1])
                        X_test_select = X_test_select.reshape(X_test_select.shape[0],1, X_test_select.shape[1])
                        
                        units=8
                        dropout_rate=0.1
                        activation='sigmoid'
                        batch_size=128
                        epochs=200
                        dense_units=1
                        
                        predict_future = method.buildManyToManyModel(units,dropout_rate,dense_units,activation,X_train_select.shape)
                        #模型訓練
                        callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")
                        predict_future.fit(X_train_select, Y_train_select, epochs=epochs, batch_size=batch_size, validation_data=(X_val_select, Y_val_select), callbacks=[callback],verbose=0)           
                    
                    elif m=='CNN':
                        X_train_select = X_train_select.reshape(X_train_select.shape[0],1, X_train_select.shape[1])
                        X_val_select = X_val_select.reshape(X_val_select.shape[0],1, X_val_select.shape[1])
                        X_test_select = X_test_select.reshape(X_test_select.shape[0],1, X_test_select.shape[1])
                   
                        filters=64
                        kernel_size=1
                        pool_size=1
                        dropout_rate=0.1
                        batch_size=128
                        epochs=200
                        dense_units=1
                        predict_future=method.build_CNN_model(filters, kernel_size, pool_size, dense_units, dropout_rate,X_train_select.shape)
                        #模型訓練
                        callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")
                        predict_future.fit(X_train_select, Y_train_select, epochs=epochs, batch_size=batch_size, validation_data=(X_val_select, Y_val_select), callbacks=[callback],verbose=0)
                        # 預測測試集
                        Y_pred_future = predict_future.predict(X_test_select)
                        
                    Y_pred_future = predict_future.predict(X_test_select) 
                    Y_pred_future = (Y_pred_future > 0.5).astype(int)
                    sensitivity, specificity ,precision,accuracy= method.calculate_metrics(Y_test_future.ravel(), Y_pred_future.ravel(),0.5) #根據threshold計算sensitivity, specificity ,accuracy
                    f1 = f1_score(Y_test_future.ravel(), Y_pred_future.ravel())
                    
                    
                    # 計算ROC曲線的真陽性率和假陽性率
                    if m =='LSTM' or m=='CNN' or m=='DT':
                        Y_pred_prob = predict_future.predict(X_test_select)
                    else:
                        Y_pred_prob = predict_future.predict_proba(X_test_select)
                        Y_pred_prob = Y_pred_prob[:, 1]
                    fpr, tpr, thresholds = roc_curve(Y_test_future, Y_pred_prob)
                    # 計算AUC (Area Under the Curve)
                    roc_auc = auc(fpr, tpr)
                    
                    
                    print("---------Day %d-------------"%(future_count+1))
                    print("Best parameters:",grid_search.best_params_)
                    print('Accuracy:', accuracy)
                    print('Precision:', precision)
                    print('Sensitivity:', sensitivity)
                    print('Specificity:', specificity)
                    print("F1 Score:", f1)
                    print("ROC auc:", roc_auc)
                    print("")
                    
                    globals()[m+'_sensitivity_list_k_'+str(k)][times][future_count] = sensitivity
                    globals()[m+'_specificity_list_k_'+str(k)][times][future_count]= specificity
                    globals()[m+'_precision_list_k_'+str(k)][times][future_count]= precision
                    globals()[m+'_accuracy_list_k_'+str(k)][times][future_count]= accuracy
                    globals()[m+'_f1_score_list_k_'+str(k)][times][future_count]= f1
                    globals()[m+'_fpr_list_k_'+str(k)][times][future_count]= fpr
                    globals()[m+'_tpr_list_k_'+str(k)][times][future_count]= tpr
                    globals()[m+'_roc_auc_list_k_'+str(k)][times][future_count]= roc_auc
            
                end = time.process_time()
                print("執行時間：%f 秒" % (end - start))
                globals()[m+'_time_list_k_'+str(k)][times]= end - start
            count+=1
#%%

#%%
'''
計算預測結果的平均值
'''
def calculate_average_shelling(result):
    average=[]
    temp=0
    for i in range(len(result)):
        temp=temp+result[i]
    temp=temp/len(result)
    average.append(temp)
    return average   

metric_list=['accuracy','precision','sensitivity','specificity','f1_score','roc_auc']
for metric in metric_list:
    for m in model_list:
        for i in k_list:
            globals()[m+'_'+metric+'_average_k_'+str(i)]=method.calculate_average(globals()[m+'_'+metric+'_list_k_'+str(i)])

#%%
'''
將結果存起來
'''
metric_list=['accuracy','precision','sensitivity','specificity','f1_score','roc_auc',]
result_all=pd.DataFrame()
for k in k_list:
    result=pd.DataFrame()
    for m in model_list:
        for metric in metric_list:
            result[m+'_'+metric+'(k='+str(k)+')']=globals()[m+'_'+metric+'_average_k_'+str(k)]      
    result_all=result_all.append(result,ignore_index=True)
result_all.to_csv('result/2.csv', index=False)


   
   
   
   
   
   
   
   


