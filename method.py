import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input,Conv1D, MaxPooling1D,Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from collections import Counter
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression,RFE

"""
# 對 features 進行歸一化
max_vals = np.max(train, axis=(0, 1))  # 沿著前兩個維度取每個 feature 的最大值
min_vals = np.min(train, axis=(0, 1))  # 沿著前兩個維度取每個 feature 的最小值
train = (train - min_vals) / (max_vals - min_vals)

# 對 samples 和 timesteps 進行歸一化
mean_vals = np.mean(train, axis=(0, 1))  # 沿著前兩個維度取均值
std_vals = np.std(train, axis=(0, 1))  # 沿著前兩個維度取標準差
train_norm = (train - mean_vals) / std_vals
"""
def normalize(train):
    # 假設資料儲存在 data 中，形狀為 (3000, 20, 18)
    # 將資料攤平成二維陣列，形狀變為 (3000*20, 18)

    data_2d = train.reshape(-1, train.shape[-1])
    
    # 初始化 MinMaxScaler，將特徵縮放到 [0, 1] 的區間
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 進行正規化
    data_normalized_2d = scaler.fit_transform(data_2d)
    
    # 將資料還原為三維形狀 (3000, 20, 18)
    data_normalized = data_normalized_2d.reshape(train.shape)
    return data_normalized,scaler


'''
將資料打散，而非照日期排序
'''
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

'''
將Training Data取一部份當作Validation Data
'''
def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate)*2:]
    Y_train = Y[int(Y.shape[0]*rate)*2:]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    X_test=X[int(X.shape[0]*rate):int(X.shape[0]*rate)*2]
    Y_test=Y[int(Y.shape[0]*rate):int(Y.shape[0]*rate)*2]
    return X_train, Y_train, X_val, Y_val,X_test,Y_test

'''
將記錄天數不夠用來預測的蝦隻資料刪掉
'''
def drop_not_enough_data(data,pastDay,futureDay):
    Eyetags = set(data['Eyetag'])
    for eyetag in Eyetags:
        if len(data[data['Eyetag']==eyetag])<pastDay+futureDay:
               data=data.drop(data[data['Eyetag']==eyetag].index)
    return data

'''
feature selection
'''
def rfe_feature_selection_future(X, Y,pastDay,futureDay,Max_pastDay,k_list):       
    # 定義特徵篩選器和LSTM模型
    
    model=RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=200,class_weight='balanced')
    #Y = Y.values.ravel()
    
    #Random forest feature importance
    model.fit(X, Y)
    col2 = X.columns
    sorted_idx = model.feature_importances_.argsort()    # 從小排到大
    col2[sorted_idx]
    plt.figure(figsize=(10,25))
    plt.barh(col2[sorted_idx], model.feature_importances_[sorted_idx],height=0.8)
    plt.xlabel('Feature importance',size=16)
    plt.ylabel('Feature',size=16)
    plt.title('RF Feature Importance(future)',size=16)
    plt.show()
    
    
    selected_feature=[]
    rfe = RFE(estimator=model, n_features_to_select=1)
    selected_features = rfe.fit_transform(X, Y)
    feature_name=X.columns
    # 獲取特徵排名
    feature_ranking = rfe.ranking_
    # 根據特徵重要性排序的索引
    sorted_indices = np.argsort(-feature_ranking)  # 由高到低排序
    
    # 根據排序後的索引重新排列特徵名稱和特徵重要性
    sorted_feature_names = [feature_name[i] for i in sorted_indices]
    sorted_feature_importance = feature_ranking[sorted_indices]


    # 繪製條形圖
    plt.figure(figsize=(10,25))
    plt.barh(sorted_feature_names, sorted_feature_importance,height=0.8)   
    #plt.yticks(range(len(feature_name)), feature_name)
    plt.xlabel('Feature Ranking',size=16)
    plt.ylabel('Feature',size=16)
    plt.title('RFE Feature selection(future)',size=16)
    plt.show()
    
    selected_feature=[]
    i=0
    for k in k_list:
        # 根據排名獲取前k個特徵的索引
        selected_features = np.where(feature_ranking <= k)[0]
        selected_feature_names = [feature_name[idx] for idx in selected_features]
        
        #feature_mask = rfe.support_
        # 獲取篩選後的特徵名稱
        #selected_feature_names = [feature_name for feature_name, selected in zip(feature_name, feature_mask) if selected]
        # 打印篩選後的特徵名稱
        selected_feature.append(selected_feature_names)
        print("---------------k=%d----------------"%(k))
        print(selected_feature[i])
        i+=1

    return selected_feature

def rfe_feature_selection_shelling(X, Y,pastDay,futureDay,Max_pastDay,k_list):       
    model=RandomForestRegressor(max_depth=20, min_samples_leaf=5, n_estimators=200)
    Y = Y.values.ravel()
    
    #Random forest feature importance
    model.fit(X, Y)
    col2 = X.columns
    sorted_idx = model.feature_importances_.argsort()    # 從小排到大
    col2[sorted_idx]
    plt.figure(figsize=(10,25))
    plt.barh(col2[sorted_idx], model.feature_importances_[sorted_idx],height=0.8)
    plt.xlabel('Feature importance',size=16)
    plt.ylabel('Feature',size=16)
    plt.title('RF Feature Importance(shelling)',size=16)
    plt.show()
    
    #Y = Y.values.ravel()
    selected_feature=[]
    rfe = RFE(estimator=model, n_features_to_select=1)
    selected_features = rfe.fit_transform(X, Y)
    feature_name=X.columns
    # 獲取特徵排名
    feature_ranking = rfe.ranking_
    # 根據特徵重要性排序的索引
    sorted_indices = np.argsort(-feature_ranking)  # 由高到低排序
    
    # 根據排序後的索引重新排列特徵名稱和特徵重要性
    sorted_feature_names = [feature_name[i] for i in sorted_indices]
    sorted_feature_importance = feature_ranking[sorted_indices]
    
    
    # 繪製條形圖
    plt.figure(figsize=(10,25))
    plt.barh(sorted_feature_names, sorted_feature_importance,height=0.8)   
    #plt.barh(range(len(feature_name)), feature_ranking.argsort(), align='center')
    #plt.yticks(range(len(feature_name)), feature_name)
    plt.xlabel('Feature Ranking',size=16)
    plt.ylabel('Feature',size=16)
    plt.title('RFE Feature selection(shelling)',size=16)
    plt.show()
    
    i=0
    selected_feature=[]
    for k in k_list:
        # 根據排名獲取前k個特徵的索引
        selected_features = np.where(feature_ranking <= k)[0]
        selected_feature_names = [feature_name[idx] for idx in selected_features]
        
        #feature_mask = rfe.support_
        # 獲取篩選後的特徵名稱
        #selected_feature_names = [feature_name for feature_name, selected in zip(feature_name, feature_mask) if selected]
        # 打印篩選後的特徵名稱
        selected_feature.append(selected_feature_names)
        print("---------------k=%d----------------"%(k))
        print(selected_feature[i])
        i+=1
    return selected_feature

def feature_importance(data,model):
    col2 = data.columns
    sorted_idx = model.feature_importances_.argsort()    # 從小排到大
    col2[sorted_idx]
    plt.figure(figsize=(10,25))
    plt.barh(col2[sorted_idx], model.feature_importances_[sorted_idx],height=0.8)
    plt.title('Feature Importance',size=16)
    plt.show()

'''
區分測試集和訓練集
'''               
def split_by_eyetags(data,rate):
    groups = data.groupby('Eyetag')
    # 將所有群組洗牌
    group_names = list(groups.groups.keys())
    np.random.shuffle(group_names)
    shuffled_groups = [groups.get_group(name) for name in group_names]
    
    # 計算訓練、驗證和測試集的大小
    total_size = len(shuffled_groups)
    train_size = int(total_size * (1-rate*2))
    val_size = int(total_size * rate)
    test_size = total_size - train_size - val_size
    
    # 將所有群組分配到訓練、驗證和測試集
    train_groups = shuffled_groups[:train_size]
    val_groups = shuffled_groups[train_size:train_size+val_size]
    test_groups = shuffled_groups[train_size+val_size:]
    
    # 將每個群組中的資料按照時間排序
    train_groups = [group.sort_values("Date") for group in train_groups]
    val_groups = [group.sort_values("Date") for group in val_groups]
    test_groups = [group.sort_values("Date") for group in test_groups]
    
    train_data = pd.concat(train_groups)
    val_data = pd.concat(val_groups)
    test_data = pd.concat(test_groups)
        
    return train_data,val_data,test_data 

def split_by_eyetags_train_test(data,rate):
    groups = data.groupby('Eyetag')
    # 將所有群組洗牌
    group_names = list(groups.groups.keys())
    np.random.shuffle(group_names)
    shuffled_groups = [groups.get_group(name) for name in group_names]
    
    # 計算訓練、驗證和測試集的大小
    total_size = len(shuffled_groups)
    train_size = int(total_size * (1-rate))
    test_size = total_size - train_size 
    
    # 將所有群組分配到訓練、驗證和測試集
    train_groups = shuffled_groups[:train_size]
    test_groups = shuffled_groups[train_size:]
    
    # 將每個群組中的資料按照時間排序
    train_groups = [group.sort_values("Date") for group in train_groups]
    test_groups = [group.sort_values("Date") for group in test_groups]
    
    train_data = pd.concat(train_groups)
    test_data = pd.concat(test_groups)
        
    return train_data,test_data

'''
計算脫殼天數
'''
def caculate_shelling_days(data,Eyetag):  
    all_shelling_daycount=[]
    eyetag_drop=[]
    for eyetag in Eyetag:          #一隻一隻蝦計算
        temp=data[data['Eyetag']==eyetag]
        count_shell=0
        shelling_daycount=[]  #用來存脫殼周期  
        for i in range(len(temp.index)):
            count_shell+=1 
            if temp['shelling'].iloc[i]==1 or temp['moulting'].iloc[i]==1:   #判斷是否遇到脫殼       
                shelling_daycount.append(count_shell)
                all_shelling_daycount.append(count_shell)
                count_shell=0
        if shelling_daycount==[]:
            eyetag_drop.append(eyetag)
        else:            
            for s in shelling_daycount:
                if s>=30 or s<=7:
                    eyetag_drop.append(eyetag)
                    break
        #print(eyetag,": ",shelling_daycount)
    return all_shelling_daycount,eyetag_drop

'''
LSTM模型
建立了一個具有 2 層 LSTM 層和一個全連接層的模型。模型的輸出是一個介於 0 和 1 之間的值，用於表示蝦隻卵巢成熟的機率。
使用 binary_crossentropy 作為損失函數，因為這是一個二元分類問題。
進行二元分類任務時，sigmoid 函數可以將輸出映射到 [0, 1] 的範圍內，並且可以將輸出解釋為預測樣本屬於正類的機率。
'''
def buildManyToManyModel(lstm_units,dropout_rate,dense_units,activation,shape):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_length=shape[1], input_dim=shape[2]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=dense_units, activation=activation))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model


'''
CNN 模型
'''
def build_CNN_model(filters, kernel_size, pool_size, dense_units, dropout_rate,shape):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(Dense(units=dense_units, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



'''
特徵重要性
'''
def featureImportance(X_train, X_valid,y_train, y_valid,model,COLS):
    K.clear_session()
    #COLS = list(X_train.columns
    # 计算特征重要性
    results = []
    print(' Computing LSTM feature importance...')
    for k in range(len(COLS)):
        if k>0: 
            save_col = X_valid[:,:,k-1].copy()
            np.random.shuffle(X_valid[:,:,k-1])
            
        oof_preds = model.predict(X_valid, verbose=0).squeeze() 
        mae = np.mean(np.abs( oof_preds-y_valid ))
        results.append({'feature':COLS[k],'mae':mae})
        if k>0: 
            X_valid[:,:,k-1] = save_col
 
    # 展示特征重要性
    print()
    df = pd.DataFrame(results)
    df = df.sort_values('mae')
    plt.figure(figsize=(10,20))
    plt.barh(np.arange(len(COLS)),df.mae)
    plt.yticks(np.arange(len(COLS)),df.feature.values)
    plt.title('LSTM Feature Importance',size=16)
    plt.ylim((-1,len(COLS)))
    plt.show()
                       
    # SAVE LSTM FEATURE IMPORTANCE
    df = df.sort_values('mae',ascending=False)
    df.to_csv(f'lstm_feature_importance.csv',index=False)
    
    
'''
決定卵巢成熟的區分方式
'''                           
'''
0:0,0-I
1:I,I-II,II,II-III,III
2:生產
3:脫殼
4:受精
'''  
def label_1(data_set):
    #建立一個用來預測的label
    data_set['life_label']=data_set['life_encode']
    data_set.loc[data_set.life_encode==0,'life_label']=0   #0
    data_set.loc[data_set.life_encode==1,'life_label']=0   #0-I
    data_set.loc[data_set.life_encode==2,'life_label']=1   #I
    data_set.loc[data_set.life_encode==3,'life_label']=1   #I-II
    data_set.loc[data_set.life_encode==4,'life_label']=1   #II
    data_set.loc[data_set.life_encode==5,'life_label']=1   #II-III
    data_set.loc[data_set.life_encode==6,'life_label']=1   #III
    data_set.loc[data_set.life_encode==7,'life_label']=6   #人工授精
    data_set.loc[data_set.life_encode==8,'life_label']=5   #剪眼
    data_set.loc[data_set.life_encode==9,'life_label']=0   #死亡
    data_set.loc[data_set.life_encode==10,'life_label']=2  #生產
    data_set.loc[data_set.life_encode==11,'life_label']=3  #脫殼
    data_set.loc[data_set.life_encode==12,'life_label']=4  #脫殼受精
    return data_set

'''
0:0  
1:0-I,I,I-II,II,II-III,III
'''
def label_2(data_set):
    #建立一個用來預測的label
    data_set['life_label']=data_set['life_encode']
    data_set.loc[data_set.life_encode==0,'life_label']=0   #0
    data_set.loc[data_set.life_encode==1,'life_label']=1   #0-I
    data_set.loc[data_set.life_encode==2,'life_label']=1   #I
    data_set.loc[data_set.life_encode==3,'life_label']=1   #I-II
    data_set.loc[data_set.life_encode==4,'life_label']=1   #II
    data_set.loc[data_set.life_encode==5,'life_label']=1   #II-III
    data_set.loc[data_set.life_encode==6,'life_label']=1   #III
    data_set.loc[data_set.life_encode==7,'life_label']=1   #人工授精
    data_set.loc[data_set.life_encode==8,'life_label']=0   #剪眼
    data_set.loc[data_set.life_encode==9,'life_label']=1   #死亡
    data_set.loc[data_set.life_encode==10,'life_label']=1  #生產
    data_set.loc[data_set.life_encode==11,'life_label']=1  #脫殼
    data_set.loc[data_set.life_encode==12,'life_label']=1  #脫殼受精
    return data_set

'''
0:0  
1:0-I,I,I-II,II,II-III,III,生產
'''
def label_3(data_set):
    #建立一個用來預測的label
    data_set['life_label']=data_set['life_encode']
    data_set.loc[data_set.life_encode==0,'life_label']=0   #0
    data_set.loc[data_set.life_encode==1,'life_label']=1   #0-I
    data_set.loc[data_set.life_encode==2,'life_label']=1   #I
    data_set.loc[data_set.life_encode==3,'life_label']=1   #I-II
    data_set.loc[data_set.life_encode==4,'life_label']=1   #II
    data_set.loc[data_set.life_encode==5,'life_label']=1   #II-III
    data_set.loc[data_set.life_encode==6,'life_label']=1   #III
    data_set.loc[data_set.life_encode==7,'life_label']=0   #人工授精
    data_set.loc[data_set.life_encode==8,'life_label']=0   #剪眼
    data_set.loc[data_set.life_encode==9,'life_label']=0   #死亡
    data_set.loc[data_set.life_encode==10,'life_label']=1  #生產
    data_set.loc[data_set.life_encode==11,'life_label']=0  #脫殼
    data_set.loc[data_set.life_encode==12,'life_label']=0  #脫殼受精
    return data_set

'''
0:0,0-I,I,脫殼,脫殼受精
1:I-II,II,II-III,III,生產
'''
def label_4(data_set):
    #建立一個用來預測的label
    data_set['life_label']=data_set['life_encode']
    data_set.loc[data_set.life_encode==0,'life_label']=0   #0
    data_set.loc[data_set.life_encode==1,'life_label']=0   #0-I
    data_set.loc[data_set.life_encode==2,'life_label']=0   #I
    data_set.loc[data_set.life_encode==3,'life_label']=1   #I-II
    data_set.loc[data_set.life_encode==4,'life_label']=1   #II
    data_set.loc[data_set.life_encode==5,'life_label']=1   #II-III
    data_set.loc[data_set.life_encode==6,'life_label']=1   #III
    data_set.loc[data_set.life_encode==7,'life_label']=1   #人工授精
    data_set.loc[data_set.life_encode==8,'life_label']=0   #剪眼
    data_set.loc[data_set.life_encode==9,'life_label']=1   #死亡
    data_set.loc[data_set.life_encode==10,'life_label']=1  #生產
    data_set.loc[data_set.life_encode==11,'life_label']=0  #脫殼
    data_set.loc[data_set.life_encode==12,'life_label']=0  #脫殼受精
    return data_set

'''
0:0,0-I,脫殼,脫殼受精
1:I,I-II,II,II-III,III,生產
'''
def label_5(data_set):
    #建立一個用來預測的label
    data_set['life_label']=data_set['life_encode']
    data_set.loc[data_set.life_encode==0,'life_label']=0   #0
    data_set.loc[data_set.life_encode==1,'life_label']=0   #0-I
    data_set.loc[data_set.life_encode==2,'life_label']=1   #I
    data_set.loc[data_set.life_encode==3,'life_label']=1   #I-II
    data_set.loc[data_set.life_encode==4,'life_label']=1   #II
    data_set.loc[data_set.life_encode==5,'life_label']=1   #II-III
    data_set.loc[data_set.life_encode==6,'life_label']=1   #III
    data_set.loc[data_set.life_encode==7,'life_label']=1   #人工授精
    data_set.loc[data_set.life_encode==8,'life_label']=0   #剪眼
    data_set.loc[data_set.life_encode==9,'life_label']=1   #死亡
    data_set.loc[data_set.life_encode==10,'life_label']=1  #生產
    data_set.loc[data_set.life_encode==11,'life_label']=0  #脫殼
    data_set.loc[data_set.life_encode==12,'life_label']=0  #脫殼受精
    return data_set

def drawLoss(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.show()


'''
ROC 曲線法：通過計算 ROC 曲線下的面積（AUC）可以評估分類器的整體性能，而選擇閾值的方法就是選擇 ROC 曲線最接近左上角（最佳性能）的點對應的閾值。    
'''
def roc_curve_threshold(Y_test, Y_pred):
    # 計算 ROC 曲線和 AUC
    fpr, tpr, thresholds = roc_curve(Y_test.ravel(), Y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    # 繪製 ROC 曲線
    '''
    plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''
    # 選擇閾值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold:", optimal_threshold)
    
    return optimal_threshold

'''
計算 sensitivity 和 specificity
'''
def calculate_metrics(actual, predicted, threshold):
    tp = np.sum((actual == 1) & (predicted >= threshold))
    tn = np.sum((actual == 0) & (predicted < threshold))
    fp = np.sum((actual == 0) & (predicted >= threshold))
    fn = np.sum((actual == 1) & (predicted < threshold))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision=tp/(tp+fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, specificity,precision,accuracy


'''
計算平均值
'''
def calculate_average(result):
    average=[]
    for i in range(len(result[0])):
        temp=0
        for j in range(len(result)):
            temp=temp+result[j][i]
        temp=temp/len(result)
        average.append(temp)
    return average


'''
CNN使用gird_search找最佳參數
'''
'''
預測卵巢成熟
#Best Parameters: {'dense_units': 128, 'dropout_rate': 0.1, 'filters': 64, 'kernel_size': 3, 'pool_size': 2}
#Best Parameters: {'dense_units': 128, 'dropout_rate': 0.1, 'filters': 32, 'kernel_size': 3, 'pool_size': 2}
'''
def cnn_grid_search(X_train, Y_train,X_val, Y_val):

    def create_model(filters=32, kernel_size=3, pool_size=2, dense_units=64, dropout_rate=0.2):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(units=dense_units, activation='relu'))
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(Dense(units=10, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        return model

    print("---------- CNN Grid Search ------------")
    # 創建CNN模型
    model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
    
    # 定義要調整的參數和參數範圍
    param_grid = {
        'filters': [16,32, 64],
        'kernel_size': [3, 5],
        'pool_size': [2, 3],
        'dense_units': [64, 128],
        'dropout_rate': [0.1,0.2]
    }
    
    # 創建Grid Search物件
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    
    # 訓練模型
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="auto")
    grid_result = grid.fit(X_train, Y_train,epochs=100,verbose=0, batch_size=128,validation_data=(X_val, Y_val),callbacks=[callback])

    # 打印最佳參數和分數
    print("Best Parameters: ", grid_result.best_params_)
    print("Best Score: ", grid_result.best_score_)

'''
預測脫殼時間
'''    
def cnn_grid_search_shelling(X_train, Y_train,X_val, Y_val):

    def create_model(filters=32, kernel_size=3, pool_size=2, dense_units=64, dropout_rate=0.2):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(units=dense_units, activation='relu'))
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        return model

    print("---------- CNN Grid Search ------------")
    # 創建CNN模型
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
    
    # 定義要調整的參數和參數範圍
    param_grid = {
        'filters': [16,32, 64],
        'kernel_size': [3, 5],
        'pool_size': [2, 3],
        'dense_units': [64, 128],
        'dropout_rate': [0.1,0.2]
    }
    
    # 創建Grid Search物件
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    
    # 訓練模型
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="auto")
    grid_result = grid.fit(X_train, Y_train,epochs=100,verbose=0, batch_size=128,validation_data=(X_val, Y_val),callbacks=[callback])

    # 打印最佳參數和分數
    print("Best Parameters: ", grid_result.best_params_)
    print("Best Score: ", grid_result.best_score_)



'''
LSTM使用gird_search找最佳參數
'''    
'''
預測卵巢成熟
'''
def lstm_grid_search(X_train, Y_train,X_val, Y_val):

    def create_model(units=8, dropout_rate=0,activation='sigmoid'):
        model = Sequential()
        model.add(LSTM(units=units, input_length=X_train.shape[1], input_dim=X_train.shape[2]))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=10, activation=activation))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        return model

    print("---------- LSTM Grid Search ------------")
    # 創建CNN模型
    model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
    
    # 定義要調整的參數和參數範圍
    param_grid = {
        'units': [8,16,32, 64],
        'dropout_rate': [0,0.1,0.2],
        'activation': ['sigmoid', 'relu']
    }
    
    # 創建Grid Search物件
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    
    # 訓練模型
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="auto")
    grid_result = grid.fit(X_train, Y_train,epochs=100,verbose=0, batch_size=128,validation_data=(X_val, Y_val),callbacks=[callback])

    # 打印最佳參數和分數
    print("Best Parameters: ", grid_result.best_params_)
    print("Best Score: ", grid_result.best_score_)    

'''
預測脫殼時間
''' 
def lstm_grid_search_shelling(X_train, Y_train,X_val, Y_val):

    def create_model(units=8, dropout_rate=0,activation='sigmoid'):
        model = Sequential()
        model.add(LSTM(units=units, input_length=X_train.shape[1], input_dim=X_train.shape[2]))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1, activation=activation))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        return model

    print("---------- LSTM Grid Search ------------")
    # 創建CNN模型
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
    
    # 定義要調整的參數和參數範圍
    param_grid = {
        'units': [8,16,32, 64],
        'dropout_rate': [0,0.1,0.2],
        'activation': ['sigmoid', 'relu','linear']
    }
    
    # 創建Grid Search物件
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    
    # 訓練模型
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="auto")
    grid_result = grid.fit(X_train, Y_train,epochs=100,verbose=0, batch_size=128,validation_data=(X_val, Y_val),callbacks=[callback])

    # 打印最佳參數和分數
    print("Best Parameters: ", grid_result.best_params_)
    print("Best Score: ", grid_result.best_score_)    



