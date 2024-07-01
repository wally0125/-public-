import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%
'''
比較不同的model預測結果
'''
def draw_plot_predict_future_compare_model(data,evaluation_index,k):
    fig = plt.figure()
    x = list(range(1,8))
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k='+str(k)+')']),color='#845EC2',linestyle="-",marker=".", label="RF")
    plt.plot(x, np.squeeze(data['DT_'+evaluation_index+'(k='+str(k)+')']),color='#D65DB1',linestyle="-",marker=".", label="DT")
    plt.plot(x, np.squeeze(data['KNN_'+evaluation_index+'(k='+str(k)+')']),color='#4B4453',linestyle="-",marker=".", label="kNN")
    plt.plot(x, np.squeeze(data['SVM_'+evaluation_index+'(k='+str(k)+')']),color='#FF6F91',linestyle="-",marker=".", label="SVM")
    plt.plot(x, np.squeeze(data['ADA_'+evaluation_index+'(k='+str(k)+')']),color='#B0A8B9',linestyle="-",marker=".", label="AdaBoost")
    plt.plot(x, np.squeeze(data['XGB_'+evaluation_index+'(k='+str(k)+')']),color='#FF9671',linestyle="-",marker=".", label="XgBoost")
    plt.plot(x, np.squeeze(data['LSTM_'+evaluation_index+'(k='+str(k)+')']),color='#FFC75F',linestyle="-",marker=".", label="LSTM")
    plt.plot(x, np.squeeze(data['CNN_'+evaluation_index+'(k='+str(k)+')']),color='#F9F871',linestyle="-",marker=".", label="CNN")
    #plt.plot(x, np.squeeze(data[evaluation_index+'(k=101)']),color='#F07B3F',linestyle="-",marker=".", label="k=101")
    
    if evaluation_index=='accuracy':
        pic_name='Accuracy'
    elif evaluation_index=='precision':
        pic_name='Precision'
    elif evaluation_index=='sensitivity':
        pic_name='Sensitivity'
    elif evaluation_index=='specificity':
        pic_name='Specificity'
    elif evaluation_index=='f1_score':
        pic_name='F1-score'
    elif evaluation_index=='roc_auc':
        pic_name='ROC AUC'
    
    plt.ylim(0.3, 1) # 設定 y 軸座標範圍
    plt.xlabel('predict_day', fontsize='12') # 設定 x 軸標題內容及大小
    plt.ylabel(pic_name, fontsize='12') # 設定 y 軸標題內容及大小
    plt.title(pic_name, fontsize='18') # 設定圖表標題內容及大小
    plt.legend(fontsize='7.5',loc='lower left')
    plt.show()
    fig.savefig('pic/result'+'('+str(evaluation_index)+').png')
#%%
k_list=[83]
result=pd.read_csv('result/predict_mix_final.csv')
model_list=['LSTM','CNN','RF','DT','KNN','SVM','ADA','XGB']
for k in k_list:
    draw_plot_predict_future_compare_model(result,evaluation_index='accuracy',k=k)
    draw_plot_predict_future_compare_model(result,evaluation_index='precision',k=k)
    draw_plot_predict_future_compare_model(result,evaluation_index='sensitivity',k=k)
    draw_plot_predict_future_compare_model(result,evaluation_index='specificity',k=k)
    draw_plot_predict_future_compare_model(result,evaluation_index='f1_score',k=k)
    draw_plot_predict_future_compare_model(result,evaluation_index='roc_auc',k=k)   
#%%
'''
比較選擇不同數目的特徵    
'''
def draw_plot_feature_select(data,evaluation_index):
    fig = plt.figure()
    x = list(range(1,8))
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=20)']),color='#845EC2',linestyle="-",marker=".", label="RF (feature_count=20)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=30)']),color='#D65DB1',linestyle="-",marker=".", label="RF (feature_count=30)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=40)']),color='#4B4453',linestyle="-",marker=".", label="RF (feature_count=40)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=50)']),color='#FF6F91',linestyle="-",marker=".", label="RF (feature_count=50)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=60)']),color='#B0A8B9',linestyle="-",marker=".", label="RF (feature_count=60)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(k=83)']),color='#FF9671',linestyle="-",marker=".", label="RF (feature_count=all)")
    #plt.plot(x, np.squeeze(data['LSTM_'+evaluation_index+'(k='+str(k)+')']),color='#FFC75F',linestyle="-",marker=".", label="LSTM")
    #plt.plot(x, np.squeeze(data['CNN_'+evaluation_index+'(k='+str(k)+')']),color='#F9F871',linestyle="-",marker=".", label="CNN")
    #plt.plot(x, np.squeeze(data[evaluation_index+'(k=101)']),color='#F07B3F',linestyle="-",marker=".", label="k=101")
    
    if evaluation_index=='accuracy':
        pic_name='Accuracy'
    elif evaluation_index=='precision':
        pic_name='Precision'
    elif evaluation_index=='sensitivity':
        pic_name='Sensitivity'
    elif evaluation_index=='specificity':
        pic_name='Specificity'
    elif evaluation_index=='f1_score':
        pic_name='F1-score'
    elif evaluation_index=='roc_auc':
        pic_name='ROC AUC'
    
    plt.ylim(0.3, 1) # 設定 y 軸座標範圍
    plt.xlabel('predict_day', fontsize='12') # 設定 x 軸標題內容及大小
    plt.ylabel(pic_name, fontsize='12') # 設定 y 軸標題內容及大小
    plt.title(pic_name, fontsize='18') # 設定圖表標題內容及大小
    plt.legend(fontsize='9',loc='lower left')
    plt.show()
    fig.savefig('pic/feature_select'+'('+str(evaluation_index)+').png')
#%%
result=pd.read_csv('result/RF_feature_select.csv')
model_list=['LSTM','CNN','RF','DT','KNN','SVM','ADA','XGB']
draw_plot_feature_select(result,evaluation_index='accuracy')
draw_plot_feature_select(result,evaluation_index='precision')
draw_plot_feature_select(result,evaluation_index='sensitivity')
draw_plot_feature_select(result,evaluation_index='specificity')
draw_plot_feature_select(result,evaluation_index='f1_score')
draw_plot_feature_select(result,evaluation_index='roc_auc')         
      


#%%         
'''
畫出各模型的roc_auc
'''
def draw_roc_auc(data,day,k):       
    fig = plt.figure()
    lw = 2
    plt.plot(data['RF_fpr'+'(k='+str(k)+')'][day], data['RF_tpr'+'(k='+str(k)+')'][day], color='#845EC2', lw=lw, label='RF (area = %0.3f)' % data['RF_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['DT_fpr'+'(k='+str(k)+')'][day], data['DT_tpr'+'(k='+str(k)+')'][day], color='#D65DB1', lw=lw, label='DT (area = %0.3f)' % data['DT_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['KNN_fpr'+'(k='+str(k)+')'][day], data['KNN_tpr'+'(k='+str(k)+')'][day], color='#4B4453', lw=lw, label='kNN (area = %0.3f)' % data['KNN_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['SVM_fpr'+'(k='+str(k)+')'][day], data['SVM_tpr'+'(k='+str(k)+')'][day], color='#FF6F91', lw=lw, label='SVM (area = %0.3f)' % data['SVM_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['ADA_fpr'+'(k='+str(k)+')'][day], data['ADA_tpr'+'(k='+str(k)+')'][day], color='#B0A8B9', lw=lw, label='AdaBoost (area = %0.3f)' % data['ADA_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['XGB_fpr'+'(k='+str(k)+')'][day], data['XGB_tpr'+'(k='+str(k)+')'][day], color='#FF9671', lw=lw, label='XgBoost (area = %0.3f)' % data['XGB_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['LSTM_fpr'+'(k='+str(k)+')'][day], data['LSTM_tpr'+'(k='+str(k)+')'][day], color='#FFC75F', lw=lw, label='LSTM (area = %0.3f)' % data['LSTM_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot(data['CNN_fpr'+'(k='+str(k)+')'][day], data['CNN_tpr'+'(k='+str(k)+')'][day], color='#F9F871', lw=lw, label='CNN (area = %0.3f)' % data['CNN_roc_auc'+'(k='+str(k)+')'][day])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic'+'(Day'+str(day+1)+')')
    plt.legend(fontsize='8',loc="lower right")
    fig.savefig('pic/roc'+str(day+1)+'.png')
    plt.show()      
      
result=pd.read_csv('result/roc_auc.csv')     

       
#%%       
'''
畫出有沒有使用脫殼時間當作特徵的結果
'''
def draw_plot_next_moiting_or_not(data,evaluation_index,k):
    fig = plt.figure()
    x = list(range(1,8))
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(with next molting)']),color='#845EC2',linestyle="-",marker=".", label="RF(with next molting)")
    plt.plot(x, np.squeeze(data['RF_'+evaluation_index+'(without next molting)']),color='#FF9671',linestyle="-",marker=".", label="RF(without next molting)")
    
    
    if evaluation_index=='accuracy':
        pic_name='Accuracy'
    elif evaluation_index=='precision':
        pic_name='Precision'
    elif evaluation_index=='sensitivity':
        pic_name='Sensitivity'
    elif evaluation_index=='specificity':
        pic_name='Specificity'
    elif evaluation_index=='f1_score':
        pic_name='F1-score'
    elif evaluation_index=='roc_auc':
        pic_name='ROC AUC'
    
    plt.ylim(0.3, 1) # 設定 y 軸座標範圍
    plt.xlabel('predict_day', fontsize='12') # 設定 x 軸標題內容及大小
    plt.ylabel(pic_name, fontsize='12') # 設定 y 軸標題內容及大小
    plt.title(pic_name, fontsize='18') # 設定圖表標題內容及大小
    plt.legend(fontsize='9',loc='lower left')
    plt.show()
    fig.savefig('pic/next_moiting_or_not'+'('+str(evaluation_index)+').png')  
       
result=pd.read_csv('result/with_or_without_next_molting.csv')

k=83
draw_plot_next_moiting_or_not(result,evaluation_index='accuracy',k=k)
draw_plot_next_moiting_or_not(result,evaluation_index='precision',k=k)
draw_plot_next_moiting_or_not(result,evaluation_index='sensitivity',k=k)
draw_plot_next_moiting_or_not(result,evaluation_index='specificity',k=k)
draw_plot_next_moiting_or_not(result,evaluation_index='f1_score',k=k)
