import math
import wave
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import serial

# Teams can add helper functions
# Add all helper functions here
import struct
import pandas as pd
from sklearn import svm
from pandas import DataFrame
from pyAudioAnalysis import audioFeatureExtraction

from features import *
from feature_helper import estimated_f0, harmonics
from utils import stft, plot_spectrum, plot_time_domain,read_all_wavedata,plot_feature
import wave
import struct

import numpy as np
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.model_selection 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#fn1
def Feature_value(x):
        Fs=44100
        F=[]
        print(x)
        F=extract_all_features(x, 44100)
        return F

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#fn2
def Serial_communication():
    
    ser = serial.Serial()
    pattern='' #consit the 1-dimensional string of onsets and instrument key to be pressed  to be communicated to bot  
    print('Communicating to Bot...')
    flag=0
    for i in Instruments:
        
        if (i=='Piano' ):
            pattern=pattern+'P'+str(Detected_Notes[flag][0])+str(Onsets[flag])+'l'
            
        if (i=='Trumpet' ):
            pattern=pattern+'T'+str(Detected_Notes[flag][0])+str(Onsets[flag])+'l'
                       
        flag+=1

    pattern=pattern+'p'
    ser.port='COM3'
    ser.open()
    ser.write(pattern)          #write to the port 
    
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#fn3   
def inst_detect():
        training_path="NewTrainingDataset_iowaAndFreesound+GOUPnigga.csv"
        testing_path="TESTFORML_sat_practise_AUDIO.csv"


        train_pt_name="TrumpetvPiano.csv"
        ##test_pt_name="test_P_T.csv"
        train_fv_name="FlutevsViolin.csv"

        from features import read_features, feature_names, extract_all_features
        from utils import TARGET_INSTRUMENTS, TARGET_CLASS, preprocess, read_all_wavedata
        import matplotlib.pyplot as plt



        def read_l1(standardize=True):
            """read featuers into X and labels for individual instruments into y. if standardize set to true, then it will return a scaler that is
            used to transform the test dataset
            """
            training_all_dataframe =pd.read_csv(training_path)
            pd.options.mode.use_inf_as_na = True
            training_all_dataframe=training_all_dataframe.fillna(10)
            
            ytrain = np.where(training_all_dataframe['Instrument']=='Flute',3,np.where(training_all_dataframe['Instrument']=='Trumpet',1,np.where(training_all_dataframe['Instrument']=='Piano',1,np.where(training_all_dataframe['Instrument']=='Violin',3,4))))
            Xtrain = training_all_dataframe[[
            'spectralCentroid_mean',
            'spectralCentroid_std',
            'spectralSpread_mean',
            'spectralSpread_std',
            'spectralFlux_mean',
            'spectralFlux_std',
            'spectralIrregularity_mean',
            'spectralIrregularity_std',
            'spectralFlatness_mean',
            'spectralFlatness_std',
            'zeroCrossingRate_mean',
            'zeroCrossingRate_std',
            'rootMeanSquare_mean',
            'rootMeanSquare_std',
            'mfcc1_mean',
            'mfcc2_mean',
            'mfcc3_mean',
            'mfcc4_mean',
            'mfcc5_mean',
            'mfcc6_mean',
            'mfcc7_mean',
            'mfcc8_mean',
            'mfcc9_mean',
            'mfcc10_mean',
            'mfcc11_mean',
            'mfcc12_mean',
            'mfcc13_mean',
            'mfcc1_std',
            'mfcc2_std',
            'mfcc3_std',
            'mfcc4_std',
            'mfcc5_std',
            'mfcc6_std',
            'mfcc7_std',
            'mfcc8_std',
            'mfcc9_std',
            'mfcc10_std',
            'mfcc11_std',
            'mfcc12_std',
            'mfcc13_std',
            'harmonicCentroid',
            'harmonicDeviation',
            'harmonicSpead',
            'logAttackTime',
            'temporalCentroid' ]].values
            Xtrain= np.asarray(Xtrain)
            ytrain=np.asarray(ytrain)

            test_all_dataframe =pd.read_csv(testing_path)
            #ytest = np.where(test_all_dataframe['Instrument']=='Flute',3,np.where(test_all_dataframe['Instrument']=='Trumpet',1,np.where(test_all_dataframe['Instrument']=='Piano',1,np.where(test_all_dataframe['Instrument']=='Violin',3,4))))
            pd.options.mode.use_inf_as_na = True
            test_all_dataframe=test_all_dataframe.fillna(0)
            Xtest = test_all_dataframe[[
            'spectralCentroid_mean',
            'spectralCentroid_std',
            'spectralSpread_mean',
            'spectralSpread_std',
            'spectralFlux_mean',
            'spectralFlux_std',
            'spectralIrregularity_mean',
            'spectralIrregularity_std',
            'spectralFlatness_mean',
            'spectralFlatness_std',
            'zeroCrossingRate_mean',
            'zeroCrossingRate_std',
            'rootMeanSquare_mean',
            'rootMeanSquare_std',
            'mfcc1_mean',
            'mfcc2_mean',
            'mfcc3_mean',
            'mfcc4_mean',
            'mfcc5_mean',
            'mfcc6_mean',
            'mfcc7_mean',
            'mfcc8_mean',
            'mfcc9_mean',
            'mfcc10_mean',
            'mfcc11_mean',
            'mfcc12_mean',
            'mfcc13_mean',
            
            'mfcc1_std',
            'mfcc2_std',
            'mfcc3_std',
            'mfcc4_std',
            'mfcc5_std',
            'mfcc6_std',
            'mfcc7_std',
            'mfcc8_std',
            'mfcc9_std',
            'mfcc10_std',
            'mfcc11_std',
            'mfcc12_std',
            'mfcc13_std',
            
            'harmonicCentroid',
            'harmonicDeviation',
            'harmonicSpead',
            'logAttackTime',
            'temporalCentroid' ]].values
            Xtest = np.asarray(Xtest)
            #ytest=np.asarray(ytest)
            
            scaler = None
            if standardize:
                scaler = preprocessing.StandardScaler()
                scaler.fit(Xtrain)
                Xtrain = scaler.transform(Xtrain)
                Xtest = scaler.transform(Xtest)
            print("read train/test ")
            return Xtrain,ytrain,Xtest,scaler



        def read_l2pt(standardize=True):
            print("reading train pt")
            """read featuers into X and labels for individual instruments into y. if standardize set to true, then it will return a scaler that is
            used to transform the test dataset
            """
            train_pt =pd.read_csv(train_pt_name)
            y = np.where(train_pt['Instrument']=='Trumpet',1,np.where(train_pt['Instrument']=='Piano',0,4))
            X = train_pt[[
            'spectralCentroid_mean',
            'spectralCentroid_std',
            'spectralSpread_mean',
            'spectralSpread_std',
            'spectralFlux_mean',
            'spectralFlux_std',
            'spectralIrregularity_mean',
            'spectralIrregularity_std',
            'spectralFlatness_mean',
            'spectralFlatness_std',
            'zeroCrossingRate_mean',
            'zeroCrossingRate_std',
            'rootMeanSquare_mean',
            'rootMeanSquare_std',
            'mfcc1_mean',
            'mfcc2_mean',
            'mfcc3_mean',
            'mfcc4_mean',
            'mfcc5_mean',
            'mfcc6_mean',
            'mfcc7_mean',
            'mfcc8_mean',
            'mfcc9_mean',
            'mfcc10_mean',
            'mfcc11_mean',
            'mfcc12_mean',
            'mfcc13_mean',
            'mfcc1_std',
            'mfcc2_std',
            'mfcc3_std',
            'mfcc4_std',
            'mfcc5_std',
            'mfcc6_std',
            'mfcc7_std',
            'mfcc8_std',
            'mfcc9_std',
            'mfcc10_std',
            'mfcc11_std',
            'mfcc12_std',
            'mfcc13_std',
            'harmonicCentroid',
            'harmonicDeviation',
            'harmonicSpead',
            'logAttackTime',
            'temporalCentroid' ]].values
            #instruments = [ins[1] for ins in y]
            instruments=y
            X = np.asarray(X)
            
            scaler = None
            if standardize:
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
            y = np.asarray(instruments)
            print("l2 pt")
            return X, y, scaler

        def read_l2fv(standardize=True):
            """read featuers into X and labels for individual instruments into y. if standardize set to true, then it will return a scaler that is
            used to transform the test dataset
            """
            train_fv =pd.read_csv(train_fv_name)
            y = np.where(train_fv['Instrument']=='Flute',0,np.where(train_fv['Instrument']=='Violin',1,4))
            X = train_fv[[
            'spectralCentroid_mean',
            'spectralCentroid_std',
            'spectralSpread_mean',
            'spectralSpread_std',
            'spectralFlux_mean',
            'spectralFlux_std',
            'spectralIrregularity_mean',
            'spectralIrregularity_std',
            'spectralFlatness_mean',
            'spectralFlatness_std',
            'zeroCrossingRate_mean',
            'zeroCrossingRate_std',
            'rootMeanSquare_mean',
            'rootMeanSquare_std',
            'mfcc1_mean',
            'mfcc2_mean',
            'mfcc3_mean',
            'mfcc4_mean',
            'mfcc5_mean',
            'mfcc6_mean',
            'mfcc7_mean',
            'mfcc8_mean',
            'mfcc9_mean',
            'mfcc10_mean',
            'mfcc11_mean',
            'mfcc12_mean',
            'mfcc13_mean',
            'mfcc1_std',
            'mfcc2_std',
            'mfcc3_std',
            'mfcc4_std',
            'mfcc5_std',
            'mfcc6_std',
            'mfcc7_std',
            'mfcc8_std',
            'mfcc9_std',
            'mfcc10_std',
            'mfcc11_std',
            'mfcc12_std',
            'mfcc13_std',
            'harmonicCentroid',
            'harmonicDeviation',
            'harmonicSpead',
            'logAttackTime',
            'temporalCentroid' ]].values
            #instruments = [ins[1] for ins in y]
            instruments=y
            X = np.asarray(X)
            
            scaler = None
            if standardize:
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
            y = np.asarray(instruments)
            print ("l2fv")
            return X, y, scaler



        X,y,X_test,scaler = read_l1(standardize=True)
        print scaler
        X_pt, y_pt, scaler1 = read_l2pt(standardize=True)
        print scaler1
        X_fv, y_fv, scaler2 = read_l2fv(standardize=True)
        print scaler2
        print("\n")
        res=list()
        #test_all_dataframe =pd.read_csv(testing_path)
        #test_labels = np.where(test_all_dataframe['Instrument']=='Flute',2,np.where(test_all_dataframe['Instrument']=='Trumpet',1,np.where(test_all_dataframe['Instrument']=='Piano',0,np.where(test_all_dataframe['Instrument']=='Violin',3,4))))

######train test begins

        e4=DecisionTreeClassifier(max_depth=2)
#11111111111111111111111111111111
        mlp=SVC(C=0.0005, gamma=0.388,kernel='poly')
        #mlp=SVC(C=0.0005, gamma=0.388,kernel='poly')
        #2mlp=RandomForestClassifier(max_depth=90, n_estimators=3000)    
        mlp.fit(X,y)
    
#111111111111111111111111111111111111111111111111111111111111111
        knn_pt = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', p=1) # manhattan_distance
        knn_pt.fit(X_pt,y_pt)

        rf_pt = RandomForestClassifier(100,criterion="gini", n_jobs=-1)
        rf_pt.fit(X_pt,y_pt)

        svm_pt = SVC(C=5.7001954868790526315788, gamma=0.0090184210526315788,kernel='poly',class_weight='balanced',decision_function_shape='ovr')
        svm_pt.fit(X_pt,y_pt)

        mlp_pt = MLPClassifier(activation='relu',alpha=0.4,solver='adam',hidden_layer_sizes=(60,20,80,8),max_iter=1000,random_state=1)
        mlp_pt.fit(X_pt,y_pt)    

        abc_pt=AdaBoostClassifier(algorithm='SAMME',base_estimator=DecisionTreeClassifier(max_depth=50), n_estimators=100, learning_rate=0.05,random_state=None)
        abc_pt.fit(X_pt,y_pt)

        #AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=0.09,random_state=None)
        
        abc_pt2=AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=0.09,random_state=None)
        abc_pt2.fit(X_pt,y_pt)
#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
        knn_fv = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', p=1) # manhattan_distance
        knn_fv.fit(X_fv,y_fv)
    
        rf_fv = RandomForestClassifier(100,criterion="gini", n_jobs=-1)
        rf_fv.fit(X_fv,y_fv)

        mlp_fv = MLPClassifier(activation='relu',alpha=0.1,solver='adam',hidden_layer_sizes=(120,80,20,8),max_iter=1000,random_state=1)
        mlp_fv.fit(X_fv,y_fv)

        svm_fv = SVC(C=5.7001954868790526315788, gamma=0.0090184210526315788,kernel='poly',class_weight='balanced',decision_function_shape='ovr')
        svm_fv.fit(X_fv,y_fv)

        abc_fv=AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=0.09,random_state=None)
        abc_fv.fit(X_fv,y_fv)

        
        for i in range(len(X_test)):
                k=mlp.predict(X_test[i])
                #---------------------
                if(k==1):
                    x=abc_pt2.predict(X_test[i])
                    k=x
                    if (k==0):
                        res.append('Piano')
                    if (k==1):
                        res.append('Trumpet')
                if(k==3):
                    y=abc_fv.predict(X_test[i])
                    k=y
                    if (k==0):
                        res.append('Flute')
                    if(k==1):
                        res.append('Violin')
        return res


    #print(confusion_matrix(test_labels,res))
    #print(classification_report(test_labels,res))
       



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#fn4
def freqFind(sound):
#---------------------------------------Obtaining the frequency spectrum of the audio file----------------------------------------------#
    mag=np.fft.fft(sound)
    mag1=mag[:(len(sound)/2)]
    mag1=np.abs(mag1)
    mag1=20*np.log10(1+mag1) 
    X=mag1.reshape(-1,1)
    minmax_scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
    data_minmax=minmax_scalar.fit_transform(X)
    
    #-----------------------------------Calculating the fundemental frequency------------------------------------------------#
    
    wsize=int((len(sound)/2)*0.01) # window size = (file_length/2) x 0.01. We are dividing the spectrum into 100 parts and checking the argmax of each part
    peak_frequency=np.zeros(100)
    peak=np.zeros(100)
    index=0
    for i in range(0,100):
        seg=data_minmax[i*wsize:(i+1)*wsize]
        peak_frequency[index]=(np.argmax(seg)+i*wsize)*44100/len(sound)
        peak[index]=np.max(seg)
        index=index+1
        
    max_freq=peak_frequency[np.argmax(peak)]
    flag=0
    fundemental_frequency=0
   # plt.stem(peak_frequency,peak)
   # plt.show()
    for i in range(0,100):
        if(peak_frequency[i] != 0):
                div=max_freq/float(peak_frequency[i])
                div=round(div,3)
                divround=round(div)
                #print(str(abs(divround-div))+" "+str(peak_frequency[i]))
                err=abs(divround-div)
        if(peak_frequency[i]>=32 and peak[i]>=0.4 and flag==0 and err<=0.025 ):
            fundemental_frequency=peak_frequency[i]
            flag=1
            break
   # print(fundemental_frequency)          
                                     
    #----------------------------------Obtaining the note based on the fundemental frequency----------------------------------------------------#
    note=["C","C#","D","D#","E","F", "F#","G","G#","A","A#","B"]
    noteF=[16.35, 17.32, 18.35, 19.45, 20.60,  21.83, 23.13, 24.50, 25.96, 27.50, 29.14, 30.87]
    error_percent=3 
    note_detected=""
    flag=0
    for i in range(0,9,1):
        for j in range(0,12,1):
            note_frequency=(2**i)*noteF[j]# sf -> standard frequency from noteF array
            if(abs(100*(fundemental_frequency-note_frequency)/note_frequency)<= error_percent and flag==0):
                #print("Note detected is "+str(note[j])+str(i))
                note_detected=note[j]+str(i)
                flag=1
                break
    return note_detected


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#fn5
def onset_notes(file_name):

        #=======================================================================DEFINING FINAL DF TO WRITE TO CSV=====================================================================================
        columns1=['spectralCentroid_mean','spectralCentroid_std','spectralSpread_mean','spectralSpread_std','spectralFlux_mean','spectralFlux_std','spectralIrregularity_mean','spectralIrregularity_std','spectralFlatness_mean','spectralFlatness_std','zeroCrossingRate_mean','zeroCrossingRate_std','rootMeanSquare_mean','rootMeanSquare_std','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean','mfcc5_mean','mfcc6_mean','mfcc7_mean','mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','mfcc1_std','mfcc2_std','mfcc3_std','mfcc4_std','mfcc5_std','mfcc6_std','mfcc7_std','mfcc8_std','mfcc9_std','mfcc10_std','mfcc11_std','mfcc12_std','mfcc13_std','harmonicCentroid','harmonicDeviation','harmonicSpead','logAttackTime','temporalCentroid']
        final_df = DataFrame(columns= ['spectralCentroid_mean','spectralCentroid_std','spectralSpread_mean','spectralSpread_std','spectralFlux_mean','spectralFlux_std','spectralIrregularity_mean','spectralIrregularity_std','spectralFlatness_mean','spectralFlatness_std','zeroCrossingRate_mean','zeroCrossingRate_std','rootMeanSquare_mean','rootMeanSquare_std','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean','mfcc5_mean','mfcc6_mean','mfcc7_mean','mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','mfcc1_std','mfcc2_std','mfcc3_std','mfcc4_std','mfcc5_std','mfcc6_std','mfcc7_std','mfcc8_std','mfcc9_std','mfcc10_std','mfcc11_std','mfcc12_std','mfcc13_std','harmonicCentroid','harmonicDeviation','harmonicSpead','logAttackTime','temporalCentroid'])

        #=======================================================================AUDIO READ=========================================================================================
        #yyy--------------------------------------------------------------------------------------------------------------------------------------------------
        filename1=file_name
        sound_file = wave.open(filename1, 'rb')
                             
        #yyy--------------------------------------------------------------------------------------------------------------------------------------------------
        #=========================================================================ONSETS+CSVWRITE===========================================================================================

        #////////////////////--------------------------//////////////
        Onsets = []
        offsets=[]
        Detected_Notes=[]
        #////////////////----------------------//////////////////

        
        file_length = sound_file.getnframes()
        sound = np.zeros(file_length)

        for i in range(file_length):
                data = sound_file.readframes(1)
                data = struct.unpack("<h", data)
                sound[i] = int(data[0])
        sound = np.divide(sound, float(2**15))

        arr2=np.concatenate((np.zeros(1,dtype=int),np.ones(99,dtype=int)),axis=0)
        soundk=abs(sound)
        sound1=np.square(soundk)
        wtime=float(1000/44100.0)
        newa1=np.convolve(sound1,arr2,mode='full')
        newa3=10*np.log10(newa1+0.001)
        leng=len(newa3)
        windows=int(200)
        leng=int(len(newa3))
        yaya=np.zeros(int(leng/windows))
        for i in range(len(yaya)):
                okay=[(newa3[(i*200):((i+1)*200)])]
                yaya[i]=np.min(okay)

        yayak=np.zeros(int(len(yaya)/5))
        windowtime=[]
        
        for i in range(len(yayak)):
               
                  okay=[(yaya[(i*5):((i+1)*5)])]
                  yayak[i]=np.min(okay)
                  windowtime.append(i*wtime)
        

        #################### detecting onsets and silence parts #####################################################################################################################################3

        y=0
        c=0                                                                 #index to store onset
        #onset=np.zeros(int((file_length/1000)))                                       #onset array
        onset=list()
        onsetsize=0 
        yayak[0]=-30
        yayak[len(yayak)-1]=-30

        plt.plot(yayak)
        plt.show()

        am_max=-1*np.max(yayak)
        am_min=-1*np.min(yayak)
        err=am_min-am_max
        err1=err*0.4
        am_max1=am_max+err1
        midleveldb=-1*am_max1
        #////////////////------------------------//////////
        Onset=list()
        detect=list()
        
        #//////////////////---------------------///////
        
        flag1=0
        so=0
        yayak[len(yayak)-1]=-30               #first and last sample of yayak is made -30 db
        yayak[0]=-30
        done=0
        startz=0 
        endz=0      
        for i in range(0,len(yayak)-1,1):
                if(yayak[i]<=-27.7 and yayak[i+1]>-27.7):
                        startz=i
                        flag1=0
                if(yayak[i]>=-15):
                        flag1=1
                if(yayak[i]>=-27.7 and yayak[i+1]<-27.7):
                        endz=i
                        if(flag1==1):
                                onsetsize=onsetsize+1
                                onset.append(startz)
                                onset.append(endz)
                                c=c+1
                                flag1=0
                                print c

        print(onset)
                                

        spectralCentroid_mean=np.zeros(onsetsize)
        spectralCentroid_std=np.zeros(onsetsize)
        spectralSpread_mean=np.zeros(onsetsize)
        spectralSpread_std=np.zeros(onsetsize)
        spectralFlux_mean=np.zeros(onsetsize)
        spectralFlux_std=np.zeros(onsetsize)
        spectralIrregularity_mean=np.zeros(onsetsize)
        spectralIrregularity_std=np.zeros(onsetsize)
        spectralFlatness_mean=np.zeros(onsetsize)
        spectralFlatness_std=np.zeros(onsetsize)
        zeroCrossingRate_mean=np.zeros(onsetsize)
        zeroCrossingRate_std=np.zeros(onsetsize)
        rootMeanSquare_mean=np.zeros(onsetsize)
        rootMeanSquare_std=np.zeros(onsetsize)
        mfcc1_mean=np.zeros(onsetsize)
        mfcc2_mean=np.zeros(onsetsize)
        mfcc3_mean=np.zeros(onsetsize)
        mfcc4_mean=np.zeros(onsetsize)
        mfcc5_mean=np.zeros(onsetsize)
        mfcc6_mean=np.zeros(onsetsize)
        mfcc7_mean=np.zeros(onsetsize)
        mfcc8_mean=np.zeros(onsetsize)
        mfcc9_mean=np.zeros(onsetsize)
        mfcc10_mean=np.zeros(onsetsize)
        mfcc11_mean=np.zeros(onsetsize)
        mfcc12_mean=np.zeros(onsetsize)
        mfcc13_mean=np.zeros(onsetsize)
        mfcc1_std=np.zeros(onsetsize)
        mfcc2_std=np.zeros(onsetsize)
        mfcc3_std=np.zeros(onsetsize)
        mfcc4_std=np.zeros(onsetsize)
        mfcc5_std=np.zeros(onsetsize)
        mfcc6_std=np.zeros(onsetsize)
        mfcc7_std=np.zeros(onsetsize)
        mfcc8_std=np.zeros(onsetsize)
        mfcc9_std=np.zeros(onsetsize)
        mfcc10_std=np.zeros(onsetsize)
        mfcc11_std=np.zeros(onsetsize)
        mfcc12_std=np.zeros(onsetsize)
        mfcc13_std=np.zeros(onsetsize)
        harmonicCentroid=np.zeros(onsetsize)
        harmonicDeviation=np.zeros(onsetsize)
        harmonicSpead=np.zeros(onsetsize)
        logAttackTime=np.zeros(onsetsize)
        temporalCentroid=np.zeros(onsetsize)

        a=0 # stores the corresponding window it time frame 
        y=[] # list of onsets 
        z=[] # list of offset 

        k=0
        
        for i in range(onsetsize):
                a=round((onset[2*i]*wtime),2)
                y.append(a)
                a=round((onset[2*i+1]*wtime),2)
                z.append(a)
                
                features=Feature_value(sound[int(onset[2*i]*1000):int(onset[2*i+1]*1000)])    #<=========================================================================features
                note=(freqFind(sound[int(onset[2*i]*1000):int(onset[2*i+1]*1000)]))        #<=========================================================================detect+freqfind
                if(note==''):
                     note=freqFind(sound[int(onset[2*i]*1000):int((onset[2*i+1]*1000)+(onset[2*i]*1000))/2])
                     if(note==''):
                         note='A4'
                detect.append(note)
                print(detect)
                spectralCentroid_mean[i]=features[0]
                spectralCentroid_std[i]=features[1]
                spectralSpread_mean[i]=features[2]
                spectralSpread_std[i]=features[3]
                spectralFlux_mean[i]=features[4]
                spectralFlux_std[i]=features[5]
                spectralIrregularity_mean[i]=features[6]
                spectralIrregularity_std[i]=features[7]
                spectralFlatness_mean[i]=features[8]
                spectralFlatness_std[i]=features[9]
                zeroCrossingRate_mean[i]=features[10]
                zeroCrossingRate_std[i]=features[11]
                rootMeanSquare_mean[i]=features[12]
                rootMeanSquare_std[i]=features[13]
                mfcc1_mean[i]=features[14]
                mfcc2_mean[i]=features[15]
                mfcc3_mean[i]=features[16]
                mfcc4_mean[i]=features[17]
                mfcc5_mean[i]=features[18]
                mfcc6_mean[i]=features[19]
                mfcc7_mean[i]=features[20]
                mfcc8_mean[i]=features[21]
                mfcc9_mean[i]=features[22]
                mfcc10_mean[i]=features[23]
                mfcc11_mean[i]=features[24]
                mfcc12_mean[i]=features[25]
                mfcc13_mean[i]=features[26]
                mfcc1_std[i]=features[27]
                mfcc2_std[i]=features[28]
                mfcc3_std[i]=features[29]
                mfcc4_std[i]=features[30]
                mfcc5_std[i]=features[31]
                mfcc6_std[i]=features[32]
                mfcc7_std[i]=features[33]
                mfcc8_std[i]=features[34]
                mfcc9_std[i]=features[35]
                mfcc10_std[i]=features[36]
                mfcc11_std[i]=features[37]
                mfcc12_std[i]=features[38]
                mfcc13_std[i]=features[39]
                harmonicCentroid[i]=features[40]
                harmonicDeviation[i]=features[41]
                harmonicSpead[i]=features[42]
                logAttackTime[i]=features[43]
                temporalCentroid[i]=features[44]

                k=k+1

         #//////////////////---------------------///////
        Detected_Notes=detect[:k]
        Onsets=y[:k]
        offsets=z[:k]

        #print(Detected_Notes)
        #print(Onsets)
        #print(offsets)
        #print(Detected_Notes)

        #=======================================================================================================================================================================

        datasamp = {     'spectralCentroid_mean':spectralCentroid_mean,
                         'spectralCentroid_std':spectralCentroid_std,
                         'spectralSpread_mean':spectralSpread_mean,
                         'spectralSpread_std':spectralSpread_std,
                         'spectralFlux_mean':spectralFlux_mean,
                         'spectralFlux_std':spectralFlux_std,
                         'spectralIrregularity_mean':spectralIrregularity_mean,
                         'spectralIrregularity_std':spectralIrregularity_std,
                         'spectralFlatness_mean':spectralFlatness_mean,
                         'spectralFlatness_std':spectralFlatness_std,
                         'zeroCrossingRate_mean':zeroCrossingRate_mean,
                         'zeroCrossingRate_std':zeroCrossingRate_std,
                         'rootMeanSquare_mean':rootMeanSquare_mean,
                         'rootMeanSquare_std':rootMeanSquare_std,
                         'mfcc1_mean':mfcc1_mean,
                         'mfcc2_mean':mfcc2_mean,
                         'mfcc3_mean':mfcc3_mean,
                         'mfcc4_mean':mfcc4_mean,
                         'mfcc5_mean':mfcc5_mean,
                         'mfcc6_mean':mfcc6_mean,
                         'mfcc7_mean':mfcc7_mean,
                         'mfcc8_mean':mfcc8_mean,
                         'mfcc9_mean':mfcc9_mean,
                         'mfcc10_mean':mfcc10_mean,
                         'mfcc11_mean':mfcc11_mean,
                         'mfcc12_mean':mfcc12_mean,
                         'mfcc13_mean':mfcc13_mean,
                         'mfcc1_std':mfcc1_std,
                         'mfcc2_std':mfcc2_std,
                         'mfcc3_std':mfcc3_std,
                         'mfcc4_std':mfcc4_std,
                         'mfcc5_std':mfcc5_std,
                         'mfcc6_std':mfcc6_std,
                         'mfcc7_std':mfcc7_std,
                         'mfcc8_std':mfcc8_std,
                         'mfcc9_std':mfcc9_std,
                         'mfcc10_std':mfcc10_std,
                         'mfcc11_std':mfcc11_std,
                         'mfcc12_std':mfcc12_std,
                         'mfcc13_std':mfcc13_std,
                         'harmonicCentroid':harmonicCentroid,
                         'harmonicDeviation':harmonicDeviation,
                         'harmonicSpead':harmonicSpead,
                         'logAttackTime':logAttackTime,
                         'temporalCentroid':temporalCentroid
                          
              }

        er=DataFrame(datasamp,columns=columns1)
        final_df=pd.concat([final_df,er])
        data_csv_name='TESTFORML_sat_practise_AUDIO.csv'
        
        pd.options.mode.use_inf_as_na = True
        final_df=final_df.fillna(0)
        final_df.to_csv ((path+'\\'+data_csv_name+'.csv'), index = None, header=True) #Don't forget to add '.csv' at the end of the path      

        #HOLY GRAIL
        return (Detected_Notes,Onsets)
        




############################### Your Code Here ##########################################################################################################################################

def Instrument_identify(Audio_file):
	

	
	Instruments = []
	Detected_Notes = []
	Onsets = []
        file_name = "Audio.wav"

	#call to detect notes and onset via : onset_detect()
	Detected_Notes,Onsets = onset_notes(file_name)


	#call to train  svm and detect instrument and appending it 
	Instruments=inst_detect()


	return Instruments, Detected_Notes, Onsets



############################### Main Function ############################################################################################################################################

if __name__ == "__main__":

	#   Instructions
	#   ------------
	#   Do not edit this function.

	# code for checking output for single audio file
	path = os.getcwd()

	#audio_test_all Practice_audio_file
	file_name = path +"\\Practice_audio_file.wav"
	#audio_file = wave.open(file_name)
	
	Instruments, Detected_Notes, Onsets = Instrument_identify()

	print("\n\tInstruments = "  + str(Instruments))
	print("\n\tDetected Notes = " + str(Detected_Notes))
	print("\n\tOnsets = " + str(Onsets))

	
	Serial_communication()
	
	
	

