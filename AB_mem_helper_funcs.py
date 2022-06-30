# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:13:33 2022

@author: Adam Dede
"""

import numpy as np
import pandas as pd
import math


path = r"C:\Users\Adam Dede\Documents\GitHub\ABmem\data\PPT09_ABstudy_2022-06-29_10h01.10.225.csv"


def findT1(trialDat):
    pics = trialDat.loc[trialDat.index.str.contains('Img')]
    return(pics.iloc[int(trialDat['T1Loc'])])

def refMem2AB(trialDat, ABDat): 
    trialLink = pd.Series(np.zeros(3), index = ['ABT1cor', 'ABT2cor', 'lag'])
    try: 
        ii = np.where((ABDat['T1imageID'].values == trialDat['Picture']) & (ABDat['T1_PA'].values == 'P') )[0][0]
        trialLink['ABT1cor'] = ABDat.iloc[ii,:]['T1cor']
        trialLink['ABT2cor'] = ABDat.iloc[ii,:]['T2cor']
        trialLink['lag'] = ABDat.iloc[ii,:]['Lag']
        return(trialLink)
    except:
        trialLink['ABT1cor'] = np.nan
        trialLink['ABT2cor'] = np.nan
        trialLink['lag'] = np.nan
        return(trialLink)

        
def readDataFile(path): 
    df = pd.read_csv(path)
    
    #get the AB data
    ABDat = df[df['TrialType']=='AB']
    
    #need to make a column that indicates what the T1 image was
    ABDat['T1imageID'] = ABDat.apply(findT1, axis = 1)
    
    #set T1cor to 0 (wrong) for all 
    ABDat['T1cor'] = 0
    #set to 1 for trials on which T1 was present and the participant pressed 'left'
    ABDat.loc[(ABDat['key_resp_2.keys'] == 'left') & (ABDat['T1_PA'] == 'P'), 'T1cor'] = 1
    #set to 1 for trials on which T1 was absent and the participant pressed 'right'
    ABDat.loc[(ABDat['key_resp_2.keys'] == 'right') & (ABDat['T1_PA'] == 'A'), 'T1cor'] = 1
    
    
    #set T2cor to 0 (wrong) for all 
    ABDat['T2cor'] = 0
    #set to 1 for trials on which T1 was present and the participant pressed 'left'
    ABDat.loc[(ABDat['key_resp_3.keys'] == 'left') & (ABDat['T2_PA'] == 'P'), 'T2cor'] = 1
    #set to 1 for trials on which T1 was absent and the participant pressed 'right'
    ABDat.loc[(ABDat['key_resp_3.keys'] == 'right') & (ABDat['T2_PA'] == 'A'), 'T2cor'] = 1
    
    
    #get the % correct for T1, T2, and T2|T1 for all trial types
    out = pd.Series()
    for cnd in ABDat['CondType'].unique():
        out[cnd +'_T1'] = np.mean(ABDat.loc[ABDat['CondType'] == cnd, 'T1cor'])
        out[cnd +'_T2'] = np.mean(ABDat.loc[ABDat['CondType'] == cnd, 'T2cor'])
        out[cnd +'_T2_T1'] = np.mean(ABDat.loc[(ABDat['CondType'] == cnd) & (ABDat['T1cor'] == 1), 'T2cor'])
    
    
    
    #get the memory data
    memDat = df[df['TrialType'] == 'Mem']
 
    f = lambda x: refMem2AB(x, ABDat)
    x = memDat.apply(f, axis = 1)
    memDat = pd.concat([memDat,x], axis = 1)
    
    #key trial types: 
    memDat['ABrefType'] = 'skip'
    #previous lag1 P P both correct 'PP1cor'
    memDat.loc[(memDat['T1PA']=='P') & (memDat['ABT1cor']==1) & (memDat['ABT2cor']==1) & (memDat['Lag']==1), 'ABrefType'] = 'PP1cor'
    #previous lag1 P P both miss 'PP1miss'
    memDat.loc[(memDat['T1PA']=='P') & (memDat['ABT1cor']==0) & (memDat['ABT2cor']==0) & (memDat['Lag']==1), 'ABrefType'] = 'PP1miss'
    #previous lag1 P P blink 'PP1blink'
    memDat.loc[(memDat['T1PA']=='P') & (memDat['ABT1cor']==1) & (memDat['ABT2cor']==0) & (memDat['Lag']==1), 'ABrefType'] = 'PP1blink'
    #previous lag1 P P T2 correct 'PP1T2'
    memDat.loc[(memDat['T1PA']=='P') & (memDat['ABT1cor']==0) & (memDat['ABT2cor']==1) & (memDat['Lag']==1), 'ABrefType'] = 'PP1T2'
    #previous lag5 P P both correct 'PP5cor'
    memDat.loc[(memDat['T1PA']=='P') & (memDat['ABT1cor']==1) & (memDat['ABT2cor']==1) & (memDat['Lag']==5), 'ABrefType'] = 'PP5cor'
    #distracter
    memDat.loc[memDat['ImageType']=='Distractor', 'ABrefType'] = 'distracter'
    #novel
    memDat.loc[memDat['ImageType']=='Novel', 'ABrefType'] = 'novel'
    
    #assessing memory
    memDat['oldNew'] = 0
    memDat.loc[memDat['MemResponse.keys']>3, 'oldNew'] = 1
    memDat['memCor'] = 0
    memDat.loc[(memDat['ABrefType'] != 'novel') & (memDat['oldNew'] == 1), 'memCor'] = 1
    memDat.loc[(memDat['ABrefType'] == 'novel') & (memDat['oldNew'] == 0), 'memCor'] = 1
    
    
    out['targMem'] = np.mean(memDat.loc[(memDat['ABrefType'] != 'distracter') & (memDat['ABrefType'] != 'skip'), 'memCor'])
    out['distMem'] = np.mean(memDat.loc[(memDat['ABrefType'] == 'distracter') | (memDat['ABrefType'] == 'novel'), 'memCor'])
    
    for cnd in memDat['ABrefType'].unique():
        if (cnd != 'skip'): 
            out[cnd + '_mem'] = np.mean(memDat.loc[(memDat['ABrefType'] == cnd), 'memCor'])
    
    
    
    out.name = df['date'][0]
    
    return(out)
    
    
    
    
    
    
    
    
    