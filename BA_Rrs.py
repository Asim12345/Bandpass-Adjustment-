# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:07:58 2022

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:27:18 2022

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:56:41 2021

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 10:19:20 2021

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:46:59 2021

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:10:27 2021

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:23:45 2021

@author: mas108
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:51:01 2020

@author: mas108
"""

# -*- coding: utf-8 -*-
## This is the main code for a Journal Paper


# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.lines as mlines
import re
from sklearn.metrics import mean_squared_error
from scipy.stats import variation
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import stats

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
tf.reset_default_graph()
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model



def normalize(X):
    col_max = np.max(X, axis=0)
    col_min = np.min(X, axis=0)
    normX = np.divide(X - col_min, col_max - col_min)
    return normX

def denormalize(X1,X2):
    col_max = np.max(X2, axis=0)
    col_min = np.min(X2, axis=0)
    denormX = X1*(col_max - col_min )+ col_min
    return denormX

def some_func(roi):
    '''
    simple function to return the mean of the region
    of interest
    '''
    #roi= np.concatenate((roi[0], roi[1], roi[3]), axis=1)
    if variation(np.concatenate((roi[0], roi[1], roi[2]), axis=0)) < 0.25:
       return np.median(roi)
    else: 
      return 0.0
  
def some_func_2(roi):
    '''
    simple function to return the mean of the region
    of interest
    '''
    #roi= np.concatenate((roi[0], roi[1], roi[3]), axis=1)
    if variation(np.concatenate((roi[0], roi[1], roi[2]), axis=0)) < 0.25:
       return np.median(roi)
    else: 
      return 0.0

    
def some_func_1(roi):
    '''
    simple function to return the mean of the region
    of interest
    '''
    return np.mean(roi)

Error_MSE=[]
GP_TR=[]
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    #print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
Linear_P=[]
MSI_Data=[]
P_Data=[]    
R2_Test_All=[]
Error_RMSE=[]
R2_Score=[]
R_Score=[]
R2_avgerage=[]
R=[]
R_B=[]
MSE_B= []
MSE=[]
RMSE_B=[]
RMSE=[]
R2_B=[]
R2=[]
OLI_Rrs=[]
MSI_Rrs=[]
Scene_L8= []
Scene_S2= []


excel_file= 'C:/Users/mas108/Desktop/Asim/aa/Results_L8S2_version5.xlsx'
excel_file= 'C:/Users/mas108/Desktop/Asim/Results/Results_L8S2_version5_a.xlsx'
excel_file= 'C:/Users/mas108/Desktop/Asim/Results/ExcelFolder/Results_L8S2_2020.xlsx'
excel_file= 'C:/Users/mas108/Desktop/Asim/S2_level3_data/Results_L8S2_2018_2020_L3.xlsx'
excel_file= 'C:/Users/mas108/Desktop/Asim/Level_1Data/Results_L8S2_2018_2020_L3_org.xlsx'
excel_file= 'C:/Users/mas108/Desktop/Asim/Level_1Data/Data_2018_2020/New folder/Results_L8S2_2018_2020_Acolite_All_check.xlsx' # both sensors data

excel_file= 'D:/US_Lakes/OC_SMART_OLI.xlsx'
file1 = pd.read_excel(excel_file,sheet_name='Sheet1')  

Rrs_All_Bands=[]
Band_B1_OL1=[]
Band_B2_OL1=[]
Band_G_OL1=[]
Band_R_OL1=[]
Band_IR_OL1=[]

Band_B1_MSI=[]
Band_B2_MSI=[]
Band_G_MSI=[]
Band_R_MSI=[]
Band_IR_MSI=[]
Time_Gap=[]
W_median=[]
B_median=[]
Weights=[]
Bias=[]

OLI_Data=[]
MSI_Data=[]
Pre_Rrs_All_Bands=[]
Ref_Rrs_All_Bands=[]
R_Rrs_All_Bands=[]
    
Pre_Data=np.array([])
MSI_Data=np.array([])

Pre_Data1=np.array([])
MSI_Data1=np.array([])

    
MSI_Band=    ['Rrs_443_Acolite_S2','Rrs_492_Acolite_S2','Rrs_560_Acolite_S2','Rrs_665_Acolite_S2','Rrs_865_Acolite_S2' ] #, 'Rrs_chl_oc2_Acolite_S2']    
OLI_Band= ['Rrs_443_Acolite', 'Rrs_483_Acolite', 'Rrs_561_Acolite','Rrs_655_Acolite','Rrs_865_Acolite']#, 'Rrs_chl_oc2_Acolite']     
#OLI_Band1= ['rhow_1', 'rhow_2', 'rhow_3','rhow_4','rhow_B8A']  
#OLI_Band1= ['440 nm']      


#MSI_Band=    ['Rrs_665_Acolite_S2' ]
#OLI_Band= ['Rrs_655_Acolite']
#MSI_Band=    ['Rrs_443_Acolite_S2','Rrs_492_Acolite_S2','Rrs_560_Acolite_S2', 'Rrs_665_Acolite_S2']
#OLI_Band= ['Rrs_443_Acolite', 'Rrs_483_Acolite', 'Rrs_561_Acolite','Rrs_655_Acolite']

MSI_Band=    ['Rrs_443_S2_OC', 'Rrs_492_S2_OC', 'Rrs_560_S2_OC','Rrs_665_S2_OC' ] 
OLI_Band= ['Rrs_443_L8_OC', 'Rrs_483_L8_OC', 'Rrs_561_L8_OC' , 'Rrs_655_L8_OC'  ] 
#MSI_Band=    ['Rrs_492_S2_OC', 'Rrs_560_S2_OC','Rrs_655_S2_OC' ] 
#OLI_Band= ['Rrs_483_L8_OC', 'Rrs_561_L8_OC' , 'Rrs_665_L8_OC'  ] 
##MSI_Band=    ['Rrs_560_S2_OC','Rrs_655_S2_OC' ] 
##OLI_Band= ['Rrs_561_L8_OC' , 'Rrs_665_L8_OC'  ] 
#
MSI_Band=    ['Rrs_492_S2_OC'] 
OLI_Band= ['Rrs_483_L8_OC'] 
#
OLI_Band=    ['Rrs_655_L8_OC'] 
MSI_Band= ['Rrs_665_S2_OC'] 
#
#
#
OLI_Band=    ['Rrs_561_L8_OC'] 
MSI_Band= ['Rrs_560_L8_OC'] 


MSI_Band=    ['Rrs_443_S2_OC' ] 
OLI_Band= ['Rrs_443_L8_OC' ] 


##MSI_Band=    ['Rrs_665_S2_OC' ] 
##OLI_Band= ['Rrs_655_L8_OC'  ] 
#
#MSI_Band=    ['Rrs_492_S2_OC', 'Rrs_560_S2_OC','Rrs_665_S2_OC' ] 
#OLI_Band= ['Rrs_483_L8_OC', 'Rrs_561_L8_OC' , 'Rrs_655_L8_OC'  ]


for (f, b) in zip(MSI_Band, OLI_Band): 
    tf.reset_default_graph()    
    print(f, b)
        
        
    MSI_Band1=  [f]
    OLI_Band1= [b] 
    
    Pre_Data =[]
    GP=[]
    MSI_Data =[]
    OLI_Data = []
    Linear_P=[]
                      # "rhow_3"    # "865 nm"  # '560 nm'  #  '480 nm'   #'440 nm' 
    for i1 in range (len(MSI_Band1)): 
                MSI_Band=MSI_Band1[i1]
                OLI_Band= OLI_Band1[i1]
                
                Band_B1_OL1=[]
                Band_B2_OL1=[]
                Band_G_OL1=[]
                Band_R_OL1=[]
                Band_IR_OL1=[]
                
                Band_B1_MSI=[]
                Band_B2_MSI=[]
                Band_G_MSI=[]
                Band_R_MSI=[]
                Band_IR_MSI=[]
                
                Band_B1_OL1_1=[]
                Band_B1_MSI_1=[]
                Band_B1_OL1_2=[]
                Band_B1_MSI_2=[]
                Band_B1_OL1_3=[]
                Band_B1_MSI_3=[]
                Band_B1_OL1_4=[]
                Band_B1_MSI_4=[]
                
                AOD_OLI_L8=[]
                AOD_MSI_S2=[]
                
                CH_OLI_L8=[]
                CH_MSI_S2=[]
                
                print()
      
                c=[]            
                Z=5.0
                for z in range(0,24 ):  # 15 problem
       
#                   if z == 15:
#                        continue
                 
                
                   #print(i, Time_G, (file1['Timedelta_L8'][i][11:19]), time) 
                   if file1.loc[z, [MSI_Band]].isnull().values.any() == False and (file1.loc[z,[MSI_Band]] == '0.').any() ==False and file1.loc[z,[OLI_Band]].isnull().values.any() == False : 
    

                       
                       CHL_MSI= np.array(file1['L2_flags_S2'][z] )
                       if  str(CHL_MSI)== 'nan':
                            continue
                       A=CHL_MSI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('        ]',']').replace('  ]',']').replace('   ]',']').replace(' ]',']').replace('[  ','[').replace('[ ','[').replace('[','').replace(']','')))
                       #A2 = np.array(re.split("\s+", A.replace(',', ''). replace('[','').replace('     ]','').replace(']',''))) 
                       A4=[]
                       CHL_B_MSI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            CHL_B_MSI.append(A3)
                            
                       CHL_B_MSI= np.array(CHL_B_MSI)
                    
                      
                       ##################################################################################################
                       
                                           
                       CHL_OLI= np.array(file1['L2_flags_L8'][z]) 
        
                       A=CHL_OLI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('     ]',']').replace('  ]',']').replace('[  ','['). replace('[  ','[').replace('[    ','[').replace('[ ','[').replace(' ]',']').replace('[','').replace(']','')))
                       A4=[]
                       CHL_B_OLI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            CHL_B_OLI.append(A3)
                            
                       
                       CHL_B_OLI= np.array(CHL_B_OLI)
                       
                       ##################################################
                       
                       CHL_OLI= np.array(file1['AOD_L8'][z]) 
        
                       A=CHL_OLI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('     ]',']').replace('  ]',']').replace('[  ','['). replace('[  ','[').replace('[    ','[').replace('[ ','[').replace(' ]',']').replace('[','').replace(']','')))
                       A4=[]
                       AOD_OLI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            AOD_OLI.append(A3)
                            
                       
                       AOD_OLI= np.array(AOD_OLI)
                       
                       CHL_MSI= np.array(file1['AOD'][z]) 
        
                       A=CHL_MSI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('     ]',']').replace('  ]',']').replace('[  ','['). replace('[  ','[').replace('[    ','[').replace('[ ','[').replace(' ]',']').replace('[','').replace(']','')))
                       A4=[]
                       AOD_MSI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            AOD_MSI.append(A3)
                            
                       
                       AOD_MSI= np.array(AOD_MSI)
                       
                       
                       
                       CHL_OLI= np.array(file1['chlor_a(yoc)_OLI'][z]) 
        
                       A=CHL_OLI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('     ]',']').replace('  ]',']').replace('[  ','['). replace('[  ','[').replace('[    ','[').replace('[ ','[').replace(' ]',']').replace('[','').replace(']','')))
                       A4=[]
                       CH_OLI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            CH_OLI.append(A3)
                            
                       
                       CH_OLI= np.array(CH_OLI)
                       
                       CHL_MSI= np.array(file1['chlor_a(yoc)'][z]) 
        
                       A=CHL_MSI.tolist()
                       #A2 = np.array([int(x.replace('.','')) for x in A[A.find('[')+1:A.find(']')].split()])
                       A2 = np.array(re.split("\s+", A.replace('     ]',']').replace('  ]',']').replace('[  ','['). replace('[  ','[').replace('[    ','[').replace('[ ','[').replace(' ]',']').replace('[','').replace(']','')))
                       A4=[]
                       CH_MSI=[]
                       for y in range(len(A2)):
                            A3=float(A2[y])
                            CH_MSI.append(A3)
                            
                       
                       CH_MSI= np.array(CH_MSI)
                       
                       
                       
                               
                
                
                
                

                      
                      
  
                       
                       
                       
                ################################3333 Main Compariosn #########################################3
                                                  # The Band to Compare 
                                                                           
                       
                        
                        
                        
                               
                       #Bands MSI
                       
                       B_MSI= np.array(file1[MSI_Band][z] )    

      
                       B_MSI_A=B_MSI.tolist()
                       #B_MSI_A2 = np.array([int(x.replace(',','')) for x in B_MSI_A[B_MSI_A.find('[')+1:B_MSI_A.find(']')].split()])
                       #B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace('        ]',']').replace('  ]',']').replace(' ]',']').replace('[','').replace(']','')))
                       B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace(',',' ').replace('        ]',']').replace('        ]',']').replace('     ]',']').replace('   ]',']').replace('  ]',']').replace(' ]',']').replace('[          ','').replace('[','').replace(']','')))
                       
                       size=int(np.sqrt(len(B_MSI_A2)))  
                       
                       MSI=[]
    
                       for y in range(len(B_MSI_A2)):
                            A3=float(B_MSI_A2[y])  # np.pi
                            MSI.append(A3)
                       MSI_img= np.array(MSI)# window applied already in the server
                       B_MSI_A4=[]  
                       
                       
                       
                       B_MSI_1= np.array(file1['Rrs_443_S2_OC'][z] )
      
                       B_MSI_A_1=B_MSI_1.tolist()
                       #B_MSI_A2 = np.array([int(x.replace(',','')) for x in B_MSI_A[B_MSI_A.find('[')+1:B_MSI_A.find(']')].split()])
                       #B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace('        ]',']').replace('  ]',']').replace(' ]',']').replace('[','').replace(']','')))
                       B_MSI_A2_1 = np.array(re.split("\s+", B_MSI_A_1.replace(',',' ').replace('        ]',']').replace('        ]',']').replace('     ]',']').replace('   ]',']').replace('  ]',']').replace(' ]',']').replace('[          ','').replace('[','').replace(']','')))
                       
                       size=int(np.sqrt(len(B_MSI_A2_1)))  
                       
                       MSI_1=[]
    
                       for y in range(len(B_MSI_A2_1)):
                            A3=float(B_MSI_A2_1[y])  # np.pi
                            MSI_1.append(A3)
                       MSI_img_1= np.array(MSI_1)# window applied already in the server
                       B_MSI_A4_1=[]  
                       
                       #Bands MSI 2
                       B_MSI_2= np.array(file1['Rrs_492_S2_OC'][z] )
      
                       B_MSI_A_2=B_MSI_2.tolist()
                       #B_MSI_A2 = np.array([int(x.replace(',','')) for x in B_MSI_A[B_MSI_A.find('[')+1:B_MSI_A.find(']')].split()])
                       #B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace('        ]',']').replace('  ]',']').replace(' ]',']').replace('[','').replace(']','')))
                       B_MSI_A2_2 = np.array(re.split("\s+", B_MSI_A_2.replace(',',' ').replace('        ]',']').replace('        ]',']').replace('     ]',']').replace('   ]',']').replace('  ]',']').replace(' ]',']').replace('[          ','').replace('[','').replace(']','')))
                       
                       size=int(np.sqrt(len(B_MSI_A2_2)))  
                       
                       MSI_2=[]
    
                       for y in range(len(B_MSI_A2_2)):
                            A3=float(B_MSI_A2_2[y])  # np.pi
                            MSI_2.append(A3)
                            
    #                                      
    #                   MSI_img= np.array(MSI_img)
                       MSI_img_2= np.array(MSI_2)# window applied already in the server
                       B_MSI_A4_2=[]  
                       
                       
                       #Bands MSI 3
                       B_MSI_3= np.array(file1['Rrs_560_S2_OC'][z] )
      
                       B_MSI_A_3=B_MSI_3.tolist()
                       #B_MSI_A2 = np.array([int(x.replace(',','')) for x in B_MSI_A[B_MSI_A.find('[')+1:B_MSI_A.find(']')].split()])
                       #B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace('        ]',']').replace('  ]',']').replace(' ]',']').replace('[','').replace(']','')))
                       B_MSI_A2_3 = np.array(re.split("\s+", B_MSI_A_3.replace(',',' ').replace('        ]',']').replace('        ]',']').replace('     ]',']').replace('   ]',']').replace('  ]',']').replace(' ]',']').replace('[          ','').replace('[','').replace(']','')))
                       
                       size=int(np.sqrt(len(B_MSI_A2_3)))  
                       
                       MSI_3=[]
    
                       for y in range(len(B_MSI_A2_3)):
                            A3=float(B_MSI_A2_3[y])  # np.pi
                            MSI_3.append(A3)
                            
    #                                      
    #                   MSI_img= np.array(MSI_img)
                       MSI_img_3= np.array(MSI_3)# window applied already in the server
                       B_MSI_A4_3=[]  
                       
                       
                                              
                       #Bands MSI 4
                       B_MSI_4= np.array(file1['Rrs_665_S2_OC'][z] )
      
                       B_MSI_A_4=B_MSI_4.tolist()
                       #B_MSI_A2 = np.array([int(x.replace(',','')) for x in B_MSI_A[B_MSI_A.find('[')+1:B_MSI_A.find(']')].split()])
                       #B_MSI_A2 = np.array(re.split("\s+", B_MSI_A.replace('        ]',']').replace('  ]',']').replace(' ]',']').replace('[','').replace(']','')))
                       B_MSI_A2_4 = np.array(re.split("\s+", B_MSI_A_4.replace(',',' ').replace('        ]',']').replace('        ]',']').replace('     ]',']').replace('   ]',']').replace('  ]',']').replace(' ]',']').replace('[          ','').replace('[','').replace(']','')))
                       
                       size=int(np.sqrt(len(B_MSI_A2_4)))  
                       
                       MSI_4=[]
    
                       for y in range(len(B_MSI_A2_4)):
                            A3=float(B_MSI_A2_4[y])  # np.pi
                            MSI_4.append(A3)
                            
    #                                      
    #                   MSI_img= np.array(MSI_img)
                       MSI_img_4= np.array(MSI_4)# window applied already in the server
                       B_MSI_A4_4=[]  
                       
                      
                    
                
                       
                       B_OLI= np.array(file1[OLI_Band][z] )  
                       B_OLI_A=B_OLI.tolist()
                       #B_OLI_A2 = np.array([int(x.replace('.','')) for x in B_OLI_A[B_OLI_A.find('[')+1:B_OLI_A.find(']')].split()])
                       B_OLI_A2 = np.array(re.split("\s+", B_OLI_A.replace('  ]',']').replace(' ]',']').replace('[    ','').replace('[ ','').replace('[','').replace(']','')))
                       B_OLI_A4=[]  
                       OLI_img = np.array(B_OLI_A2)
                       
                       #Bands OLI 1   
                       B_OLI_1= np.array(file1['Rrs_443_L8_OC'][z] )  
                       B_OLI_A_1=B_OLI_1.tolist()
                       #B_OLI_A2 = np.array([int(x.replace('.','')) for x in B_OLI_A[B_OLI_A.find('[')+1:B_OLI_A.find(']')].split()])
                       B_OLI_A2_1 = np.array(re.split("\s+", B_OLI_A_1.replace('  ]',']').replace(' ]',']').replace('[    ','').replace('[ ','').replace('[','').replace(']','')))
                       B_OLI_A4_1=[]  
                       OLI_img_1 = np.array(B_OLI_A2_1)
                       
                       
                       #Bands OLI 2   
                       B_OLI_2= np.array(file1['Rrs_483_L8_OC'][z] )  
                       B_OLI_A_2=B_OLI_2.tolist()
                       #B_OLI_A2 = np.array([int(x.replace('.','')) for x in B_OLI_A[B_OLI_A.find('[')+1:B_OLI_A.find(']')].split()])
                       B_OLI_A2_2 = np.array(re.split("\s+", B_OLI_A_2.replace('  ]',']').replace(' ]',']').replace('[    ','').replace('[ ','').replace('[','').replace(']','')))
                       B_OLI_A4_2=[]  
                       OLI_img_2 = np.array(B_OLI_A2_2)
                       
                       
                       #Bands OLI 3   
                       B_OLI_3= np.array(file1['Rrs_561_L8_OC'][z] )  
                       B_OLI_A_3=B_OLI_3.tolist()
                       #B_OLI_A2 = np.array([int(x.replace('.','')) for x in B_OLI_A[B_OLI_A.find('[')+1:B_OLI_A.find(']')].split()])
                       B_OLI_A2_3 = np.array(re.split("\s+", B_OLI_A_3.replace('  ]',']').replace(' ]',']').replace('[    ','').replace('[ ','').replace('[','').replace(']','')))
                       B_OLI_A4_3=[]  
                       OLI_img_3 = np.array(B_OLI_A2_3)
                       
                       #Bands OLI 4   
                       B_OLI_4= np.array(file1['Rrs_655_L8_OC'][z] )  
                       B_OLI_A_4=B_OLI_4.tolist()
                       #B_OLI_A2 = np.array([int(x.replace('.','')) for x in B_OLI_A[B_OLI_A.find('[')+1:B_OLI_A.find(']')].split()])
                       B_OLI_A2_4 = np.array(re.split("\s+", B_OLI_A_4.replace('  ]',']').replace(' ]',']').replace('[    ','').replace('[ ','').replace('[','').replace(']','')))
                       B_OLI_A4_4=[]  
                       OLI_img_4 = np.array(B_OLI_A2_4)
                       
                       
    
                       if len(OLI_img_1) <= len(MSI_img) :
                           L= len(OLI_img_1)
                       if len(MSI_img_1) <= len(OLI_img) :
                            L= len(MSI_img_1)
                            
                            
                            
                            
                     
                       band=   str(OLI_Band)     
                       if band[2:-2] == 'Rrs_655_Acolite':
                         band_th= 1#0.009    
                            
                       else:
                           band_th=1# 0.03
                           
                           
                           
                       B_MSI_AOD=[]
                       B_OLI_AOD=[]
                       
                       B_MSI_CH= []
                       B_OLI_CH=[]

                       for y in range(L):
                           
                           B_MSI_A3=  float(MSI_img[y])            # float(B_MSI_A2[y])
                           B_OLI_A3=  float(OLI_img[y])   
              
                           B_MSI_A3_1=  float(MSI_img_1[y])            # float(B_MSI_A2[y])
                           B_OLI_A3_1=  float(OLI_img_1[y])            # float(B_OLI_A2[y])
                                         
                           B_MSI_A3_2=  float(MSI_img_2[y])            # float(B_MSI_A2[y])
                           B_OLI_A3_2=  float(OLI_img_2[y])            # float(B_OLI_A2[y])
                                         
                           B_MSI_A3_3=  float(MSI_img_2[y])            # float(B_MSI_A2[y])
                           B_OLI_A3_3=  float(OLI_img_3[y]) 
                           # float(B_OLI_A2[y])              
                           B_MSI_A3_4=  float(MSI_img_4[y])            # float(B_MSI_A2[y])
                           B_OLI_A3_4=  float(OLI_img_4[y])            # float(B_OLI_A2[y])
                           
                           CHL_B_MSI_ = float(CHL_B_MSI[y])
                           CHL_B_OLI_ = float(CHL_B_OLI[y])
                           AOD_L8 = float(AOD_OLI[y])
                           AOD_S2 = float(AOD_MSI[y])
                           CH_L8=    float(CH_OLI[y])
                           CH_S2=   float(CH_MSI[y])
                           
                           th=1#0.001
        
                             
                           if  0.0 < B_MSI_A3 <= th and AOD_L8 < 0.5 and AOD_S2 < 0.5 and  np.isnan(B_MSI_A3)==False  and 0.0 < B_OLI_A3<= th and CHL_B_MSI_==0.0 and CHL_B_OLI_==0.0:
                            if    0.0 < B_MSI_A3_3 and 0.0 < B_OLI_A3_3 and  0.0 < B_MSI_A3_4 and 0.0 < B_OLI_A3_4  and  0.0 < B_MSI_A3_1 and 0.0 < B_OLI_A3_1 and  0.0 < B_MSI_A3_2 and 0.0 < B_OLI_A3_2:
                              
                            
                              B_MSI_A4.append(B_MSI_A3)
                              B_OLI_A4.append(B_OLI_A3)
                              
                              B_MSI_A4_1.append(B_MSI_A3_1)
                              B_OLI_A4_1.append(B_OLI_A3_1)
                              
                              B_MSI_A4_2.append(B_MSI_A3_2)
                              B_OLI_A4_2.append(B_OLI_A3_2)
                              
                              B_MSI_A4_3.append(B_MSI_A3_3)
                              B_OLI_A4_3.append(B_OLI_A3_3)
                              
                              B_MSI_A4_4.append(B_MSI_A3_4)
                              B_OLI_A4_4.append(B_OLI_A3_4)
                              
                              B_MSI_AOD.append(AOD_S2 )
                              B_OLI_AOD.append(AOD_L8 )
                              
                              B_OLI_CH.append(CH_L8)
                              B_MSI_CH.append(CH_S2)
                              
                              print("Pass Scenes", file1.loc[z, 'L8-Scene'],file1.loc[z, 'S2_Scenes'] )  
                              
                              
                              
                              
                              
                              
                              #if y == L-1:
                                  
                               #print("Pass Scenes", file1.loc[z, ['Mid_Lat'][0]],file1.loc[z, ['Mid_Lon'][0]], z )   
                           
                       if  len(B_OLI_A4) > 50:
                           
                           Band_B1_OL1= np.append(Band_B1_OL1,B_OLI_A4)
                           #Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           
                           Band_B1_OL1_1= np.append(Band_B1_OL1_1,B_OLI_A4_1)
                           #Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           Band_B1_MSI_1= np.append(Band_B1_MSI_1,B_MSI_A4_1)
                           
                           Band_B1_OL1_2= np.append(Band_B1_OL1_2,B_OLI_A4_2)
                           #Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           Band_B1_MSI_2= np.append(Band_B1_MSI_2,B_MSI_A4_2)
                           
                           Band_B1_OL1_3= np.append(Band_B1_OL1_3,B_OLI_A4_3)
                           #Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           Band_B1_MSI_3= np.append(Band_B1_MSI_3,B_MSI_A4_3)
                           
                           Band_B1_OL1_4= np.append(Band_B1_OL1_4,B_OLI_A4_4)
                           #Band_B1_MSI= np.append(Band_B1_MSI,B_MSI_A4)
                           Band_B1_MSI_4= np.append(Band_B1_MSI_4,B_MSI_A4_4)
                           
                            
                           AOD_OLI_L8= np.append(AOD_OLI_L8, B_OLI_AOD)
                           AOD_MSI_S2= np.append(AOD_MSI_S2, B_MSI_AOD)
                           
                           CH_OLI_L8 = np.append(CH_OLI_L8 , B_OLI_CH)
                           CH_MSI_S2 = np.append(CH_MSI_S2 , B_MSI_CH)
                           

                        
                print(len(Band_B1_OL1))        
                   
                #df = pd.DataFrame(E, columns= ['480 nm','492.4 nm'])
                is_training = tf.placeholder_with_default(False, (), 'is_training')
                
                

                
    #                   
                print(len(Band_B1_OL1))
                
                if MSI_Band=='Rrs_492_S2_OC':
                    th= 0.3
                    xlim= 0.002
                    ylim=0.06
                
                    
                if MSI_Band=='Rrs_443_S2_OC':
                    th= 0.4 
                    xlim= 0.0
                    ylim=0.05
                    
                    
                                
                if MSI_Band=='Rrs_560_S2_OC':
                    th= 0.3 
                    xlim= 0.002
                    ylim=0.1
                    
                if MSI_Band=='Rrs_665_S2_OC'   :
                    th= 0.3
                    xlim= 0.002
                    ylim=0.1
                    
                    
                    
                x_vals_train1 = [] #Band_B1_MSI   #ref data
                y_vals_train1 = [] #Band_B1_OL1   # MSI converted to OLI 
                ##  OLI= MSI.X + Bias_
                c=0
                for x in range(len(Band_B1_MSI)): 
                 #if np.abs(Band_B1_MSI[x]- Band_B1_OL1[x])*100 < 0.4 : #th:
                     
                 if np.abs(Band_B1_MSI[x]- Band_B1_OL1[x])/ np.abs(Band_B1_OL1[x])*100 <  100 * np.median(np.abs(Band_B1_MSI- Band_B1_OL1)/ np.abs(Band_B1_OL1)*100): 
                  if np.abs(Band_B1_MSI_1[x]- Band_B1_OL1_1[x])/ np.abs(Band_B1_OL1_1[x])*100 <  100 * np.median(np.abs(Band_B1_MSI_1- Band_B1_OL1_1)/ np.abs(Band_B1_OL1_1)*100): 
                   if np.abs(Band_B1_MSI_2[x]- Band_B1_OL1_2[x])/ np.abs(Band_B1_OL1_2[x])*100 <  100 * np.median(np.abs(Band_B1_MSI_2- Band_B1_OL1_2)/ np.abs(Band_B1_OL1_2)*100):
                    if np.abs(Band_B1_MSI_3[x]- Band_B1_OL1_3[x])/ np.abs(Band_B1_OL1_3[x])*100 <  100 * np.median(np.abs(Band_B1_MSI_3- Band_B1_OL1_3)/ np.abs(Band_B1_OL1_3)*100):   
                     if np.abs(Band_B1_MSI_4[x]- Band_B1_OL1_4[x])/ np.abs(Band_B1_OL1_4[x])*100 <  100 * np.median(np.abs(Band_B1_MSI_4- Band_B1_OL1_4)/ np.abs(Band_B1_OL1_4)*100):   

                        x_vals_train1.append(Band_B1_MSI[x])
                        y_vals_train1.append(Band_B1_OL1[x])
 
                  
                print('Len of X data', len(x_vals_train1))
                
                x_batch=  np.reshape(x_vals_train1, [len(x_vals_train1),1])
                y_batch=  np.reshape(y_vals_train1, [len(y_vals_train1),1])
                
                
                x_batch_org=x_batch
                y_batch_org=y_batch
                
                                            
                init = tf.global_variables_initializer() 
    #
    
    
    
    neurons1=25# =25
    print ("Number of Neurons", neurons1)
    #Note epoch=500, Batch
    Epoch=1000
    batch_size= 32#64  #32    # 128 was used before
    #    learning_rate=0.0075 #0.00075
    momentum=0.5
    
    neurons2=25 #15
    neurons3=25 #5
    neurons4 = 5
    
    neurons5=25
    tf.set_random_seed(1)
    np.random.seed(1)
    #phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float32)
    def neural_net_model(X_data,input_dim):
                        #W_1 = tf.Variable(tf.random_uniform([input_dim,100]))
                        W_1 = tf.Variable(initializer([input_dim,neurons1]))                                # this one
                        #W_1=tf.Variable(tf.truncated_normal([input_dim,neurons1],mean = 0.0,stddev=0.1))
                        #b_1 = tf.Variable(tf.zeros([250]))
                        b_1 = tf.Variable(initializer([neurons1])) # this one
                        #b_1=tf.Variable(tf.constant(0.1,shape = [neurons1]))
                        layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
                        layer_1 = tf.nn. tanh(layer_1)
                        #layer_1= tf.layers.batch_normalization(layer_1)
                    
                        layer_1=  tf.layers.batch_normalization(layer_1, training=is_training)
                        layer_1  = tf.nn.dropout( layer_1 , keep_prob)
                        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        #with tf.control_dependencies(update_ops):
                        layer_1 = tf.identity(layer_1)
                    
            
                        #W_2 = tf.Variable(tf.random_uniform([100,1000]))
                        W_2 = tf.Variable(initializer([neurons1,neurons2]))   # this one
                        #W_2=tf.Variable(tf.truncated_normal([neurons1,neurons2],mean = 0.0,stddev=0.1))
                        #b_2 = tf.Variable(tf.zeros([1000]))
                        b_2 = tf.Variable(initializer([neurons2])) # this one
                        #b_2=tf.Variable(tf.constant(0.1,shape = [neurons2]))
                        layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
                        layer_2 = tf.nn. tanh(layer_2)
                        #layer_2= tf.layers.batch_normalization(layer_2)
                        layer_2=  tf.layers.batch_normalization(layer_2, training=is_training)
                        layer_2  = tf.nn.dropout( layer_2 , keep_prob)
                        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        #with tf.control_dependencies(update_ops):
                        layer_2 = tf.identity(layer_2)
            #            
                        #W_3 = tf.Variable(tf.random_uniform([1000,1000]))
                        W_3 = tf.Variable(initializer([neurons2,neurons3])) # this one
                        #b_3 = tf.Variable(tf.zeros([1000]))
                        b_3 = tf.Variable(initializer([neurons3]))  # this one
                        
                        layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
                        #layer_3 = tf.nn.leaky_ tanh(layer_3)
                        layer_3 = tf.nn. tanh(layer_3)
                        #layer_3= tf.layers.batch_normalization(layer_3)
                    
                        layer_3=  tf.layers.batch_normalization(layer_3, training=is_training)
                        layer_3  = tf.nn.dropout( layer_3 , keep_prob)
                        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        #with tf.control_dependencies(update_ops):
                        layer_3 = tf.identity(layer_3)
                        
#                                W_3 = tf.Variable(tf.random_uniform([1000,1000]))
#                        W_4 = tf.Variable(initializer([neurons3,neurons4])) # this one
#                        #b_3 = tf.Variable(tf.zeros([1000]))
#                        b_4 = tf.Variable(initializer([neurons4]))  # this one
#                        
#                        layer_4 = tf.add(tf.matmul(layer_3,W_4), b_4)
#                        #layer_3 = tf.nn.leaky_ tanh(layer_3)
#                        layer_4 = tf.nn. tanh(layer_4)
#                        #layer_3= tf.layers.batch_normalization(layer_3)
#                       
#                        layer_4=  tf.layers.batch_normalization(layer_4, training=is_training)
#                        layer_4  = tf.nn.dropout( layer_4 , keep_prob)
#                       
#                        #with tf.control_dependencies(update_ops):
#                        layer_4 = tf.identity(layer_4)
                        
                            
                        # layer 2 multiplying and adding bias then activation function
                        #W_O = tf.Variable(tf.random_uniform([1000,1]))
                        W_O = tf.Variable(initializer([neurons3,1]),name='W_O')
                        #W_O=tf.Variable(tf.truncated_normal([neurons,1],mean = 0.0,stddev=0.1))
                        #b_O = tf.Variable(tf.zeros([1]))
                        b_O = tf.Variable(initializer([1]),name='b_O')
                        #b_O=tf.Variable(tf.constant(0.1,shape = [1]))
                        output = tf.add(tf.matmul(layer_3,W_O), b_O)
                        return output,W_O,b_O            # -0.057620753 0.875748187999985 -0.51166356 0.30784807
    #                    

    
    # Normalizing
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    #x_batch = scaler.fit_transform(x_batch)
    #x_batch = np.log10(x_batch)
    
    x_batch= np.concatenate((x_batch, scaler.fit_transform(x_batch), np.log10(x_batch)),axis=1)
    #y_batch = scaler.fit_transform(y_batch)
    y_batch = np.log10(y_batch)
    
    
    lr=0.0075
    
    #x_data = tf.placeholder(tf.float32, [None,1], name = 'x')  # number of features == number of columns 
    x_data = tf.placeholder(tf.float32, [None,x_batch.shape[1]], name = 'x')
    keep_prob = tf.placeholder(tf.float32)
    y_target = tf.placeholder(tf.float32, [None, 1], name = 'y')  # number of outputs  == number of columns
    L_data = tf.placeholder(tf.float32, [1], name = 'z')  # number of features == number of columns
    n_d=len(x_batch[0])

        
    initializer = tf.contrib.layers.xavier_initializer(uniform=False,seed=10)
       # initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True,seed=1)
    model_output,W_O,b_O= neural_net_model(x_data,n_d)
    epsilon = tf.constant([0.5])
    #loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))
    #loss = tf.reduce_mean(tf.square(y_target-model_output))
    #losss=tf.nn.l2_loss(W_O) + tf.nn.l2_loss(b_O)+ tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) +tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate =  tf.train.exponential_decay(lr, global_step,100, 0.95, staircase=True)  #learning_rate =  tf.train.exponential_decay(lr, global_step,100, 0.95, staircase=True)
    #learning_rate = tf.train.exponential_decay(lr, 0, 15, 0.1, staircase=False)  
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_target, model_output)))) 
    loss = tf.add_n([loss] + reg_losses, name="loss")
    
    #                loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_target, model_output)))) 
    ##                W_2 = tf.Variable(initializer([25,25]))
    ##                regularizer = tf.nn.l2_loss(W_2)
    #                loss = tf.reduce_mean(loss + 0.01*reg_losses)
    
    #loss=tf.losses.mean_squared_error(y_target,model_output)
    
    #my_opt = tf.train.GradientDescentOptimizer(0.1)
    #learning_rate=0.00075
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #this one
    my_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)#,beta1=0.9, beta2=0.999, epsilon=1e-10)
    #my_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9,epsilon=1e-10)
    #my_opt = tf.train.RMSPropOptimizer(learning_rate=0.0001)
    # clippig gradient
    #my_opt = tf.contrib.estimator.clip_gradients_by_norm(my_opt, clip_norm=1.0)
    #my_opt = tf.train.AdagradOptimizer(learning_rate=lr)
    
    
    
    
    #my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)#learning_rate
    #my_opt=tf.train.MomentumOptimizer(learning_rate, momentum)
       # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    saver = tf.train.Saver()   
    train_step = my_opt.minimize(loss,global_step)
            
    from sklearn.model_selection import KFold 
    #from sklearn.model_selection import RepeatedKFold
    #kf = KFold(len(x_batch),True,1) 
    kf = KFold(5,random_state=None,shuffle=True) # 5,random_state=14,shuffle=True
    for train_index, test_index in kf.split(x_batch):
            x_vals_train1, x_vals_test = x_batch[train_index], x_batch[test_index] 
            y_vals_train1, y_vals_test = y_batch[train_index], y_batch[test_index]
            
            train_indices = np.random.choice(len(x_vals_train1), round(len(x_vals_train1)*0.70), replace=False)
            test_indices = np.array(list(set(range(len(x_vals_train1))) - set(train_indices)))
            x_vals_train = x_vals_train1[train_indices]
            x_vals_val = x_vals_train1[test_indices]
            y_vals_train = y_vals_train1[train_indices]
            y_vals_val = y_vals_train1[test_indices]
            
            x_vals_train1=np.concatenate((x_vals_train,x_vals_val),axis=0)
            y_vals_train1=np.concatenate((y_vals_train,y_vals_val),axis=0)
                
    
    
            
            
    #                    x_data = tf.placeholder(tf.float32, [len(x_vals_train),x_batch.shape[1]], name = 'x')  # number of features == number of columns 
    #                    keep_prob = tf.placeholder(tf.float32)
    #                    y_target = tf.placeholder(tf.float32, [len(y_vals_train), y_batch.shape[1]], name = 'y')  # number of outputs  == number of columns
    #                    L_data = tf.placeholder(tf.float32, [1], name = 'z')  # number of features == number of columns
    #                    n_d=len(x_batch[0])
            
     
            train_loss = []
            test_loss =[]
            best_R2=-1000000
            Best_R2_val=-10000
            
            test_R2=[]
            train_R2=[]
            val_R2=[]
            print("starting Training the Model ")
        
            init = tf.global_variables_initializer()
            train_step=tf.group([train_step,update_ops])
            
            
            dropout_prob = 1
            eval_Loss=[]
            val_loss=[]
            
            best_rmse_val=25
            best_mape_val= 1000
            best_rmse_train=25 
            best_med = 25
            E=100
            R2_Test=[]
            RMSE_Test =[]
            with tf.Session() as sess:
                sess.run(init)
                sess.run(tf.global_variables_initializer())    
                
                print("startingTraining1")
                for i in range(5):
            #       
                    # Batch Normalizatin##############################################3##
                    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
                    X = np.transpose([x_vals_train[rand_index]])
                    X = x_vals_train[rand_index]
                    Y = np.transpose([y_vals_train[rand_index]])
                    #####################################################################
                    # shuffle data
                    from sklearn.utils import shuffle
                    #x_vals_train, y_vals_train = shuffle(x_vals_train,y_vals_train )
                    #x_vals_val, y_vals_val = shuffle(x_vals_val,y_vals_val )
                    #x_vals_test, y_vals_test = shuffle(x_vals_test,y_vals_test )
                    ###
    #                        X = x_vals_train
    #                        Y = np.transpose([y_vals_train])
                    s=Y.shape[1]
                    Y= np.reshape(Y, [s,1])
                    sess.run(train_step, feed_dict={x_data: X, y_target: Y,keep_prob:dropout_prob,is_training: True})
                    
                    temp_train_loss = sess.run(loss, feed_dict={x_data:x_vals_train, y_target:y_vals_train,keep_prob:dropout_prob,is_training: True})
                    train_loss.append(temp_train_loss)
                    temp_val_loss = sess.run(loss, feed_dict={x_data:x_vals_val, y_target:y_vals_val,keep_prob:1,is_training: False})
                    val_loss.append(temp_val_loss)
                    
    #                        temp_test_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target: y_vals_test,keep_prob:1,is_training: False})
    #                        test_loss.append(temp_test_loss)
                    
                    temp_train_R2 =r2_score(y_vals_train, sess.run( model_output, feed_dict={x_data:x_vals_train,keep_prob:dropout_prob,is_training: True}))
                    train_R2.append(temp_train_R2)
                    temp_val_R2 = r2_score(y_vals_val, sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))
                    temp_val_mape= 100*mean_absolute_percentage_error(y_vals_val, sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))
                    val_R2.append(temp_val_R2)
                    MD_train= abs(np.median(y_vals_val - sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: True})))
                    MD_val= np.median(y_vals_val - sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))
    #                        temp_test_R2 = r2_score(y_vals_test, sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))
    #                        test_R2.append(temp_test_R2)
                    
    #                        y_pred_batch_tr = sess.run(model_output, feed_dict={x_data:x_vals_train, y_target: y_vals_train,keep_prob:dropout_prob,is_training: True})
    #                        y_pred_batch_val = sess.run(model_output, feed_dict={x_data:x_vals_val, y_target: y_vals_val,keep_prob:1,is_training: False})
                    MD_val= abs (MD_val)
                    
                    if best_rmse_val > temp_val_loss: # and best_mape_val > temp_val_mape: # and best_R2 < temp_val_R2) and  
                       best_rmse_val =temp_val_loss
                       best_rmse_train= temp_train_loss
                       best_R2 = temp_val_R2
                       best_med = MD_val
                       best_mape_val  = temp_val_mape
                       best_epoch=i
    #                           Te_loss=temp_test_loss
    #                           Te_R2=temp_test_R2
                       y_pred_batch = sess.run(model_output, feed_dict={x_data:x_vals_test, y_target: y_vals_test,keep_prob:1,is_training: False})
                       
                      
                       #saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                       #R2_T = r2_score(y_vals_test,y_pred_batch)
                    #R2_T =  sess.run(r2_score(y_vals_test, y_pred_batch)) 
                    
                       rmse_T= sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_vals_test, y_pred_batch)))))
                       R2_T=   r2_score(y_vals_test, sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))  
                       MD_T= np.median(y_vals_test - sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))
                       mape_T= 100*mean_absolute_percentage_error(y_vals_test, sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))
#                       if  i> 3000:   
#                         saver.save(sess, 'my_model')
#                         
#                         print('Model Saved',i)   
                    
    #                RMSE_Test.append(rmse_T)   
    #                R2_Test.append(R2_T)
    
    #                        rmse_T = sess.sun(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_vals_test, y_pred_batch)))))
    #                        RMSE_Test.append(rmse_T)
                                                
                          
                    if (i % 1000==0):   
    #                           print(i, "Trian Loss:",temp_train_loss,"Val_Loss=", temp_val_loss, "Test Loss:",temp_test_loss, "Best_val_Loss:", best_rmse_val)
    #                           print(i,'R2score_train:',temp_train_R2,'R2score_val:',temp_val_R2, 'R2score_test:',temp_test_R2, 'Best_R2_val=',best_R2)  
                        
                       print(i, "Trian Loss:",temp_train_loss,"Val_Loss=", temp_val_loss, "Best_val_Loss:", best_rmse_val, "Test Loss:", (rmse_T))
                       print(i, "temp_val_mape=:", temp_val_mape,  "Test_mape:", mape_T)
                       print(i,'R2score_train:',temp_train_R2,'R2score_val:',temp_val_R2, 'R2score_test:',R2_T, 'Best_R2_val=',best_R2)
                       
                       
           
                          
                
                sess.close()    
                R2_Test_All.append(R2_Test)       
                
                Pre_Data = np.append (Pre_Data,y_pred_batch ) #Pre_Data = np.append (Pre_Data,y_pred_batch )
                OLI_Data = np.append (OLI_Data,y_vals_test )  # This data is to check the model performamce 
                
#                Pre_Rrs_All_Bands.append(Pre_Data)
#                Ref_Rrs_All_Bands.append(MSI_Data)
                
                #MSI_Data= np.append (MSI_Data,x_vals_test[:,2] )    # Ref
                v= x_vals_test[:,2]
                v= np.reshape(v, [len(v),1])
                MSI_Data= np.append (MSI_Data, v )    # Ref
                #reg = linear_model.LinearRegression().fit(np.reshape(x_vals_train1[:,2], [len(x_vals_train1[:,2]),1]), y_vals_train1)
                reg = linear_model.LinearRegression().fit(np.reshape(x_vals_train1, [len(x_vals_train1),3]), y_vals_train1)
                #lin= reg.predict(np.reshape(x_vals_test[:,2], [len(x_vals_test[:,2]),1]))
                lin= reg.predict(np.reshape(x_vals_test, [len(x_vals_test),3]))
                Linear_P=np.append(Linear_P, lin)
                print('len of MSI Data.............', len(MSI_Data), len(x_vals_test))
                 ############Guassian
#                import sklearn.gaussian_process as gp
#                from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#                from sklearn.metrics import mean_squared_error
#                from sklearn.svm import SVR
#                from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
#                
#                
#                #kernel =   C(1.0, (1e-3, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
#                #kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(14.0, (1e-10, 1e3))
#                #kernel = 1.0 * gp.kernels.RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + gp.kernels.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)) # best
#                kernel = DotProduct() + WhiteKernel()
#                kernel= 1 * gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
#                dy = np.random.random(y_vals_train1.shape)
#                dy1=np.reshape(dy,-1)
#                al=np.array([0.005, 0.01,0.03,0.05,0.1])
#                #al=np.array([0.005, 0.01])
#                el=np.array([1])
#                GP_Val=[]
#                for v in range (len(al)):
#                  
#                     #kernel= gp.kernels.RBF(14.0, (1e-10, 1e1)) +   gp.kernels.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
#                     #model = gp.GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=1, alpha=0.05, normalize_y=True)
#                     model = gp.GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=1, alpha=0.05, normalize_y=True)
#                     #model = SVR(kernel='rbf', C=al[v],epsilon=al[b])
#                     #model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=al[v], normalize_y=True)
#                     model.fit(x_vals_train[0:1000,:], y_vals_train[0:1000])
#                     #params = model.kernel_.get_params()
#                     y_pred_tr= model.predict(x_vals_train)
#                     G_MSE_TR = ((y_vals_train-y_pred_tr)**2).mean()   
#                     G_R2_Tr=r2_score(y_vals_train, y_pred_tr) 
#                     y_pred_val = model.predict(x_vals_val)
#                     G_MSE_val = ((y_vals_val-y_pred_val)**2).mean()  
#                    
#                     G_MSRE_val=np.sqrt(mean_squared_error(y_vals_val,y_pred_val))
#                     GP_Val=np.append(GP_Val,  G_MSRE_val)
#                     G_R2_val=r2_score(y_vals_val, y_pred_val) 
#    #                         print ("R2_score value of GP:Validation=", G_R2_val)
#    #                         print ("MSE of GP Validation=", G_MSE_val)
#                m=min( GP_Val)
#                mi=np.where(GP_Val==m)
#        
#                alpha1=al[mi]  
#                alpha1= alpha1[0]
#                #alpha1=0.1
#                print('Best Alpha:', alpha1)    
#                model = gp.GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=0, alpha=alpha1, normalize_y=False)
#                #model = SVR(kernel='rbf', C=al[v],epsilon=el[0])
#                     #  model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
#                model.fit(x_vals_train[0:,:], y_vals_train[0:])
#                #params = model.kernel_.get_params()
#                #y_pred_tr, std = model.predict(x_vals_train, return_std=True)
#                #y_pred_tr = model.predict(x_vals_train[0:100,:])
#                #y_pred_val= model.predict(x_vals_val[0:100,:])
#                #y_pred_val, std = model.predict(x_vals_val, return_std=True)
#                #y_pred_val = model.predict(x_vals_val[0:100,:])
#                #G_MSE_TR = ((y_vals_train[0:100]-y_pred_tr)**2).mean()   
#                #G_R2_Tr=r2_score(y_vals_train[0:100], y_pred_tr)
#                #G_R2_val=r2_score(y_vals_val[0:100], y_pred_val)
#                #print ("R2_score value of GP:Training=", G_R2_Tr)
#                #print ("R2_score value of GP:Validation=", G_R2_val)
#                #print ("MSE of GP Training=", G_MSE_TR)
#                
#                #y_pred, std = model.predict(x_vals_test, return_std=True)
#                y_pred= model.predict(x_vals_test[0:,:])
#                G_MSE = ((y_vals_test[0:]-y_pred)**2).mean()
#                G_MSRE=np.sqrt(mean_squared_error(y_vals_test[0:],y_pred))
#                G_R2=r2_score(y_vals_test[0:], y_pred) 
#                GP= np.append(GP,y_pred)
#                print ("R2_score value of GP Testing=", G_R2)
#                print ("MSE of GP Testing=", G_MSE)
#                print ("RMSE of GP Testing=", G_MSRE)
                    
                
    
    
#    np.save('Red_Weights',  Weights, allow_pickle=True, fix_imports=True)
#    np.save('Red_Bias', Bias, allow_pickle=True, fix_imports=True)
#    
#    np.save('Red_W_median', W_median, allow_pickle=True, fix_imports=True)
#    np.save('Red_B_median', B_median, allow_pickle=True, fix_imports=True)
    
    
       
                
                
                
    
        
                            
                        
#    MSI_Data=10**np.array(MSI_Data)   
#    Pre_Data=10**np.array(Pre_Data)  
#    OLI_Data=10**np.array(OLI_Data)
#    Linear_P= 10**np.array(Linear_P)
    
    MSI_Data=10**(MSI_Data)   
    
    Pre_Data=10**(Pre_Data)  
    
    #GP_Data= 10**(GP)
    OLI_Data=10**(OLI_Data)
    Linear_P= 10**(Linear_P)
    
    MSI_Data= MSI_Data * np.pi
    Linear_P= np.pi*(Linear_P)
    OLI_Data=np.pi*(OLI_Data)
    Pre_Data=np.pi*(Pre_Data)  
    
    
    
    
#    np.save('MSI_Data'+ OLI_Band[4:7],  MSI_Data, allow_pickle=True, fix_imports=True)
#    np.save('Pre_Data'+OLI_Band[4:7], Pre_Data, allow_pickle=True, fix_imports=True)
#    #np.save('GP_Data'+OLI_Band[4:7], GP_Data, allow_pickle=True, fix_imports=True)
#    np.save('Linear_P'+OLI_Band[4:7], Linear_P, allow_pickle=True, fix_imports=True)
#    np.save('OLI_Data'+OLI_Band[4:7], OLI_Data, allow_pickle=True, fix_imports=True)
##    
    MSI_Data= np.load('MSI_Data'+ str(OLI_Band[4:7]) + '.npy')
    Pre_Data= np.load('Pre_Data' +  str(OLI_Band[4:7]) + '.npy')
    #GP_Data= np.load('GP_Data' +  str(OLI_Band[4:7]) + '.npy')
    Linear_P= np.load('Linear_P' +  str(OLI_Band[4:7]) + '.npy' )
    
    OLI_Data= np.load('OLI_Data' +  str(OLI_Band[4:7]) + '.npy')

        ###################################################################################
    ### NO Band Adjustment 
    plt.figure()
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.85)
    xdata =  OLI_Data 
    ydata =  MSI_Data 
    
    indy= np.where(ydata == np.amax(ydata))
    ydata= np.delete(ydata, indy)
    xdata= np.delete(xdata, indy)
    
    indx= np.where(xdata == np.amax(xdata))
    xdata= np.delete(xdata, indx)
    ydata= np.delete(ydata, indx)
    
    
    xdata=np.reshape(xdata, [len(xdata),1])
    ydata=np.reshape(ydata, [len(ydata),1])
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
    #plt.colorbar()
    #    a, b = best_fit(xdata, ydata)
    #    yfit = [a + b * xi for xi in xdata]  
        #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
    line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    #ax.set_xlabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    #ax.set_ylabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2',fontsize=30)
    
    ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2',fontsize=30)
    

    
    

#    min1= np.min (xdata)
#    min2= np.min (ydata)
    ax.set_ylim(xlim, ylim)
    ax.set_xlim(xlim, ylim) 
#       
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.legend(fontsize=24)
    ax.set_title('No-Bandpass Adjustment',fontsize=30)
    for axis in ['top','bottom','left','right']:
     ax.spines[axis].set_linewidth(4)
    plt.savefig('No BA-'+'Band' + MSI_Band[0:7], bbox_inches = 'tight', dpi=600)
    plt.show()
    

    MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
    MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
    rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
    rmse=mean_squared_error(ydata, xdata, squared=False)
    BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
    MD= np.median(xdata - ydata)
    z=np.median((np.log(ydata/xdata)))
    z= 100 * np.sign(z) * (10**abs(z) -1)
    
    y=np.median(abs((np.log(ydata/xdata))))
    y=100 * (10**y -1)
    
    r2= r2_score(xdata,ydata)
    rRMSD= np.sqrt(np.mean(np.square((ydata-xdata)/xdata)))*100
    
    corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
    r= corr1 * corr1
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))
    
    
    print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
    f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
    print('Results for Band' + MSI_Band[0:7], '\n', file=f)
    print('Results before Band Conversion:','\n', file=f)
    print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
    print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)  
      
    plt.figure()
    fig, ax = plt.subplots()
    kwargs = dict(alpha=0.5, bins=50)
    plt.hist(MSI_Data, **kwargs, color='orange', label='MSI')
    plt.hist(OLI_Data, **kwargs, color='dodgerblue', label='OLI')
    plt.gca().set(title='Frequency Histogram of OLI-MSI')
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    plt.xlim(0.0001,ylim)
    plt.legend();
    plt.savefig('xxorg Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    
    plt.figure(figsize=(10,7), dpi= 80)
    fig, ax = plt.subplots()
    kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':4})
    sns.distplot(MSI_Data, color="orange", label="MSI", **kwargs)
    sns.distplot(OLI_Data, color="dodgerblue", label="OLI", **kwargs)
    
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    plt.xlim(0.000,ylim)
    plt.legend();
    plt.savefig('xxorg Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    
    
    #plt.legend();
    #############################3
    ################ NN ####################
        
    plt.figure()
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.85)
    xdata =  (OLI_Data)
    ydata =  (Pre_Data)
    
    
    indy= np.where(ydata == np.amax(ydata))
    ydata= np.delete(ydata, indy)
    xdata= np.delete(xdata, indy)
    
    indx= np.where(xdata == np.amax(xdata))
    xdata= np.delete(xdata, indx)
    ydata= np.delete(ydata, indx)
    
    xdata=np.reshape(xdata, [len(xdata),1])
    ydata=np.reshape(ydata, [len(ydata),1])
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
    #plt.colorbar()
    #    a, b = best_fit(xdata, ydata)
    #    yfit = [a + b * xi for xi in xdata]  
    #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
    line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    #ax.plot(xdata, yfit, label='Fitted Line',linewidth=1.0)
    #ax.set_xlabel('OLI_L8',fontsize=18)
    #ax.set_ylabel('OLI_L8' + '*' + ' '+ '[R$rs$_443 nm]', fontsize=18)
    
    #ax.set_ylabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)
    #ax.set_xlabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    
    ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)
    
    
    #ax.plot([],[],' ',color = "blue", label='OCN')
    ax.set_title('NN Bandpass Adjustment',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
    ax.set_ylim(xlim, ylim)
    ax.set_xlim(xlim, ylim) 
    #ax.set_xticks(xlim, ylim,0.04)
    
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    plt.legend(fontsize=24)
    
    plt.savefig('Conversion-MSI-OLI-8' + 'Band' + str(MSI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    plt.show()
    
    MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
    MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
    rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
    rmse=mean_squared_error(ydata, xdata, squared=False)
    BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
    MD= np.median(xdata - ydata)
    z=np.median((np.log(ydata/xdata)))
    z= 100 * np.sign(z) * (10**abs(z) -1)
    
    y=np.median(abs((np.log(ydata/xdata))))
    y=100 * (10**y -1)
    
    r2= r2_score(xdata,ydata)
    
    corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
    r= corr1 * corr1
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))
    
    print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
    f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
    print('Results for Band' + MSI_Band[0:7], '\n', file=f)
    print('Results after NN Band Conversion:','\n', file=f)
    print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
     
    print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)  
       

    plt.figure()
    fig, ax = plt.subplots()
    kwargs = dict(alpha=1, bins=10)
    plt.hist(xdata, **kwargs, color='orange', label='OLI')
    plt.hist(ydata, **kwargs, color='dodgerblue', label='MSI')
    plt.gca().set(title='Frequency Histogram of OLI-MSI'+'*')
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    plt.xlim(0.0001,ylim)
    plt.savefig('NN Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    

    plt.figure(figsize=(10,7), dpi= 80)
    fig, ax = plt.subplots()
    kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':4})
    sns.distplot(xdata, color="dodgerblue", label="OLI", **kwargs)
    sns.distplot(ydata, color="orange", label="MSI", **kwargs)
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    plt.xlim(0.0001,ylim)
    #plt.legend();
    plt.savefig('NN Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    
     
    ##############################################3333   
    
    ################  GP #############################33
#    plt.figure()
#    fig, ax = plt.subplots()
#    fig.subplots_adjust(top=0.85)
#    xdata =  (OLI_Data)
#    #ydata =  GP_Data
#    
#    
#    indy= np.where(ydata == np.amax(ydata))
#    ydata= np.delete(ydata, indy)
#    xdata= np.delete(xdata, indy)
#    
#    indx= np.where(xdata == np.amax(xdata))
#    xdata= np.delete(xdata, indx)
#    ydata= np.delete(ydata, indx)
#    
#    xdata=np.reshape(xdata, [len(xdata),1])
#    ydata=np.reshape(ydata, [len(ydata),1])
#    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
#    plt.scatter(xdata, ydata,  marker='o', s=150, c= ydata, cmap='Spectral')
#    #plt.colorbar()
#    #    a, b = best_fit(xdata, ydata)
#    #    yfit = [a + b * xi for xi in xdata]  
#    #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
#    line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
#    transform = ax.transAxes
#    line.set_transform(transform)
#    ax.add_line(line)
#    #ax.plot(xdata, yfit, label='Fitted Line',linewidth=1.0)
#    #ax.set_xlabel('OLI_L8',fontsize=18)
#    #ax.set_ylabel('OLI_L8' + '*' + ' '+ '[R$rs$_443 nm]', fontsize=18)
#    ax.set_ylabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)
#    ax.set_xlabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI_L8',fontsize=30)
#    #ax.plot([],[],' ',color = "blue", label='OCN')
#    ax.set_title('GPR Bandpass Adjustment',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
#    ax.set_ylim(xlim, ylim)
#    ax.set_xlim(xlim, ylim) 
#    #ax.set_xticks(xlim, ylim,0.04)
#    
#    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
#    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
#    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
#    
#    for axis in ['top','bottom','left','right']:
#      ax.spines[axis].set_linewidth(4)
#    plt.legend(fontsize=24)
#    
#    plt.savefig('GPR_Conversion-MSI-OLI-8' + 'Band' + str(MSI_Band[0:7]), bbox_inches = 'tight', dpi=600)
#    #plt.show()
#    
#    
#    MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
#    MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
#    rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
#    rmse=mean_squared_error(ydata, xdata, squared=False)
#    BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
#    MD= np.median(xdata - ydata)
#    z=np.median((np.log(ydata/xdata)))
#    z= 100 * np.sign(z) * (10**abs(z) -1)
#    
#    y=np.median(abs((np.log(ydata/xdata))))
#    y=100 * (10**y -1)
#    
#    r2= r2_score(xdata,ydata)
#    
#    corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
#    r= corr1 * corr1
#    slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))
#    
#    #print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
#    f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
#    print('Results for Band' + MSI_Band[0:7], '\n', file=f)
#    print('Results after GP Band Conversion:','\n', file=f)
#    print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
#    print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)  
#      
#      
      

    
##################################################################################3    
    xdata= OLI_Data
    ydata= Linear_P
    
    MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
    MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
    rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
    rmse=mean_squared_error(ydata, xdata, squared=False)
    BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
    MD= np.median(xdata - ydata)
    z=np.median((np.log(ydata/xdata)))
    z= 100 * np.sign(z) * (10**abs(z) -1)
    
    y=np.median(abs((np.log(ydata/xdata))))
    y=100 * (10**y -1)
    
    r2= r2_score(xdata,ydata)
    
    corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
    r= corr1 * corr1
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(xdata,-1),np.reshape(ydata,-1))
    
    print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
    f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
    print('Results for Band' + MSI_Band[0:7], '\n', file=f)
    print('Results after Liner Band Conversion:','\n', file=f)
    print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
    print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)
      
      
    
    plt.figure()
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.85)
    xdata =  (OLI_Data)
    ydata =  (Linear_P)
    indy= np.where(ydata == np.amax(ydata))
    ydata= np.delete(ydata, indy)
    xdata= np.delete(xdata, indy)
    
    indx= np.where(xdata == np.amax(xdata))
    xdata= np.delete(xdata, indx)
    ydata= np.delete(ydata, indx)
    xdata=np.reshape(xdata, [len(xdata),1])
    ydata=np.reshape(ydata, [len(ydata),1])
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
    #plt.colorbar()
    #    a, b = best_fit(xdata, ydata)
    #    yfit = [a + b * xi for xi in xdata]  
        #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
    line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    #ax.plot(xdata, yfit, label='Fitted Line',linewidth=1.0)
#    ax.set_xlabel('OLI_L8',fontsize=18)
#    ax.set_ylabel('OLI_L8' + '*',fontsize=18)
    #ax.set_ylabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'MSI_S2' + '*',fontsize=30)
    #ax.set_xlabel('$R_{rs} (sr^{-1})$' +  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    
    ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
    ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)
    
    
    #ax.plot([],[],' ',color = "blue", label='OCN')
    ax.set_title('OLS Bandpass Adjustment',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
    ax.set_ylim(xlim, ylim)
    ax.set_xlim(xlim, ylim) 
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    for axis in ['top','bottom','left','right']:
     ax.spines[axis].set_linewidth(4)
    plt.legend(fontsize=24)
    plt.savefig('Linear Conversion-MSI-' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    plt.show()
    
    import seaborn as sns
#    plt.figure()
#    fig, ax = plt.subplots()
#    kwargs = dict(alpha=0.5, bins=10)
#    sns.distplot(xdata, **kwargs, color='orange', label='OLI')
#    plt.hist(ydata, **kwargs, color='dodgerblue', label='MSI')
#    #plt.gca().set(title='Frequency Histogram of OLI-MSI'+'*')
#    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
#    plt.xlim(0.0001,ylim)
#    plt.savefig('Linear Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    
    plt.figure()
    plt.figure(figsize=(10,7), dpi= 80)
    fig, ax = plt.subplots()
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':4})
    sns.distplot(xdata, color="dodgerblue", label="OLI", **kwargs)
    sns.distplot(ydata, color="orange", label="MSI", **kwargs)
    ax.tick_params(axis='both', which='major', width=10, labelsize=30)
    plt.xlim(0.0001,ylim)
    #plt.legend();
    plt.savefig('Linear Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)

plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.85)
xdata =  AOD_OLI_L8
ydata=   AOD_MSI_S2
indy= np.where(ydata == np.amax(ydata))
ydata= np.delete(ydata, indy)
xdata= np.delete(xdata, indy)

indx= np.where(xdata == np.amax(xdata))
xdata= np.delete(xdata, indx)
ydata= np.delete(ydata, indx)
xdata=np.reshape(xdata, [len(xdata),1])
ydata=np.reshape(ydata, [len(ydata),1])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
#plt.colorbar()
#    a, b = best_fit(xdata, ydata)
#    yfit = [a + b * xi for xi in xdata]  
    #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.plot(xdata, yfit, label='Fitted Line',linewidth=1.0)
#    ax.set_xlabel('OLI_L8',fontsize=18)
#    ax.set_ylabel('OLI_L8' + '*',fontsize=18)
ax.set_ylabel('AOD' + '-' + 'MSI',fontsize=30)
ax.set_xlabel('AOD' + '-'+ 'OLI',fontsize=30)
ax.set_title('OLI-MSI AOD intercomparison',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
ax.set_ylim(0.001, 0.6)
ax.set_xlim(0.001, 0.6) 
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
for axis in ['top','bottom','left','right']:
 ax.spines[axis].set_linewidth(4)
plt.legend(fontsize=24)
plt.savefig('AOD Plot', bbox_inches = 'tight', dpi=600)
plt.show()


plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.85)
xdata =  CH_OLI_L8
ydata=   CH_MSI_S2
indy= np.where(ydata == np.amax(ydata))
ydata= np.delete(ydata, indy)
xdata= np.delete(xdata, indy)

indx= np.where(xdata == np.amax(xdata))
xdata= np.delete(xdata, indx)
ydata= np.delete(ydata, indx)
xdata=np.reshape(xdata, [len(xdata),1])
ydata=np.reshape(ydata, [len(ydata),1])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
#plt.colorbar()
#    a, b = best_fit(xdata, ydata)
#    yfit = [a + b * xi for xi in xdata]  
    #ax.plot(xdata,xdata, color="black", label='1:1 Line',linewidth=4.0)
line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.plot(xdata, yfit, label='Fitted Line',linewidth=1.0)
#    ax.set_xlabel('OLI_L8',fontsize=18)
#    ax.set_ylabel('OLI_L8' + '*',fontsize=18)
ax.set_ylabel('Chl-a $[mg^{-3}]$' + '-' + 'MSI',fontsize=30)
ax.set_xlabel('Chl-a $[mg^{-3}]$' + '-'+ 'OLI',fontsize=30)
#ax.plot([],[],' ',color = "blue", label='OCN')
ax.set_title('OLI-MSI Chl-a intercomparison',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
ax.set_ylim(-0.1, 10)
ax.set_xlim(-0.1, 10) 
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
for axis in ['top','bottom','left','right']:
 ax.spines[axis].set_linewidth(4)
plt.legend(fontsize=24)
plt.savefig('CHl-a Plot', bbox_inches = 'tight', dpi=600)
plt.show()
    
    
