# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 08:31:18 2024

@author: Aya

this function take as an input the csv file that contain all identified calls in field recording
and save these calls to be examined. 
it also return a week plot with how many calls per night, and a pivot table that summerize 
how many calls were identify from each species.

need to change- lines 113-125 and maybe 75 with the date
 
"""

import librosa
import soundfile as sf
import pandas as pd
import os


def save_calls (df, output , files_path) :

    # give a name for each run to get a different folder so that the saved calls will not overlap -     
    run_name = 'test'
    
    # create one more column for later analysis - 
    df['start_seg1'] = df['start_seg'].astype('Int64').astype('str')
    df['file_name'] = df['recorder_id'].str[0:-4]+'_'+ df['start_seg1']
    df['species_1'] = df['species_1'].str.lower()
    df['species_2'] = df['species_2'].str.lower()

    
    #creat two data frams - for species 1 rank and species 2 rank
    # that present only bird identifications with higher probability then 0.1 - 
    df_species_1 = df[['recorder_id' , 'species_1' , 'probability_1' , 'file_name']].copy() 
    df_1 = df_species_1.drop(df_species_1[df_species_1.probability_1 < 0.1].index)

    non_event = ['nothing' , 'anthropogenic' , 'anuran' , 'jackal' , 'cricket' , 'background_birds' , 'background_cettis_warbler'
                 ,'background_eurasian_thick-knee' , 'background_spur-winged_lapwing'] # 'background_birds', 'cricket' , 'grey_heron_ardea_cinerea'
    # non_event = []
    df_1_cleen = df_1[~df_1['species_1'].isin(non_event)]    

    # save all calls that are in species 2 and have more then 0.1 score and their first identification is nothing - 

    df_species_noise = df[['recorder_id' , 'species_1' , 'species_2' , 'probability_2' , 'file_name']].copy()
    df_noise = df_species_noise.drop(df_species_noise[df_species_noise.probability_2 < 0.1].index)

    df_noise_cleen =  df_noise[df_noise['species_1'].isin(non_event)]
    df_noise_cleen =  df_noise_cleen[~df_noise_cleen['species_2'].isin(non_event)]
    
    df_cleen = pd.concat([df_1_cleen, df_noise_cleen], ignore_index=True)  
    
    # sort by species name - 
    #--------------------------------------------
    df_for_validation = df_cleen.sort_values(by=['species_1'])
    
    #creat a table -pivot like - with numbers of identification and average probability - 

    df_pivot = df_1.groupby(['species_1']).agg({'species_1':'count','probability_1':'mean'})
    df_pivot_cleen = df_noise_cleen.groupby(['species_2']).agg({'species_2':'count','probability_2':'mean'})
    result = pd.concat([df_pivot, df_pivot_cleen[:]], axis=1)
    print(result.iloc[:, :3])



    df_species_2 = df[['recorder_id' , 'species_2' , 'probability_2' , 'file_name']].copy()
    df_2 = df_species_2.drop(df_species_2[df_species_2.probability_2 < 0.1].index)

    df_2_cleen =  df_2[~df_2['species_2'].isin(non_event)]
    
    #creat a table -pivot like - with numbers of identification and average probability - 

    df_pivot2 = df_2.groupby(['species_2']).agg({'species_2':'count','probability_2':'mean'})
    # result = pd.concat([df_pivot, df_pivot2[:]], axis=1)
    # print(result)
    
    #save files by file name and rank (for 2 ranks)-
    #---------------------------------------------------------
    rank_id = [df_1_cleen , df_noise_cleen]
    x = 0
    for rank in rank_id:        
        ind = rank.index
        x += 1
        for row_index in ind:
            row = rank.loc[row_index]
            file_name = row['recorder_id'] 
            species = row['species_'+str(x)]
    
            output_path = output + run_name +'/'+str(species)
        
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            orig_row = df.loc[row_index]
            
            input_file = files_path +'/'+ file_name
            y, sr = librosa.load(input_file, sr=None)
            start_sample = orig_row['start_seg'] * sr
            end_sample = orig_row['end_seg'] * sr
            slice_y = y[start_sample:end_sample]
            
            output_file = output_path +'/'+ file_name[:-4] +'_'+ str(orig_row['start_seg'])+'_'+species+'.wav'
            sf.write(output_file, slice_y, sr)
            
        #save files by file name and rank (only for rank 1)- 
#--------------------------------------------------------------       
    # ind = df_1_cleen.index
    # for row_index in ind:
    #         row = df_1_cleen.loc[row_index]
    #         file_name = row['recorder_id'] 
    #         species = row['species_1']
    
    #         output_path = output +'/run2'+'/'+str(species)
    #         output_path = output +'/'+str(species)

        
    #         if not os.path.exists(output_path):
    #             os.makedirs(output_path)
            
    #         orig_row = df.loc[row_index]
            
    #         input_file = files_path +'/'+ file_name
    #         y, sr = librosa.load(input_file, sr=None)
    #         start_sample = int(orig_row['start_seg'] * sr)
    #         end_sample = int(orig_row['end_seg'] * sr)
    #         slice_y = y[start_sample:end_sample]
            
    #         output_file = output_path +'/'+ file_name[:-4] +'_'+ str(orig_row['start_seg'])+'_'+species+'.wav'
    #         sf.write(output_file, slice_y, sr)
    
      
    
    return df_1_cleen , result , df_for_validation #, df_2_cleen
    
    

files_path = r'../data/example_recordings'
input_path = r'../output/'
df_identified_calls = 'costum_NFC_model_on_data.csv'
output_path = r'../output/saved_identified_calls/'

#### save identified calls -
df = pd.read_csv(input_path + df_identified_calls)
df_species_1 , pivot , df_for_validation = save_calls(df , output_path , files_path )


#### save calls for rank 1 and 2
# df_species_1, df_species_2 , pivot , df_for_validation = save_calls(df , output_path , files_path )

# save dataframe that is arranged by species - 
#--------------------------------------------
df_for_validation.to_csv('../output/' +df_identified_calls[:-4]+'_for_validation.csv', index=False, quoting=1)

# present_calls_per_night(df_species_1 , date , days , per = 'day')       

# if you want to save the pivot table as csv - 
# df_pivot = pivot.reset_index()
# df_pivot.to_csv('Output_BirdNet/'+df_identified_calls[:-12]+'_pivot.csv', index=False, quoting=1)
