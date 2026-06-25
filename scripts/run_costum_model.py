# -*- coding: utf-8 -*-
"""
run costum model for applying birdNET 2.4 on new dataset
2024_05
"""



import os
import librosa
import pandas as pd
import librosa.display
from operator import itemgetter
import time
import analyze1


def process_bnt_pred(preds_bnt_in, topN_bnt, brdnt_train):
    # remove unwanted labels - labels that doesn't have match in agamon
    # keep only integer keys, because birdnet-agamon conversion is done at birdnet lib level.
    if brdnt_train:
        preds_bnt = preds_bnt_in
    else:
      preds_bnt = {k: v for k, v in preds_bnt_in.items() if isinstance(k, int)}

    # keep only topN_bnt highest results
    n_preds_bnt = topN_bnt if topN_bnt <= len(preds_bnt) else len(preds_bnt)
    preds_bnt = dict(sorted(preds_bnt.items(), key=itemgetter(1), reverse=True)[:n_preds_bnt])

    return preds_bnt


# Function to process and append event to the DataFrame
def append_event_to_dataframe(df, event, metadata):
    # Extract species and probabilities from the event dictionary
    species = list(event.keys())
    probabilities = list(event.values())
    conf_score = 0.1
    # combined_data = {**metadata, **event}
    
    # Create a dictionary for the row
    correct0 = check_pred_name_in_true_label(species[0], metadata['recorder_id']) \
            and probabilities[0] > conf_score
    correct1 = check_pred_name_in_true_label(species[1], metadata['recorder_id']) \
            and probabilities[1] > conf_score
    correct2 = check_pred_name_in_true_label(species[2], metadata['recorder_id']) \
            and probabilities[2] > conf_score  
    correct = correct0 or correct1 or correct2
       
    row = {
        'recorder_id': metadata['recorder_id'],
        'start_seg': metadata['start_seg'],
        'end_seg': metadata['end_seg'],
        'species_1': species[0], 'probability_1': probabilities[0],
        'species_2': species[1], 'probability_2': probabilities[1],
        'species_3': species[2], 'probability_3': probabilities[2]
    }
    
    # Convert the row dictionary to a DataFrame
    row_df = pd.DataFrame([row], columns=columns)
    
    # Append the row DataFrame to the main DataFrame
    df = pd.concat([df, row_df], ignore_index=True)
    return df


def analyze_audio(file_path, output_path, df, folder):
    audio, sr = librosa.load(file_path, sr = 48000)
    duration = librosa.get_duration(y=audio, sr=sr)
    segment_length = 3  # segment length in seconds
    #sr = librosa.get_samplerate(file_path)
    topN_bnt = 3
    for start_time in range(0, int(duration), segment_length):
        segment = audio[start_time * sr:(start_time + segment_length) * sr]
        brdnt_train = True
        ts, preds_bnt_raw = analyze1.birdnet_predict(segment, sr)
        preds_bnt = process_bnt_pred(preds_bnt_raw, topN_bnt, brdnt_train)
        print(preds_bnt)
        #recorder = file_path.split('/')[-1] # was ('\\')
        
        start_seg = start_time
        end_seg = start_time + segment_length
        metadata = {'recorder_id': folder,'start_seg': start_seg, 'end_seg': end_seg}  
        df = append_event_to_dataframe(df, preds_bnt, metadata)
        
        print('--'*27)
        # sd.play(segment, sr)
        # sd.wait()
        
    return df
        
def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Replace spaces and hyphens with underscores
    s = s.replace(' - ', '_').replace('-', '_').replace(' ', '_')
    return s


def check_pred_name_in_true_label(pred_name, filename):
    # Normalize both the substring and the filename
    normalized_pred_name = normalize_string(pred_name)
    normalized_filename = normalize_string(filename)
    
    # Check if the normalized substring is in the normalized filename
    correct = normalized_pred_name in normalized_filename or normalized_filename in normalized_pred_name 
    return correct 


columns = [
    'recorder_id',
    'start_seg', 'end_seg',
    'species_1', 'probability_1', 
    'species_2', 'probability_2', 
    'species_3', 'probability_3']


df = pd.DataFrame(columns=columns)

# selecting the directory path of the recordings needed to be scanned - 
# ------------------------------------------------------------

directory_path = r'../data/example_recordings/'

#calculate running time-
start_time = time.time()

output_path = r"../output"  # Replace with your desired output directory
print('Birdnet loaded from local environment')

if not os.path.exists(output_path):
    os.makedirs(output_path)

## if the folder contain the files- 
## ------------------------------------

selected_files = os.listdir(directory_path)
for file in selected_files:
    # print(file)
    file_path = os.path.join(directory_path + file)
    df = analyze_audio(file_path, output_path, df , file)


file = 'costum_NFC_model_on_data'

df.to_csv(str(output_path) + '/'+ file +'.csv', index=False, quoting=1)

#print running time-
print("--- %s seconds ---" % (time.time() - start_time))
print("created file - - - " + file)  
   
