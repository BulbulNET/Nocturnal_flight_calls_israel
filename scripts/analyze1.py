import os
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

import sys
import json
import operator
import argparse
import datetime
import traceback

import numpy as np

import config as cfg
import audio
import model

def clearErrorLog():
    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)


def writeErrorLog(msg):
    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')


def parseInputFiles(path, allowed_filetypes=['wav', 'flac', 'mp3', 'ogg', 'm4a']):
    # Add backslash to path if not present
    if not path.endswith(os.sep):
        path += os.sep

    # Get all files in directory with os.walk
    files = []
    for root, dirs, flist in os.walk(path):
        for f in flist:
            if len(f.rsplit('.', 1)) > 1 and f.rsplit('.', 1)[1].lower() in allowed_filetypes:
                files.append(os.path.join(root, f))

    print('Found {} files to analyze'.format(len(files)))

    return sorted(files)


def loadCodes():
    # with open(cfg.CODES_FILE, 'r') as cfile:
    with open(cfg.CODES_FILE, 'r') as cfile:
        codes = json.load(cfile)

    return codes


def loadLabels(labels_file):
    labels = []
    # with open(labels_file, 'r') as lfile:
    with open(labels_file, 'r') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\n', ''))

    return labels


def loadSpeciesList(fpath):
    slist = []
    if not fpath == None:
        with open(fpath, 'r') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                slist.append(species)

    return slist


def predictSpeciesList():
    l_filter = model.explore(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK)
    cfg.SPECIES_LIST_FILE = None
    cfg.SPECIES_LIST = []
    for s in l_filter:
        if s[0] >= cfg.LOCATION_FILTER_THRESHOLD:
            cfg.SPECIES_LIST.append(s[1])


def saveResultFile(r, path, afile_path):
    # Make folder if it doesn't exist
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # Selection table
    out_string = ''

    if cfg.RESULT_TYPE == 'table':

        # Raven selection header
        header = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies Code\tCommon Name\tConfidence\n'
        selection_id = 0

        # Write header
        out_string += header

        # Extract valid predictions for every timestamp
        for timestamp in getSortedTimestamps(r):
            rstring = ''
            start, end = timestamp.split('-')
            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                    selection_id += 1
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += '{}\tSpectrogram 1\t1\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\n'.format(
                        selection_id,
                        start,
                        end,
                        150,
                        12000,
                        cfg.CODES[c[0]],
                        label.split('_')[1],
                        c[1])

            # Write result string to file
            if len(rstring) > 0:
                out_string += rstring

    elif cfg.RESULT_TYPE == 'audacity':

        # Audacity timeline labels
        for timestamp in getSortedTimestamps(r):
            rstring = ''
            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += '{}\t{}\t{:.4f}\n'.format(
                        timestamp.replace('-', '\t'),
                        label.replace('_', ', '),
                        c[1])

            # Write result string to file
            if len(rstring) > 0:
                out_string += rstring

    elif cfg.RESULT_TYPE == 'r':

        # Output format for R
        header = 'filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model'
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ''
            start, end = timestamp.split('-')
            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += '\n{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}'.format(
                        afile_path,
                        start,
                        end,
                        label.split('_')[0],
                        label.split('_')[1],
                        c[1],
                        cfg.LATITUDE,
                        cfg.LONGITUDE,
                        cfg.WEEK,
                        cfg.SIG_OVERLAP,
                        (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
                        cfg.MIN_CONFIDENCE,
                        cfg.SPECIES_LIST_FILE,
                        os.path.basename(cfg.MODEL_PATH)
                    )
            # Write result string to file
            if len(rstring) > 0:
                out_string += rstring

    else:

        # CSV output file
        header = 'Start (s),End (s),Scientific name,Common name,Confidence\n'

        # Write header
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ''
            for c in r[timestamp]:
                start, end = timestamp.split('-')
                if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += '{},{},{},{},{:.4f}\n'.format(
                        start,
                        end,
                        label.split('_')[0],
                        label.split('_')[1],
                        c[1])

            # Write result string to file
            if len(rstring) > 0:
                out_string += rstring

    # Save as file
    with open(path, 'w') as rfile:
        rfile.write(out_string)


def getSortedTimestamps(results):
    return sorted(results, key=lambda t: float(t.split('-')[0]))

def keep_top_N(vin, N):
    v = np.copy(vin)
    # number of bottom predictions
    n_bottom = len(v) - N

    # get bottom indices using partition
    bottom_n_ind = np.argpartition(v, n_bottom)[:n_bottom]

    # set bottom prediction to 0
    v[bottom_n_ind] = 0

    return v

def getRawAudioFromFile(fpath):
    # Open file
    sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    return chunks

def getRawAudioFromWavArr(sig, rate):
    # Open file
    # sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    return chunks

def predict(samples):
    """ Agamon id translation """
    # Prepare sample and pass through model
    data = np.array(samples, dtype='float32')
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-cfg.SIGMOID_SENSITIVITY)

    return prediction


def analyze_file(x, fs):

    # split into 3-second chunks
    chunks = getRawAudioFromWavArr(x, fs)

    # If no chunks, show error and skip
    if len(chunks) == 0:
        msg = 'Error: Cannot open audio file'
        print(msg, flush=True)
        # writeErrorLog(msg)
        return False

    # Process each chunk
    try:
        start, end = 0, cfg.SIG_LENGTH
        results = {}
        samples = []
        timestamps = []
        pred = np.zeros(len(cfg.LABELS))
        for c in range(len(chunks)):

            # Add to batch
            samples.append(chunks[c])

            # Check if batch is full or last chunk
            if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                continue

            # Predict
            prob = predict(samples)[0]
            max_pi = np.argmax(prob) #index of maximum probability of prediction

            # save result
            timestamps.append([start, end, cfg.LABELS[max_pi]])

            # Advance start and end
            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH
            # p = keep_top_N(p, 10)     # keep top N values

            pred += prob                   # accumulate current chunks probs

        # average probabilities
        pred /= len(chunks)
        p_labels = dict(zip(cfg.LABELS, pred))

        # Sort by score
        p_sorted = dict(sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True))


    except:
        # Print traceback
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot analyze audio file'
        print(msg, flush=True)
        # writeErrorLog(msg)
        return False

    return timestamps, p_sorted


""" Get absolute path to resource, works for dev and for PyInstaller """
try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
    rel_path = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[-6:-1])

    print('base_path = sys._MEIPASS ', base_path = sys._MEIPASS)
    print('rel_path = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[-6:-1]) ', rel_path = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[-6:-1]))
except Exception:
    base_path = os.path.dirname(os.path.realpath(__file__))
#

# Set paths relative to script path (requested in #3)
cfg.MODEL_PATH = os.path.join(base_path, cfg.MODEL_PATH)
cfg.LABELS_FILE = os.path.join(base_path, cfg.LABELS_FILE)
cfg.TRANSLATED_LABELS_PATH = os.path.join(base_path, cfg.TRANSLATED_LABELS_PATH)
cfg.MDATA_MODEL_PATH = os.path.join(base_path, cfg.MDATA_MODEL_PATH)
cfg.CODES_FILE = os.path.join(base_path, cfg.CODES_FILE)
cfg.ERROR_LOG_FILE = os.path.join(base_path, cfg.ERROR_LOG_FILE)

def birdnet_predict(x, fs, LAT=-1, LON=-1):
    """ Agamon id translation """
    src_path = ()

    if len(x) / fs < cfg.SIG_MINLEN:
        raise Exception("Birdnet: The file is too \nshort")

    # Freeze support for excecutable
    # freeze_support()

    # Clear error log
    # clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default=os.path.abspath('resources/'), help='Path to input file or folder. If this is a file, --o needs to be a file too.')
    parser.add_argument('--o', default='results/', help='Path to output file or folder. If this is a file, --i needs to be a file too.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.')
    parser.add_argument('--slist', default='', help='Path to species list file or folder. If folder is provided, species list needs to be named \"species_list.txt\". If lat and lon are provided, this list will be ignored.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.')
    parser.add_argument('--rtype', default='table', help='Specifies output format. Values in [\'table\', \'audacity\', \'r\', \'csv\']. Defaults to \'table\' (Raven selection table).')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')
    parser.add_argument('--batchsize', type=int, default=1, help='Number of samples to process at the same time. Defaults to 1.')
    parser.add_argument('--locale', default='en', help='Locale for translated species common names. Values in [\'af\', \'de\', \'it\', ...] Defaults to \'en\'.')
    parser.add_argument('--sf_thresh', type=float, default=0.03, help='Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.')

    args = parser.parse_args()

    # Load eBird codes, labels
    cfg.CODES = loadCodes()
    cfg.LABELS = loadLabels(cfg.LABELS_FILE)

    # Load translated labels
    lfile = os.path.join(cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace('.txt', '_{}.txt'.format(args.locale)))
    if not args.locale in ['en'] and os.path.isfile(lfile): # if there exists a file named lfile, and args.locale is not in English
        cfg.TRANSLATED_LABELS = loadLabels(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

        ### Make sure to comment out appropriately if you are not using args. ###

    # Load species list from location filter or provided list
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week

    # TODO: ADDED
    cfg.LATITUDE, cfg.LONGITUDE = LAT, LON


    cfg.LOCATION_FILTER_THRESHOLD = max(0.01, min(0.99, float(args.sf_thresh)))
    if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
        if len(args.slist) == 0:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(base_path)), args.slist)
            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, 'species_list.txt')
        cfg.SPECIES_LIST = loadSpeciesList(cfg.SPECIES_LIST_FILE)
    else:
        predictSpeciesList()

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(args.sensitivity) - 1.0), 1.5))

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set result type
    cfg.RESULT_TYPE = args.rtype.lower()
    if not cfg.RESULT_TYPE in ['table', 'audacity', 'r', 'csv']:
        cfg.RESULT_TYPE = 'table'

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = max(1, int(args.threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(args.threads))

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(args.batchsize))

   # for entry in flist:
    try:
        preds = analyze_file(x, fs)
    except Exception as e:
        print('Birdnet Exception: ', e)
        raise Exception("Birdnet is unable \nto process file.")

    # print(preds)
    return preds

# if __name__ == '__main__':
#     import librosa
#     x, fs = librosa.load('resources/518_11_S4A13411_20210301_180000.wav', sr=48000)
#     r = birdnet_predict(x, fs)
