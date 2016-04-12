"""Beat-synchronous chroma feature calculation with LabROSA.
   Dan Ellis dpwe@ee.columbia.edu 2016-04-08
"""
from __future__ import print_function

import cPickle as pickle
import getopt
import os
import sys
import time

import numpy as np
import scipy
import sklearn.mixture

import librosa


def read_iso_label_file(filename):
    """Read in an isophonics-format chord label file."""
    times = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            start_secs = float(fields[0])
            end_secs = float(fields[1])
            times.append((start_secs, end_secs))
            labels.append(fields[2])
    return np.array(times), labels


def calculate_overlap_durations(ranges_a, ranges_b):
    """Calculate duration of overlaps between all (start, end) intervals."""
    max_starts_matrix = np.maximum.outer(ranges_a[:, 0], ranges_b[:, 0])
    min_ends_matrix = np.minimum.outer(ranges_a[:, 1], ranges_b[:, 1])
    overlap_durations = np.maximum(0, min_ends_matrix - max_starts_matrix)
    return overlap_durations


def sample_label_sequence(sample_ranges, label_ranges, labels):
    """Find the most-overlapping label for a list of (start, end) intervals."""
    overlaps = calculate_overlap_durations(sample_ranges, label_ranges)
    best_label = np.argmax(overlaps, axis=1)
    return [labels[i] for i in best_label]


def chord_name_to_index(labels):
    """Convert chord name strings into model indices (0..25)."""
    indices = np.zeros(len(labels), dtype=int)
    root_degrees = {'C': 0, 'D': 2, 'E': 4, 'F':5, 'G': 7, 'A':9, 'B': 11}
    for label_index, label in enumerate(labels):
        if label == 'N' or label == 'X':
            # Leave at zero.
            continue
        root_degree = root_degrees[label[0].upper()]
        minor = False
        if len(label) > 1:
            if label[1] == '#':
                root_degree = (root_degree + 1) % 12
            if label[1] == 'b':
                root_degree = (root_degree - 1) % 12
            if ':' in label:
                modifier = label[label.index(':') + 1:]
                if modifier[:3] == 'min':
                    minor = True
        indices[label_index] = 1 + root_degree + 12 * minor
    return indices


def calculate_beat_sync_chroma_of_file(wavfilename):
    """Read the audio, calculate beat-sync chroma."""
    y, sr = librosa.load(wavfilename, sr=None)
    hop_length = 128  # 8 ms at 16 kHz
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, 
                                                 hop_length=hop_length, 
                                                 start_bpm=240)
    # Append a final beat time one beat beyond the end.
    extended_beat_frames = np.hstack([beat_frames, 
                                      2*beat_frames[-1] - beat_frames[-2]])
    frame_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # Drop the first beat_chroma which is stuff before the first beat, 
    # and the final beat_chroma which is everything after the last beat time.
    beat_chroma = librosa.feature.sync(frame_chroma, 
                                       extended_beat_frames).transpose()
    # Drop first row if the beat_frames start after the beginning.
    if beat_frames[0] > 0:
        beat_chroma = beat_chroma[1:]
    # Keep only as many frames as beat times.
    beat_chroma = beat_chroma[:len(beat_frames)]
    assert beat_chroma.shape[0] == beat_frames.shape[0]
    frame_rate = sr / float(hop_length)
    beat_times = beat_frames / frame_rate
    return beat_times, beat_chroma


def calculate_label_indices(labfilename, beat_times):
    """Read a label file, sample at beat times, return 0..25 indices."""
    # MP3s encoded with lame have a 68 ms delay
    LAME_DELAY_SECONDS = 0.068
    extended_beat_times = (np.hstack([beat_times, 
                                      2*beat_times[-1] - beat_times[-2]]) -
                           LAME_DELAY_SECONDS)
    beat_ranges = np.hstack([extended_beat_times[:-1, np.newaxis], 
                             extended_beat_times[1:, np.newaxis]])
    label_time_ranges, labels = read_iso_label_file(labfilename)
    beat_labels = sample_label_sequence(beat_ranges, label_time_ranges, labels)
    label_indices = chord_name_to_index(beat_labels)
    return label_indices


def write_beat_chroma_labels(filename, beat_times, chroma_features, 
                             label_indices):
    """Write out the computed beat-synchronous chroma data."""
    # Create the enclosing directory if needed.
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w") as f:
        pickle.dump((beat_times, chroma_features, label_indices), 
                    f, pickle.HIGHEST_PROTOCOL)


def read_beat_chroma_labels(filename):
    """Read back a precomputed beat-synchronous chroma record."""
    with open(filename, "r") as f:
        beat_times, chroma_features, label_indices = pickle.load(f)
    return beat_times, chroma_features, label_indices


def read_file_list(filename):
    """Read a text file with one item per line."""
    items = []
    with open(filename, 'r') as f:
        for line in f:
            items.append(line.strip())
    return items


def process_items(input_list_file, wav_base_dir, lab_base_dir, output_base_dir, 
                  start_index, num_to_process):
    """Process files from a list."""
    all_ids = read_file_list(input_list_file)
    print("total ids in list:", len(all_ids))

    if num_to_process > 0:
        ids_to_process = all_ids[start_index : start_index + num_to_process]
    else:
        ids_to_process = all_ids[start_index:]

    for number, file_id in enumerate(ids_to_process):
        print(time.ctime(), "File {:d} of {:d}: {:s}".format(
            number, len(ids_to_process), file_id))
        wavfilename = os.path.join(wav_base_dir, file_id + '.mp3')
        beat_times, beat_chroma = calculate_beat_sync_chroma_of_file(
            wavfilename)
        if lab_base_dir:
            labfilename = os.path.join(lab_base_dir, file_id + '.txt')
            label_indices = calculate_label_indices(labfilename, beat_times)
        else:
            label_indices = None
        beatchromlab_filename = os.path.join(output_base_dir, file_id + '.pkl')
        write_beat_chroma_labels(beatchromlab_filename, beat_times, 
                                 beat_chroma, label_indices)


#DATA_DIR = '/q/porkpie/porkpie-p9/hog-restored/hog-p9/drspeech/data/music/'

HELP_STRING = '-i <inputlistfile> -o <outputbasedir> -w <wavbasedir> -l <labbasedir> -s <startindex> -n <numtoprocess>'


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:s:n:w:l:",
                                   ["inputlistfile=", "outputbasedir=",
                                    "startindex=", "numtoprocess=",
                                    "wavbasedir=", "labbasedir="])
    except getopt.GetoptError:
        print(argv[0], HELP_STRING)
        sys.exit(2)
    input_list_file = 'mp3s-mp3s.txt'
    output_base_dir = 'beatchromftrs'
    wav_base_dir = 'mp3s-32k'
    lab_base_dir = None
    start_index = 0
    num_to_process = -1
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0], HELP_STRING)
            sys.exit()
        elif opt in ("-i", "--inputlistfile"):
            input_list_file = arg
        elif opt in ("-o", "--outputbasedir"):
            output_base_dir = arg
        elif opt in ("-s", "--startindex"):
            start_index = int(arg)
        elif opt in ("-n", "--numtoprocess"):
            num_to_process = int(arg)
        elif opt in ("-w", "--wavbasedir"):
            wav_base_dir = arg
        elif opt in ("-l", "--labbasedir"):
            lab_base_dir = arg
    process_items(input_list_file, wav_base_dir, lab_base_dir, 
                  output_base_dir, start_index, num_to_process)


if __name__ == "__main__":
   main(sys.argv)
