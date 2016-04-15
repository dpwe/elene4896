"""Match subregions within beat-chroma matrices.
   2016-04-09 Dan Ellis dpwe@ee.columbia.edu
"""

"""
Plan:
 - read in beat-chroma matrix
 - break into 32 beat segments every ?8 beats
 - take 2DFTM
 - PCA down to ? 8 dimensions
 - build (8752*100), 8 matrix = 28 MB of float32

 - find closest match to query
"""

import os
import numpy as np
import cPickle as pickle
import numpy.lib.stride_tricks
import sklearn.decomposition

DATA_DIR = '/Users/dpwe/Downloads/prac10/data'
CHROMA_BASE_DIR = os.path.join(DATA_DIR, 'beatchromlabs')


def read_beat_chroma_labels(filename):
    """Read back a precomputed beat-synchronous chroma record."""
    with open(filename, "rb") as f:
        beat_times, chroma_features, label_indices = pickle.load(f)
    return beat_times, chroma_features, label_indices


def read_beat_chroma_labels_for_id(id_):
  chroma_filename = os.path.join(CHROMA_BASE_DIR, id_ + ".pkl")
  return read_beat_chroma_labels(chroma_filename)


def frame_array(data, frame_length=48, frame_hop=8):
  """Return multiple overlapping submatrices from data."""
  item_bytes = data.itemsize
  num_vectors, num_dimensions = data.shape
  frame_starts = np.arange(0, num_vectors - frame_length, frame_hop)
  num_frames = len(frame_starts)
  data_frames = np.lib.stride_tricks.as_strided(
    data, strides=(frame_hop * num_dimensions * item_bytes, 
                   num_dimensions * item_bytes, item_bytes), 
    shape=(num_frames, frame_length, num_dimensions))
  return data_frames, frame_starts
  

def construct_pca_object(ids, num_pcas=20, frame_length=48, frame_hop=8):
  """Build a PCA transform based on random draws from a subset of tracks."""
  all_features, ids, starts = build_incipit_array(
    ids, pca=None, frame_length=frame_length, frame_hop=frame_hop)
  pca = sklearn.decomposition.PCA(n_components=num_pcas, whiten=True, copy=True)
  pca.fit(all_features)
  return pca


def build_incipit_array(ids_to_process, pca=None, frame_length=48, frame_hop=8):
  """Build a single matrix of stacked 2DFTM projections for frames of IDs."""
  # all_features is a single np.array of (total_num_frames, pca_dimensions).
  # all_ids[i] gives the file id from which row i of all features derives.
  # all_starts[i] gives the beat offset within that id for the frame.
  all_features = []
  all_ids = []
  all_starts = []
  for id_ in ids_to_process:
    _, chroma, _ = read_beat_chroma_labels_for_id(id_)
    chroma_frames, frame_starts = frame_array(chroma, frame_length, frame_hop)
    stftm_frames = np.abs(np.fft.fft2(chroma_frames))
    flattened_frames = np.reshape(
      stftm_frames, (stftm_frames.shape[0], 
                     stftm_frames.shape[1] * stftm_frames.shape[2]))
    if pca is not None:
      flattened_frames = pca.transform(flattened_frames)
    num_frames = flattened_frames.shape[0]
    all_features.append(flattened_frames)
    all_ids.extend([id_] * num_frames)
    all_starts.extend(frame_starts)
  return np.concatenate(all_features), all_ids, all_starts


def read_list_file(filename):
    """Read a text file with one item per line."""
    items = []
    with open(filename, 'r') as f:
        for line in f:
            items.append(line.strip())
    return items


id_list_file = os.path.join(DATA_DIR, 'trainfilelist.txt')

all_ids = read_list_file(id_list_file)

#pca_ids = all_ids[0:-1:10]
pca_ids = all_ids
frame_length = 16
frame_hop = 4

pca_object = construct_pca_object(pca_ids, num_pcas=16, 
                                  frame_length=frame_length, 
                                  frame_hop=frame_hop)

all_features, all_ids, all_starts = build_incipit_array(
  all_ids, pca_object, frame_length=frame_length, frame_hop=frame_hop)

print("frame_length=", frame_length, "frame_hop=", frame_hop, 
      "all_features.shape=", all_features.shape)
