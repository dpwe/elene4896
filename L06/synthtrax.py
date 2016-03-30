"""Resynthesis of signals described as sinusoid tracks."""

import numpy as np


def synthtrax(F, M, SR, SUBF=128, DUR=0):
  """
  % X = synthtrax(F, M, SR, SUBF, DUR)     Reconstruct a sound from track rep'n.
  %	Each row of F and M contains a series of frequency and magnitude 
  %	samples for a particular track.  These will be remodulated and 
  %	overlaid into the output sound X which will run at sample rate SR, 
  %	although the columns in F and M are subsampled from that rate by 
  %	a factor SUBF (default 128).  If DUR is nonzero, X will be padded or
  %	truncated to correspond to just this much time.
  % dpwe@icsi.berkeley.edu 1994aug20, 1996aug22
  """
  rows, cols = F.shape

  opsamps = int(np.round(DUR * SR))
  if not DUR:
    opsamps = cols * SUBF

  X = np.zeros(opsamps)

  for row in xrange(rows):
    mm = M[row]
    ff = F[row]
    # First, find onsets - points where mm goes from zero (or NaN) to nzero
    # Before that, even, set all nan values of mm to zero
    nzv = np.nonzero(mm)[0]
    firstcol = np.min(nzv)
    lastcol = np.max(nzv)
    # for speed, chop off regions of initial and final zero magnitude - 
    # but want to include one zero from each end if they are there 
    zz = np.arange(np.maximum(0, firstcol-1), np.minimum(cols, lastcol+1))
    nzcols = zz.shape[0]
    if nzcols > 0:
      mm = mm[zz]
      ff = ff[zz]
      mz = mm == 0
      # Copy frequency values to one point past each end of nonzero stretches.
      onsets = np.nonzero(np.logical_and(mz > 0, np.hstack(
        [1, mz[:-1]]) == 0))[0]
      ff[onsets - 1] = ff[onsets]
      offsets = np.nonzero(np.logical_and(mz[:-1] > 0, mz[1:] == 0))[0]
      ff[offsets + 1] = ff[offsets]
      # Do interpolation.
      ff = np.interp(np.arange(ff.shape[0] * SUBF)/float(SUBF), 
                     np.arange(ff.shape[0]), ff)
      mm = np.interp(np.arange(mm.shape[0] * SUBF)/float(SUBF), 
                     np.arange(mm.shape[0]), mm)
      # Convert frequency to phase values.
      pp = np.cumsum(2*np.pi*ff/SR)
      # Run the oscillator and apply the magnitude envelope.
      xx = mm * np.cos(pp)
      # Add it in to the correct place in the array.
      X[SUBF * zz[0] + np.arange(xx.shape[0])] += xx
  return X


def spearread(FN):
  """
  % [F,M,T] = spearread(FN)
  %    Read in a sinusoidal analysis file written by Michael
  %    Klingbeil's SPEAR program, into Frequency and Magnitude
  %    matrices suitable for synthtrax.m.  T is the actual time 
  %    values for each column.
  % 2010-02-14 Dan Ellis dpwe@ee.columbia.edu
  """
  # Files begin:
  #par-text-frame-format
  #point-type index frequency amplitude
  #partials-count 32
  #frame-count 549
  #frame-data
  #0.124943 1 0 430.064423 0.001209
  #0.134943 1 0 429.900024 0.002103
  #0.144943 5 0 430.215668 0.003097 4 855.366638 0.002075 3 1742.146851 0.002967 2 2165.423096 0.001978 1 2565.337402 0.001767
  #0.154943 9 0 431.365143 0.004033 4 865.541565 0.003474 8 1298.919067 0.001814 3 1743.450806 0.00

  # Each line is: time nharmonics indx0 freq0 amp0 indx1 freq1 amp1 ...
  # indx values serve to connect tracks between frames.

  with open(FN, "r") as f:
    s = f.next().strip()
    if s != 'par-text-frame-format':
      raise ValueError(FN + ' does not look like SPEAR harmonics file')
    s = f.next().strip()
    if s != 'point-type index frequency amplitude':
      raise ValueError('Did not see point-type ... in ' + FN)
    s = f.next().strip()
    if s.split(' ')[0] != 'partials-count':
      raise ValueError('Missing partials-count in ' + FN)
    partials_count = int(s.split(' ')[1])
    s = f.next().strip()
    if s.split(' ')[0] != 'frame-count':
      raise ValueError('Missing frame-count in ' + FN)
    frame_count = int(s.split(' ')[1])
    s = f.next().strip()
    if s != 'frame-data':
      raise ValueError('Missing frame-data in ' + FN)

    T = np.zeros(frame_count)
    F = np.zeros((partials_count, frame_count))
    M = np.zeros((partials_count, frame_count))
    frame = 0
    for s in f:
      vals = [float(v) for v in s.split(' ')]
      T[frame] = vals[0]
      partials_this_frame = int(vals[1])
      field_index = 2
      for _ in xrange(partials_this_frame):
        partial_index = int(vals[field_index])
        F[partial_index, frame] = vals[field_index + 1]
        M[partial_index, frame] = vals[field_index + 2]
        field_index += 3
      frame += 1

    return F, M, T
