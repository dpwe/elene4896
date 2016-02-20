"""Linear Predictive Coding analysis and resynthesis for audio."""

import numpy as np
import scipy.signal

def lpcfit(x, p=12, h=128, w=None, overlaps=True):
  """Perform LPC analysis of short-time windows of a waveform.

  Args:
    x: 1D np.array containing input audio waveform.
    p: int, order of LP models to fit.
    h: int, hop in samples between successive short-time windows.
    w: int, analysis window length. Defaults to 2 x h.
    overlaps: bool, if true, residuals are overlap-added between 
      windows (for a continuous excitation), otherwise only the 
      residual for each hop portion is kept (for perfect reconstruction).

  Returns:
    a: np.array of (n_frames, p + 1) containing the LPC filter coefficients for
      each frame.
    g: np.array of (n_frames,) giving the gain for each frame.
    e: np.array of (n_frames * h + (w - h),) giving the normalized-energy
      excitation (residual).
  """
  if not w:
    w = 2 * h
  npts = x.shape[0]
  nhops = int(npts/h)
  # Pad x with zeros so that we can extract complete w-length windows from it.
  x = np.hstack([np.zeros(int((w-h)/2)), x, np.zeros(int(w-h/2))])
  a = np.zeros((nhops, p+1))
  g = np.zeros(nhops)
  if overlaps:
    e = np.zeros((nhops - 1) * h + w)
  else:
    e = np.zeros(npts)
  # Pre-emphasis
  pre = [1, -0.9]
  x = scipy.signal.lfilter(pre, 1 , x)
  for hop in np.arange(nhops):
    # Extract segment of signal.
    xx = x[hop * h + np.arange(w)]
    # Apply hanning window
    wxx = xx * np.hanning(w)
    # Form autocorrelation (calculates *way* too many points)
    rxx = np.correlate(wxx, wxx, 'full')
    # Extract just the points we need (middle p+1 points).
    rxx = rxx[w - 1 + np.arange(p + 1)]
    # Setup the normal equations
    coeffs = np.dot(np.linalg.inv(scipy.linalg.toeplitz(rxx[:-1])), rxx[1:])
    # Calculate residual by filtering windowed xx
    aa = np.hstack([1.0, -coeffs])
    if overlaps:
      rs = scipy.signal.lfilter(aa, 1, wxx)
    else:
      rs = scipy.signal.lfilter(aa, 1, xx[int((w - h) / 2) + np.arange(h)])
    G = np.sqrt(np.mean(rs**2))
    # Save filter, gain and residual
    a[hop] = aa
    g[hop] = G
    if overlaps:
      e[hop * h + np.arange(w)] += rs / G
    else:
      e[hop *h  + np.arange(h)] = rs / G
  # Throw away first (win-hop)/2 pts if in overlap mode
  # for proper synchronization of resynth
  if overlaps:
    e = e[int((w - h) / 2):]
  return a, g, e

def lpcsynth(a, g, e=None, h=128, overlaps=True):
  """Resynthesize a short-time LPC analysis to audio.

  Args:
    a: np.array of (nframes, order + 1) giving the per-frame LPC filter 
      coefficients.
    g: np.array of (nframes,) giving the gain for each frame.
    e: np.array of (nframes * hop + (window - hop)) giving the excitation 
      signal to feed into the filters.  If a scalar, an impulse train with the
      specified period is used.  Defaults to Gaussian white noise.
    h: int, hop between successive reconstruction frames, in samples.  
      Reconstruction window is always 2 * h.
    overlaps: bool.  If true, successive frames are windowed and overlap-
      added.  If false, we assume e contains exact residuals for each 
      window, so reconstructions are similarly truncated and concatenated.

  Returns:
      1D np.array of the resynthesized waveform.
  """
  w = 2 * h
  nhops, p = a.shape
  npts = nhops * h + w
  # Excitation needs extra half-window at the end if in overlap mode
  nepts = npts + overlaps*(w - h)
  if e is None:
      e = np.random.randn(nepts)
  elif type(e) == int:
      period = e;
      e = np.sqrt(period) * (np.mod(np.arange(nepts), period) == 0).astype(float)
  else:
      nepts = e.shape[0]
      npts = nepts + h
  # Try to make sure we don't run out of e (in ov mode)
  e = np.hstack([e, np.zeros(w)])
  d = np.zeros(npts)
  for hop in np.arange(nhops):
      hbase = hop * h
      #print d.shape, hbase, hop, nhops
      oldbit = d[hbase + np.arange(h)]
      aa = a[hop, :]
      G = g[hop]
      if overlaps:
          d[hbase + np.arange(w)] += np.hanning(w) * G * scipy.signal.lfilter([1], aa, e[hbase + np.arange(w)])
      else:
          d[hbase + np.arange(h)] = G * scipy.signal.lfilter(1, aa, e[hbase + np.arange(h)])
  # De-emphasis (must match pre-emphasis in lpcfit)
  pre = [1, -0.9]
  d = scipy.signal.lfilter([1], pre, d)
  return d
