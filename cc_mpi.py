import pandas as pd
import numpy as np
import obspy
from obspy.signal.invsim import cosine_taper
import scipy
from obspy.signal.regression import linear_regression
from datetime import datetime, timedelta
import sys
import os
import h5py
import math
from mpi4py import MPI


class WaveformAccess:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.f = h5py.File(h5_path, 'r')
        # Only print loading message on Rank 0 to avoid clutter
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Loading metadata index...")
        
        self.meta = pd.DataFrame({
            'event_id': self.f['event_id'][:].astype(str), 
            'network': self.f['network'][:].astype(str),
            'station': self.f['station'][:].astype(str),
            'phasetype':self.f['phasetype'][:].astype(str)
        })
        self.meta.reset_index(inplace=True) 
        self.meta.set_index(['event_id', 'network', 'station','phasetype'], inplace=True)
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Index built. Loaded {len(self.meta)} entries.")

    def get_waveform(self, evid, net, sta, phasetype):
        try:
            target_row = self.meta.loc[(evid, net, sta, phasetype)]
            if isinstance(target_row, pd.DataFrame):
                idx = target_row['index'].iloc[0]
            else:
                idx = target_row['index']
            wf = self.f['waveform'][idx]
            return wf
        except KeyError:
            return None

    def close(self):
        self.f.close()

def getCoherence(dcs, ds1, ds2):
    n = len(dcs)
    coh = np.zeros(n).astype("complex")
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def smooth(x, window="boxcar", half_win=3):
    window_len = 2 * half_win + 1
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2:-window_len-1:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype("complex")
    else:
        w = scipy.signal.windows.hann(window_len).astype("complex")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[half_win : len(y) - half_win]

def mwcs(sig1, sig2, dt):
    freqmin = 1
    freqmax = 20
    # wind_len = len(sig1) # Original commented out
    wind_len = pow(2, int(np.log(len(sig1))/np.log(2)))
    tp = cosine_taper(wind_len, 0.05)
    minidx = 0
    slide = 4
    smoothing_half_win = 3
    delta_t = []
    delta_weight = []
    delta_ccv = []
    
    while (((minidx+wind_len)<=len(sig1)) & ((minidx+wind_len)<=len(sig2))):
        cci = sig1[minidx:minidx+wind_len]
        cri = sig2[minidx:minidx+wind_len]
        
        # Check for essentially zero signals to avoid divide by zero/NaN later
        if np.max(np.abs(cci)) == 0 or np.max(np.abs(cri)) == 0:
            minidx += slide
            continue

        cc1 = np.correlate(cci, cri, 'valid')
        cc2 = np.correlate(cri,cri, 'valid')
        cc3 = np.correlate(cci,cci,'valid')
        
        # Safety for division
        denom = (np.sqrt(cc2*cc3))
        if denom == 0:
            ccv = 0
        else:
            ccv = np.max(np.abs(cc1 / denom))
            # ccv = (cc1/denom)[0]
        delta_ccv.append(ccv)

        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp
        minidx += slide
        
        fcur = scipy.fftpack.fft(cci, n=len(cci))[:wind_len//2]
        fref = scipy.fftpack.fft(cri, n=len(cri))[:wind_len//2]
        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2
        
        X = fref * (fcur.conj())

        dcur = np.sqrt(smooth(fcur2, window="hanning", half_win=smoothing_half_win))
        dref = np.sqrt(smooth(fref2, window="hanning", half_win=smoothing_half_win))
        X = smooth(X, window="hanning", half_win=smoothing_half_win)

        dcs = np.abs(X)
        try:
            freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[: wind_len // 2]
        except:
            continue
            
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin, freq_vec <= freqmax))
        
        # Empty range check
        if len(index_range) == 0:
            continue

        coh = getCoherence(dcs, dref, dcur)
        # mcoh = np.mean(coh[index_range]) # Unused var

        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        v = np.real(freq_vec[index_range]) * 2 * np.pi

        phi = np.angle(X)
        phi[0] = 0.0
        phi = np.unwrap(phi)
        phi = phi[index_range]
        
        try:
            m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
            delta_t.append(m)
            e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
            s2x2 = np.sum(v**2 * w**2)
            sx2 = np.sum(w * v**2)
            e = np.sqrt(e * s2x2 / sx2**2)
            delta_weight.append(1./e)
        except:
            # Handle regression failures
            continue

    delta_t = np.array(delta_t)
    delta_weight = np.array(delta_weight)
    delta_ccv = np.array(delta_ccv)
    
    if len(delta_t) == 0 or np.sum(delta_weight) == 0:
        return 0.0, 0.0

    return np.sum(delta_t*delta_weight/sum(delta_weight)), np.sum(delta_ccv*delta_weight) / np.sum(delta_weight)

def mwcs3(w1, w2, dt):
    ncomp = min(w1.shape[0], w2.shape[0])
    dts, cvvs = [], []
    for k in range(ncomp):
        dtk, cck = mwcs(w1[k,:], w2[k,:], dt)
        dts.append(dtk); cvvs.append(cck)
    if not dts:
        return 0.0, 0.0
    return float(np.mean(dts)), float(np.mean(cvvs))


def process_rows(df_chunk, db, meta_lookup, dt):
    """
    Processes a chunk of the dataframe and returns a list of output strings.
    """
    results = []
    if df_chunk is None or df_chunk.empty:
        return results

    for row in df_chunk.itertuples(index=False):
        for phase in ['P', 'S']:
            tt1 = meta_lookup.get((row.evid1, row.station_code, phase))
            tt2 = meta_lookup.get((row.evid2, row.station_code, phase))
            
            if tt1 is None or tt2 is None:
                continue
                
            wf1 = db.get_waveform(evid=str(int(row.evid1)), net=row.network_code, sta=row.station_code, phasetype=phase)
            if wf1 is None: continue 
            
            wf2 = db.get_waveform(evid=str(int(row.evid2)), net=row.network_code, sta=row.station_code, phasetype=phase)
            if wf2 is None: continue

            # Run Math
            delta_t, cvv = mwcs3(wf1, wf2, dt)

            # Format Output String
            out_str = f"{int(row.event1_index)},{int(row.event2_index)},{row.network_code},{row.station_code},{tt1-tt2+delta_t:6.4f},{cvv:4.2f},{phase}\n"
            results.append(out_str)
            
    return results


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    file_path = "/scratch/users/yuyifan/final_catalog_dataset2.h5"
    sampling_rate = 100
    dt = 1/sampling_rate

    db = WaveformAccess(file_path)

    if rank == 0:
        print("Building metadata lookup...")
        metadata = pd.read_csv("./genie/clean2.csv")
        # Pre-build dictionary to speed up lookups
        meta_lookup = {
            (getattr(row, 'evid'), getattr(row, 'sta'), getattr(row, 'phase')): getattr(row, 'tt')
            for row in metadata.itertuples(index=False)
        }
    else:
        meta_lookup = None
    
    meta_lookup = comm.bcast(meta_lookup, root=0)

    f_out = None
    reader = None
    if rank == 0:
        f_out = open('mpi.output', 'w')
        chunksize = 50000 
        reader = pd.read_csv("./event_pairs.csv", chunksize=chunksize)

    while True:
        chunks = None
        if rank == 0:
            try:
                data = next(reader)
                chunks = np.array_split(data, size)
            except StopIteration:
                chunks = [None] * size # Signal termination
        
        local_df = comm.scatter(chunks, root=0)

        if local_df is None:
            break

        local_results = process_rows(local_df, db, meta_lookup, dt)

        all_results = comm.gather(local_results, root=0)

        if rank == 0:
            for process_batch in all_results:
                for line in process_batch:
                    f_out.write(line)
            f_out.flush() # Ensure data is written to disk periodically

    if rank == 0:
        f_out.close()
        print("Processing complete.")
    
    db.close()
