import obspy
from obspy import read, UTCDateTime
import numpy as np
import glob
import pandas as pd
import h5py
import re
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pattern = re.compile(r'(\d{4})_(\d{1,2})_(\d{1,2})')
base_output_filename = "/scratch/users/yuyifan/catalog_dataset"
output_filename = f"{base_output_filename}_rank_{rank}.h5"
stas = pd.read_csv("./stations_network.txt", names=["network", "station", "channel"])
sampling_rate = 100.0

files = sorted(glob.glob("./genie/Catalog/*/*.hdf5"))
my_files = files[rank::size]

print(f"Rank {rank} processing {len(my_files)} files.")

# --- Helper: Batch Writer ---
def flush_batch(f_h5, batch_data):
    """Writes accumulated data to HDF5 and clears the buffer."""
    n_items = len(batch_data['evid'])
    if n_items == 0:
        return
    dset_evid = f_h5['event_id']
    dset_net = f_h5['network']
    dset_sta = f_h5['station']
    dset_wf = f_h5['waveform']
    dset_phase = f_h5['phasetype']
    current_size = dset_evid.shape[0]
    new_size = current_size + n_items
    dset_evid.resize((new_size,))
    dset_net.resize((new_size,))
    dset_sta.resize((new_size,))
    dset_wf.resize((new_size, 3, 128))
    dset_phase.resize((new_size,))
    dset_evid[current_size:] = batch_data['evid']
    dset_net[current_size:] = batch_data['net']
    dset_sta[current_size:] = batch_data['sta']
    dset_wf[current_size:] = np.array(batch_data['wf'])
    dset_phase[current_size:] = batch_data['phase']
    for key in batch_data:
        batch_data[key] = []

with h5py.File(output_filename, 'w') as f_out:
    # Create extensible datasets
    f_out.create_dataset('event_id', (0,), maxshape=(None,), dtype=h5py.string_dtype())
    f_out.create_dataset('network', (0,), maxshape=(None,), dtype=h5py.string_dtype())
    f_out.create_dataset('station', (0,), maxshape=(None,), dtype=h5py.string_dtype())
    f_out.create_dataset('waveform', (0, 3, 128), maxshape=(None, 3, 128), dtype='float32')
    f_out.create_dataset('phasetype', (0,), maxshape=(None,), dtype=h5py.string_dtype())
    
    batch_buffer = {'evid': [], 'net': [], 'sta': [], 'wf': [], 'phase': []}
    BATCH_SIZE = 500 
    total_count = 0 
    
    for file in my_files:
        print(f"Rank {rank}: Processing {file}")
        try:
            f1 = h5py.File(file, 'r')
        except Exception as e:
            print(f"Rank {rank}: Bad file {file}: {e}")
            continue
        m = pattern.search(file)
        if not m:
            f1.close(); continue
        year, month, day = map(int, m.groups())
        try:
            srcs = f1['srcs_trv'][:]
        except KeyError:
            f1.close(); continue
        if srcs.size == 0:
            f1.close(); continue
        for j in range(len(srcs)):
            evid = f'{year}{month:02d}{day:02d}{j:04d}'
            t0_day = UTCDateTime(year, month, day)
            if f'{j}_Picks_P' not in f1['Picks']:
                continue
            
            p_picks = f1['Picks'][f'{j}_Picks_P'][:]
            s_picks = f1['Picks'][f'{j}_Picks_S'][:] if f'{j}_Picks_S' in f1['Picks'] else np.empty((0, 2))
            
            pick_groups = [(p_picks, 'P'), (s_picks, 'S')]
            
            for picks, phase_type in pick_groups:
                for k in range(picks.shape[0]):
                    try:
                        pick_time = t0_day + picks[k, 0]
                        st_idx = int(picks[k, 1])
                        
                        sta_row = stas.iloc[st_idx]
                        sta = sta_row['station']
                        net = sta_row['network']
                        channel_prefix = sta_row['channel'][:2]
                        
                        wf_path = f"/oak/stanford/groups/beroza/yuyifan/data/{pick_time.year}/{pick_time.month:02d}/{pick_time.day:02d}/{net}.{sta}*{channel_prefix}*{pick_time.year}{pick_time.month:02d}{pick_time.day:02d}_{pick_time.hour:02d}0000.seed"
                        read_buffer = 10.0 # Read slightly more than needed to avoid edge effects
                        st = read(wf_path, starttime=pick_time - read_buffer, endtime=pick_time + read_buffer)
                        if st[0].stats.sampling_rate != sampling_rate:
                            st.resample(sampling_rate)
                        
                        if not st: continue
                        
                        st_slice = st.copy()
                        process_buffer = 3.0 
                        st_slice.trim(pick_time - process_buffer, pick_time + process_buffer)
                        if len(st_slice) == 0: continue

                        st_slice.detrend('constant')
                        st_slice.detrend('linear')
                        st_slice.taper(type='cosine', max_percentage=0.05)
                        st_slice.filter('bandpass', freqmin=1.0, freqmax=20.0, zerophase=True)
                        
                        half_dur = 0.64
                        st_slice.trim(pick_time - half_dur, pick_time + half_dur, nearest_sample=True)
                        st_slice.sort(keys=['channel']) 
                        data_matrix = np.zeros((3, 128), dtype=np.float32)
                        for idx, tr in enumerate(st_slice[:3]): 
                            d = tr.data
                            if len(d) > 128:
                                d = d[:128]
                            elif len(d) < 128:
                                d = np.pad(d, (0, 128 - len(d)), 'constant')
                            data_matrix[idx, :] = d

                        batch_buffer['evid'].append(evid)
                        batch_buffer['net'].append(net)
                        batch_buffer['sta'].append(sta)
                        batch_buffer['wf'].append(data_matrix)
                        batch_buffer['phase'].append(phase_type)
                        total_count += 1

                        if len(batch_buffer['evid']) >= BATCH_SIZE:
                            flush_batch(f_out, batch_buffer)

                    except Exception as e:
                        pass

        f1.close()

    flush_batch(f_out, batch_buffer)

print(f"Rank {rank} Finished. Saved {total_count} waveforms.")
