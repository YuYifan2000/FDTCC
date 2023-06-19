import numpy as np
from obspy.geodetics.base import gps2dist_azimuth,locations2degrees
import obspy
from obspy.signal.invsim import cosine_taper
import scipy
from obspy.signal.regression import linear_regression
import pandas as pd
from datetime import datetime, timedelta
from obspy import UTCDateTime
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def getCoherence(dcs, ds1, ds2):
    n = len(dcs)
    coh = np.zeros(n).astype("complex")
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def smooth(x, window="boxcar", half_win=3):
    # TODO: docsting
    window_len = 2 * half_win + 1
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2:-window_len-1:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype("complex")
    else:
        w = scipy.signal.windows.hann(window_len).astype("complex")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[half_win : len(y) - half_win]

def read_sac(filename, st,et, filter):
    f = obspy.read(filename)
    #begin = f[0].stats.starttime
    f.trim(UTCDateTime(st), UTCDateTime(et))
    if filter[0] != -1:
        f.detrend('linear')
        f.taper(0.05, type='cosine')
        f.filter('bandpass', freqmin=filter[0], freqmax=filter[1])
    try:
        data = f[0].data
    except:
        data = None
    return data

def mwcs(sig1,sig2,dt):
    freqmin = 0.1
    freqmax = 10
    wind_len = pow(2, int(np.log(len(sig1))/np.log(2)))
    wind_len = len(sig1)
    tp = cosine_taper(wind_len, 0.05)
    minidx = 0
    slide = 4
    smoothing_half_win = 3
    delta_t = []
    delta_weight = []
    delta_ccv = []
    while ((minidx+wind_len)<=len(sig1)):
        cci = sig1[minidx:minidx+wind_len]
        cri = sig2[minidx:minidx+wind_len]
        cc1 = np.correlate(cci, cri, 'valid')
        cc2 = np.correlate(cri,cri, 'valid')
        cc3 = np.correlate(cci,cci,'valid')
        ccv = np.max(np.abs(cc1 / (np.sqrt(cc2*cc3))))
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
        # get cross-spectrum & do filtering
        X = fref * (fcur.conj())

        dcur = np.sqrt(smooth(fcur2, window="hanning", half_win=smoothing_half_win))
        dref = np.sqrt(smooth(fref2, window="hanning", half_win=smoothing_half_win))
        X = smooth(X, window="hanning", half_win=smoothing_half_win)

        dcs = np.abs(X)
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[: wind_len // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin, freq_vec <= freqmax))
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.0
        phi = np.unwrap(phi)
        phi = phi[index_range]
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
        delta_t.append(m)
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v**2 * w**2)
        sx2 = np.sum(w * v**2)
        e = np.sqrt(e * s2x2 / sx2**2)
        delta_weight.append(1./e)
    delta_t = np.array(delta_t)
    delta_weight = np.array(delta_weight)
    delta_ccv = np.array(delta_ccv)
    return np.sum(delta_t*delta_weight/sum(delta_weight)), np.sum(delta_ccv*delta_weight) / np.sum(delta_weight)

def calcu_ps_interval(eve, sta, tb, tb_parameter):
    trx = tb_parameter[0]
    tdx = tb_parameter[2]
    trh = tb_parameter[1]
    tdh = tb_parameter[3]
    GCARC1 = locations2degrees(sta.iloc[0]["latitude"], sta.iloc[0]["longitude"], eve.iloc[0]["latitude"], eve.iloc[0]["longitude"])
    ih = int((eve.iloc[0]["depth"]+sta.iloc[0]["elevation"])/tdh)
    ig = ih * int(trx/tdx) + int(GCARC1/tdx)
    ptime = tb["ptime"][ig] + (GCARC1-tb["gdist"][ig])*tb["prayp"][ig] + ((eve.iloc[0]["depth"]+sta.iloc[0]["elevation"])-tb["dep"][ig])*tb["phslow"][ig]
    stime = tb["stime"][ig] + (GCARC1-tb["gdist"][ig])*tb["srayp"][ig] + ((eve.iloc[0]["depth"]+sta.iloc[0]["elevation"]) - tb["dep"][ig])*tb["shslow"][ig]
    return stime-ptime

def calcu_P(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, waveformdir, filter,tb,tb_parameter):
    sta = sta_df[sta_df['sta'] == station]
    eve1 = event_df[event_df['index'] == event1]
    eve2 = event_df[event_df['index'] == event2]
    filepath1 = f'{waveformdir}/{eve1.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}Z'
    filepath2 = f'{waveformdir}/{eve2.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}Z'
    date1 = datetime.strptime(f"{eve1.iloc[0]['date']}T{eve1.iloc[0]['hour']}:{eve1.iloc[0]['min']}:{eve1.iloc[0]['sec']}.{eve1.iloc[0]['msec']}", "%Y%m%dT%H:%M:%S.%f")
    date2 = datetime.strptime(f"{eve2.iloc[0]['date']}T{eve2.iloc[0]['hour']}:{eve2.iloc[0]['min']}:{eve2.iloc[0]['sec']}.{eve2.iloc[0]['msec']}", "%Y%m%dT%H:%M:%S.%f")
    p_s1 = calcu_ps_interval(eve1, sta,tb,tb_parameter)
    p_s2 = calcu_ps_interval(eve2, sta,tb,tb_parameter)
    if wa>0.9*(min(p_s1, p_s2)):
        wa = 0.9*(min(p_s1,p_s2))
    st1 = date1+timedelta(seconds=arr1-wb)
    et1 = date1+timedelta(seconds=arr1+wa)
    st2 = date2+timedelta(seconds=arr2-wb)
    et2 = date2+timedelta(seconds=arr2+wa)
    sig1 = read_sac(filepath1, st1,et1,filter)
    sig2 = read_sac(filepath2, st2,et2,filter)
    if ((sig1 is None) | (sig2 is None)):
        print(filepath1, st1,et1)
        print(filepath2, st2,et2)
        delta_t = 0
        cvv = 0
    else:
        delta_t, cvv = mwcs(sig1,sig2,dt)
    return arr1+delta_t-arr2, cvv

def calcu_S(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, waveformdir, filter, tb,tb_parameter):
    sta = sta_df[sta_df['sta'] == station]
    eve1 = event_df[event_df['index'] == event1]
    eve2 = event_df[event_df['index'] == event2]
    date1 = datetime.strptime(f"{eve1.iloc[0]['date']}T{eve1.iloc[0]['hour']}:{eve1.iloc[0]['min']}:{eve1.iloc[0]['sec']}.{eve1.iloc[0]['msec']}", "%Y%m%dT%H:%M:%S.%f")
    date2 = datetime.strptime(f"{eve2.iloc[0]['date']}T{eve2.iloc[0]['hour']}:{eve2.iloc[0]['min']}:{eve2.iloc[0]['sec']}.{eve2.iloc[0]['msec']}", "%Y%m%dT%H:%M:%S.%f")
    p_s1 = calcu_ps_interval(eve1, sta,tb,tb_parameter)
    p_s2 = calcu_ps_interval(eve2, sta,tb,tb_parameter)
    if wb>0.5*(min(p_s1, p_s2)):
        wb = 0.5*(min(p_s1,p_s2))
    st1 = date1+timedelta(seconds=arr1-wb)
    et1 = date1+timedelta(seconds=arr1+wa)
    st2 = date2+timedelta(seconds=arr2-wb)
    et2 = date2+timedelta(seconds=arr2+wa)
    filepath1 = f'{waveformdir}/{eve1.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}N'
    filepath2 = f'{waveformdir}/{eve2.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}N'
    sig1 = read_sac(filepath1, st1,et1,filter)
    sig2 = read_sac(filepath2, st2,et2,filter)
    if ((sig1 is None) | (sig2 is None)):
        print(filepath1, st1,et1)
        print(filepath2, st2,et2)                                           
        delta_t1 = 0
        cvv1 = 0
        flag = 1
    else:
        delta_t1, cvv1 = mwcs(sig1,sig2,dt)
        flag = 0
    filepath1 = f'{waveformdir}/{eve1.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}E'
    filepath2 = f'{waveformdir}/{eve2.iloc[0]["date"]}/{sta.iloc[0]["net"]}.{sta.iloc[0]["sta"]}.{sta.iloc[0]["channel"][:2]}E'
    sig1 = read_sac(filepath1, st1,et1,filter)
    sig2 = read_sac(filepath2, st2,et2,filter)
    if ((sig1 is None) | (sig2 is None)):
        print(filepath1, st1,et1)
        print(filepath2, st2,et2)                                           
        delta_t2 = 0
        cvv2 = 0
        flag = 1
    else:
        delta_t2, cvv2 = mwcs(sig1,sig2,dt)
        flag = 0
    delta_t = (delta_t1+delta_t2)/2.
    cvv  = (cvv1+cvv2)/2.
    if flag == 1:
        return 0, 0
    else:
        return arr1+delta_t-arr2, cvv

# size of problem setting
dt = 0.01

# path of required file
event_sel_path = './event.sel'
sta_path = './station.dat'
ttt_path = './tt_db/ttdb.txt'
wavDir = './waveforms'
dt_path = './dt.ct'
pha_path = './hypoDD.pha'

# for p type
wb = 0.5
wa = 1.0
wf = 0.3 #maximum shift length?

# for s type
wbs = 0.5
was = 1.5
wfs = 0.5

# threshold for detection
cc_threshold = 1.0
SNR_threshold = 1.0
ccv_threshold = 0.1

bp_filter = [-1, -1]

# read travel time table
tb = pd.read_csv(ttt_path, delim_whitespace=True,names=['gdist', 'dep', 'ptime', 'stime', 'prayp', 'srayp', 'phslow', 'shslow','P', 'S'])
tb_parameter = [3, 20, 0.02, 2]

# read stations
sta_df = pd.read_csv(sta_path, delim_whitespace=True, names=['longitude', 'latitude', 'net', 'sta','channel', 'elevation'])
ns = len(sta_df)
print('FDTCC reads %d stations\n'%ns)

# read events
event_df = pd.read_csv(event_sel_path, delim_whitespace=True, names=['date', 'time', 'latitude','longitude', 'depth', 'tmp1', 'tmp2', 'tmp3', 'tmp4', 'index'])
event_df["hour"] = event_df.apply(lambda x: str(x["time"]//pow(10,6)).split('.')[0], axis=1)
event_df["min"] = event_df.apply(lambda x: f'{(x["time"]%pow(10,6))//pow(10,4)}', axis=1)
event_df["sec"] = event_df.apply(lambda x: f'{(x["time"]%pow(10,4))//pow(10,2)}', axis=1)
event_df["msec"] = event_df.apply(lambda x: f'{x["time"]%pow(10,2)}', axis=1)
ne = len(event_df)
print("FDTCC reads %d events\n"%ne)

# read dt.ct while output
print('begin calculate cc')
'''
output = open('dt.cc', 'w')
dtct = open(dt_path, 'r')
Lines = dtct.readlines()
for line in Lines:
    if line[0] == '#':
        event1 = int(line.split()[1])
        event2 = int(line.split()[2])
        output.write(f'# {event1}  {event2}   0\n')

    else:
        content = line.split()
        station = content[0]
        arr1 = float(content[1])
        arr2 = float(content[2])
        phase_type = content[4]
        if phase_type == 'P':
            delta_t, ccv = calcu_P(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, wavDir, bp_filter, tb, tb_parameter)
        elif phase_type == 'S':
            delta_t, ccv = calcu_S(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, wavDir, bp_filter, tb,tb_parameter)
        else:
            continue
        if (ccv<ccv_threshold):
            continue
        output.write(f'{station} {delta_t:10.4f} {ccv:10.2f} {phase_type}\n')

dtct.close()
output.close()
'''

cores = 10
print("OPENMPI version")
if rank == 0:
    dtct = open(dt_path, 'r')
    Lines = dtct.readlines()
    dtct.close()
    ss = np.linspace(0, len(Lines), cores+1).astype('int')
    seperation = []
    for i in range(1,len(ss)):
        tmp = Lines[ss[i]:ss[i]+200]
        for j in range(0,len(tmp)):
            if tmp[j][0] == '#':
                seperation.append(j+ss[i])
                break
else:
    Lines = None
    seperation = None
Lines = comm.bcast(Lines, root=0)
seperation = comm.bcast(seperation, root=0)

if rank == 0:
    data = Lines[0:seperation[0]]
elif rank == cores-1:
    data = Lines[seperation[-1]:]
else:
    data = Lines[seperation[rank-1]:seperation[rank]] 

comm.barrier()

output = open(f'tmp{rank}.output', 'w')
for line in data:
    if line[0] == '#':
        event1 = int(line.split()[1])
        event2 = int(line.split()[2])
        output.write(f'# {event1}  {event2}   0\n')

    else:
        content = line.split()
        station = content[0]
        arr1 = float(content[1])
        arr2 = float(content[2])
        phase_type = content[4]
        if phase_type == 'P':
            delta_t, ccv = calcu_P(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, wavDir, bp_filter, tb, tb_parameter)
        elif phase_type == 'S':
            delta_t, ccv = calcu_S(station, arr1, arr2, event1, event2, sta_df, event_df, wb, wa, wavDir, bp_filter, tb,tb_parameter)
        else:
            continue
        if (ccv<ccv_threshold):
            continue
        output.write(f'{station} {delta_t:10.4f} {ccv:10.2f} {phase_type}\n')
output.close()
print(rank,' done')
comm.barrier()
print('done')

if rank==0:
    f_output = open('dt.cc', 'w')
    for i in range(0, cores):
        with open(f'tmp{i}.output') as infile:
            for line in infile:
                f_output.write(line)
    f_output.close()
