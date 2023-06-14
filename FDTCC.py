import numpy as np
from obspy.geodetics.base import gps2dist_azimuth,locations2degrees
import obspy
from obspy.signal.invsim import cosine_taper
import scipy
from obspy.signal.regression import linear_regression


INPUT1 = "input.p"
INPUT2 = "input.s1"
INPUT3 = "input.s2"



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


def Search_event(PO, EVE, n,ne):
    serial = []
    for i in range(0, ne):
        if (EVE[i]["event"] == PO[n]["event1"]):
            serial.append(i)
            break
    for i in range(0, ne):
        if (EVE[i]["event"] == PO[n]["event2"]):
            serial.append(i)
            break
    return serial

def Cal_tt(PO, PT, EVE, ST,npp, ns, ne, trx, tdx,tdh, TB):
    for i in range(0, npp):
        event = Search_event(PO, EVE, i, ne)
        PT[i]["event1"] = PO[i]["event1"]
        PT[i]["event2"] = PO[i]["event2"]
        k = 0
        for j in range(0, ns):
            PT[i]["pk"][k]["sta"] = ST[j]["sta"]
            PT[i]["pk"][k+1]["sta"] = ST[j]["sta"]
            PT[i]["pk"][k]["phase"] = "P"
            PT[i]["pk"][k+1]["phase"] = "S"
            GCARC1 = locations2degrees(ST[j]["stla"], ST[j]["stlo"], EVE[event[0]]["evla"], EVE[event[0]]["evlo"])
            ih = int(EVE[event[0]]["evdp"]/tdh)
            ig = ih * int(trx/tdx) + int(GCARC1/tdx)
            PT[i]["pk"][k]["arr1"] = TB[ig]["ptime"] + (GCARC1-TB[ig]["gdist"])*TB[ig]["prayp"] + (EVE[event[0]]["evdp"]-TB[ig]["dep"])*TB[ig]["phslow"]
            PT[i]["pk"][k+1]["arr1"] = TB[ig]["stime"] + (GCARC1-TB[ig]["gdist"])*TB[ig]["srayp"] + (EVE[event[0]]["evdp"]-TB[ig]["dep"])*TB[ig]["shslow"]
            GCARC2 = locations2degrees(ST[j]["stla"], ST[j]["stlo"], EVE[event[1]]["evla"], EVE[event[1]]["evlo"])
            ih = int(EVE[event[1]]["evdp"]/tdh)
            ig = ih * int(trx/tdx) + int(GCARC2/tdx)
            PT[i]["pk"][k]["arr2"] = TB[ig]["ptime"] + (GCARC2-TB[ig]["gdist"])*TB[ig]["prayp"] + (EVE[event[1]]["evdp"]-TB[ig]["dep"])*TB[ig]["phslow"]
            PT[i]["pk"][k+1]["arr2"] = TB[ig]["stime"] + (GCARC2-TB[ig]["gdist"])*TB[ig]["srayp"] + (EVE[event[1]]["evdp"]-TB[ig]["dep"])*TB[ig]["shslow"]
            PT[i]["pk"][k]["diff"] = PT[i]["pk"][k]["arr2"] - PT[i]["pk"][k]["arr1"]
            PT[i]["pk"][k+1]["diff"] = PT[i]["pk"][k+1]["arr2"] - PT[i]["pk"][k+1]["arr1"]
            k = k + 2
    return  PT

def replace(PT, PIN, a, b,ns):
    for i in range(0, 2*ns):
        try:
            if ((PT[a]["pk"][b]["sta"] == PIN[PT[a]["event1"]]["pa"][i]["sta"]) & (PT[a]["pk"][b]["phase"] == PIN[PT[a]["event1"]]["pa"][i]["phase"])):
                PT[a]["pk"][b]["arr1"] = PIN[PT[a]["event1"]]["time"][i]
                break
        except TypeError:
            continue
    for i in range(0, 2*ns):
        try:
            if ((PT[a]["pk"][b]["sta"] == PIN[PT[a]["event2"]]["pa"][i]["sta"]) & (PT[a]["pk"][b]["phase"] == PIN[PT[a]["event2"]]["pa"][i]["phase"])):
                PT[a]["pk"][b]["arr2"] = PIN[PT[a]["event2"]]["time"][i]
                break
        except TypeError:
            continue
    return PT

def Correct_Pshift(PT, a,b,c,ne,ns,npp):
    for i in range(0, ne*ns):
        for j in range(0, npp):
            if (PT[j]["event1"] == c[i]):
                for k in range(0, 2*ns):
                    if ((PT[j]["pk"][k]["sta"]==b[i]) & (PT[j]["pk"][k]["phase"]=='P')):
                        a[i] = a[i] + float(PT[j]["pk"][k]["arr1"])
                        break
                break
            if (PT[j]["event2"] == c[i]):
                for k in range(0, 2*ns):
                    if ((PT[j]["pk"][k]["sta"]==b[i]) & (PT[j]["pk"][k]["phase"]=='P')):
                        a[i] = a[i] + float(PT[j]["pk"][k]["arr2"])
                        break
                break
    return a

def Correct_Sshift(PT, a,b,c,ne,ns,npp):
    for i in range(0, ne*ns):
        for j in range(0, npp):
            if (PT[j]["event1"] == c[i]):
                for k in range(0, 2*ns):
                    if ((PT[j]["pk"][k]["sta"]==b[i]) & (PT[j]["pk"][k]["phase"]=='S')):
                        a[i] = a[i] + float(PT[j]["pk"][k]["arr1"])
                        break
                break
            if (PT[j]["event2"] == c[i]):
                for k in range(0, 2*ns):
                    if ((PT[j]["pk"][k]["sta"]==b[i]) & (PT[j]["pk"][k]["phase"]=='S')):
                        a[i] = a[i] + float(PT[j]["pk"][k]["arr2"])
                        break
                break
    return a

def read_sac(filename, st,et, filter):
    f = obspy.read(filename)
    begin = f[0].stats.starttime
    f.trim(begin+st, begin+et)
    if filter[0] != -1:
        f.detrend('linear')
        f.taper(0.05, type='cosine')
        f.filter('bandpass', freqmin=filter[0], freqmax=filter[1])
    data = f[0].data
    return data

def mwcs(sig1,sig2,dt):
    freqmin = 0.1
    freqmax = 10
    #wind_len = pow(2, int(np.log(len(sig1))/np.log(2)))
    wind_len = len(sig1)
    tp = cosine_taper(wind_len, 0.05)
    minidx = 0
    slide = 4
    smoothing_half_win = 3
    delta_t = []
    delta_weight = []
    cc1 = np.correlate(sig1, sig2, 'valid')
    cc2 = np.correlate(sig1,sig1, 'valid')
    cc3 = np.correlate(sig2,sig2,'valid')
    ccv = np.max(np.abs(cc1 / (np.sqrt(cc2*cc3))))
    while ((minidx+wind_len)<=len(sig1)):
        cci = sig1[minidx:minidx+wind_len]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = sig2[minidx:minidx+wind_len]
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

    return abs(np.sum(delta_t*delta_weight/sum(delta_weight))),ccv

def SubccP(PT, waveP, event, i,j, wa, wf,dt,ns,cc_threshold):
    quality = 1
    t_shift = int(1/dt)
    s_p = PT[i]["pk"][2*j+1]["arr1"]-PT[i]["pk"][2*j]["arr1"]
    if (s_p<=0):
        quality = 0
    else:
        wa1 = wa
        w = wf
        if (wa>0.9*s_p):
            wa1 = 0.9*s_p
        if (w>0.5*(wa1+wb)):
            w = 0.5*(wa1+wb)
        ref_shift = int(w/dt)
        w = ref_shift * dt
        Npoint = int(2*w/dt-0.5)
        Wpoint = int((wa1+wb)/dt-0.5)
        #sig1 = waveP[event[0]*ns+j][t_shift:t_shift+Wpoint]
        #sig2 = waveP[event[1]*ns+j][t_shift:t_shift+Wpoint]
        sig1 = waveP[event[0]*ns+j][:]
        sig2 = waveP[event[1]*ns+j][:]
        cc_delta, ccv = mwcs(sig1,sig2,dt)

        if (abs(PT[i]["pk"][2*j]["arr1"]-PT[i]["pk"][2*j]["arr2"]+cc_delta)>cc_threshold):
            quality = 0
    return quality, cc_delta, ccv

def SubccS(PT, waveS1, waveS2, event, i,j, was, wbs, wfs,dt,ns,cc_threshold):
    quality = 1
    t_shift = int(1/dt)
    s_p = PT[i]["pk"][2*j+1]["arr1"]-PT[i]["pk"][2*j]["arr1"]
    if (s_p<=0):
        quality = 0
    else:
        wbs1 = wbs
        w = wfs
        tt = 0
        if (wbs1>0.5*s_p):
            wbs1 = 0.5*s_p
            tt = int((wbs-0.5*s_p)/dt-0.5)
        if (w>0.5*(was+wbs1)):
            w = 0.5*(was+wbs1)
        ref_shift = int(w/dt)
        w = ref_shift * dt
        Npoint = int(2*w/dt-0.5)
        Wpoint = int((was+wbs)/dt-0.5)
        sig1 = waveS1[event[0]*ns+j][t_shift+tt:tt+t_shift+Wpoint]
        sig2 = waveS1[event[1]*ns+j][t_shift+tt:tt+t_shift+Wpoint]
        cc_delta1, ccv1 = mwcs(sig1,sig2,dt)

        sig1 = waveS2[event[0]*ns+j][t_shift+tt:tt+t_shift+Wpoint]
        sig2 = waveS2[event[1]*ns+j][t_shift+tt:tt+t_shift+Wpoint]
        cc_delta2, ccv2 = mwcs(sig1,sig2,dt)

        cc_delta = (cc_delta1+cc_delta2)/2.
        ccv = (ccv1+ccv2)/2.
        if (abs(PT[i]["pk"][2*j]["arr1"]-PT[i]["pk"][2*j]["arr2"]+cc_delta)>1):
            quality = 0
    return quality, cc_delta, ccv




# size of problem setting
NPP_ = 400000
NS = 1000
NE = 40000
ntb = 4000

'''
event_sel_path = './test/event.sel'
sta_path = './test/station.dat'
ttt_path = './test/tt_db/ttdb.txt'
wavDir = './waveforms'
dt_path = './test/dt.ct'
pha_path = './test/hypoDD.pha'
'''
event_sel_path = './Demo/event.sel'
sta_path = './Demo/station.dat'
ttt_path = './Demo/tt_db/ttdb.txt'
wavDir = './Demo/waveforms'
dt_path = './Demo/dt.ct'
pha_path = './Demo/hypoDD.pha'

# for p type
wb = 0.2
wa = 1.0
wf = 0.3 #maximum shift length?

# for s type
wbs = 0.5
was = 1.5
wfs = 0.5

# threshold for detection
dt = 0.01
cc_threshold = 1.0
SNR_threshold = 1.0
max_traveltime = 2.
ccv_threshold = 0.7

ttt_range = [3, 20] # first value horzion degree, second value depth in km
trx = ttt_range[0]
trh = ttt_range[1]
ttt_grid = [0.02, 2] # first value horizontal degree, second value depth in km
tdx = ttt_grid[0]
tdh = ttt_grid[1]
bp_filter = [-1, -1] # low frequency and high frequency


# read_phase
PIN = [None] * NE
phasefile = open(pha_path, 'r')
Lines = phasefile.readlines()
for line in Lines:
    if line[0] == '#':
        ID_event = int(line.split()[-1])
        j = 0
        pa = [None] * (2 * NS)
        time = [None] * (2 * NS)
        PIN[ID_event] = {'time': time, 'pa': pa}
    else:
        content = line.split()
        PIN[ID_event]["pa"][j]={"sta":content[0], "phase":content[3]}
        PIN[ID_event]["time"][j] = float(content[1])
        j = j+1
phasefile.close()
del j
# read dt.ct
PO = [None] * NPP_
dtct = open(dt_path, 'r')
Lines = dtct.readlines()
k = 0
for line in Lines:
    if line[0] == '#':
        PO[k] = {"event1":int(line.split()[1]), "event2":int(line.split()[2]),"pk":[None]*2*NS}
        k = k + 1
        if k>NPP_:
            raise KeyError("Need bigger NPP_")
        kk = 0
    else:
        content = line.split()
        PO[k-1]["pk"][kk]={"sta":content[0], "phase":content[4],"arr1":float(content[1]),"arr2":float(content[2]),"ccv":-1,"shift":-1,"quality":0,"diff":-1}
        kk = kk + 1
        if kk > NS:
            raise KeyError("Need bigger NS")
dtct.close()
npp = k
del k, kk
print("FDTCC reads %d event-pairs\n"%npp)

# read stations
k = 0
ST = [None] * NS
sta_file = open(sta_path, 'r')
Lines = sta_file.readlines()
for line in Lines:
    content = line.split()
    ST[k] = {"stlo":float(content[0]), "stla": float(content[1]), "net":content[2],"sta":content[3], "comp":content[4],"elev": content[5]}
    k = k + 1
    if k > NS:
        raise ValueError('increase NS!')
ns = k
print('FDTCC reads %d stations\n'%k)
# read events
k = 0
EVE = [None] * NE
eve_file = open(event_sel_path, 'r')
Lines = eve_file.readlines()
for line in Lines:
    content = line.split()
    EVE[k] = {'pSNR':-1, 'sSNR':-1, 'evla':float(content[2]), 'evlo':float(content[3]), 'evdp':float(content[4]), 'date':content[0], 'time':float(content[1]),'sec':float(content[1][-4:])/100., 'event':int(content[-1])}
    if (EVE[k]["evdp"]>ttt_range[1]):
        raise ValueError("Increase depth range in time table")
    k = k + 1
    if k>NE:
        raise ValueError("Increase NE!")    
ne = k

print("FDTCC reads %d events\n"%k)
del k

# read travel time table
TB = [None] * ntb
tttb = open(ttt_path,'r')
Lines = tttb.readlines()
k = 0
for line in Lines:
    content = line.split()
    TB[k] = {'gdist':float(content[0]), 'dep':float(content[1]), 'ptime':float(content[2]), 'stime':float(content[3]), 'prayp':float(content[4]), 'srayp':float(content[5]), 'phslow':float(content[6]), 'shslow':float(content[7]), 'pphase':'P', 'sphase':'S'}
    k = k + 1
    if k>ntb:
        raise ValueError("increase ntb!")
print("FDTCC reads %d travel-times\n"%k)
del k

# create event pairs with theoretical travel times
PT = [None] * npp
for i in range(0, npp):
    pk =[]
    for j in range(0, 2*NS):
        pk.append({"sta":"", "phase":"","arr1":-1,"arr2":-1,"ccv":-1,"shift":-1,"quality":0,"diff":-1})
    PT[i] = {"pk":pk}

PT = Cal_tt(PO, PT, EVE, ST,npp, ns, ne, trx, tdx,tdh, TB)
print("done creating database")


# update PT based on PO
for i in range(0, npp):
    for j in range(0, 2*ns):
        PT = replace(PT, PIN, i, j,ns)

# read waveform
fp1 = open(INPUT1, 'w')
fp2 = open(INPUT2, 'w')
fp3 = open(INPUT3, 'w')
for i in range(0, ne):
    for j in range(0, ns):
        fp1.write(f"{wavDir}/{EVE[i]['date']}/{ST[j]['net']}.{ST[j]['sta']}.{ST[j]['comp'][0]}{ST[j]['comp'][1]}Z  {ST[j]['sta']}   {EVE[i]['sec']:.2f}  {EVE[i]['event']}\n")
        fp2.write(f"{wavDir}/{EVE[i]['date']}/{ST[j]['net']}.{ST[j]['sta']}.{ST[j]['comp'][0]}{ST[j]['comp'][1]}E  {ST[j]['sta']}   {EVE[i]['sec']:.2f}  {EVE[i]['event']}\n")
        fp3.write(f"{wavDir}/{EVE[i]['date']}/{ST[j]['net']}.{ST[j]['sta']}.{ST[j]['comp'][0]}{ST[j]['comp'][1]}N  {ST[j]['sta']}   {EVE[i]['sec']:.2f}  {EVE[i]['event']}\n")
fp1.close()
fp2.close()
fp3.close()

fp1 = open(INPUT1, 'r')
fp2 = open(INPUT2, 'r')
fp3 = open(INPUT3, 'r')
Lines1 = fp1.readlines()
Lines2 = fp2.readlines()
Lines3 = fp3.readlines()
staP = []; staS1=[]; staS2 = []; la_staP=[]; la_staS1 = [];la_staS2=[]
ptriger = []; s1triger = [];s2triger = []; labelP=[];labelS1=[];labelS2=[]
for i in range(0, ne*ns):
    staP.append(Lines1[i].split()[0])
    la_staP.append(Lines1[i].split()[1])
    ptriger.append(float(Lines1[i].split()[2]))
    labelP.append(int(Lines1[i].split()[3]))

    staS1.append(Lines2[i].split()[0])
    la_staS1.append(Lines2[i].split()[1])
    s1triger.append(float(Lines2[i].split()[2]))
    labelS1.append(int(Lines2[i].split()[3]))

    staS2.append(Lines3[i].split()[0])
    la_staS2.append(Lines3[i].split()[1])
    s2triger.append(float(Lines3[i].split()[2]))
    labelS2.append(int(Lines3[i].split()[3]))
fp1.close()
fp2.close()
fp3.close()
ptriger = Correct_Pshift(PT, ptriger,la_staP,labelP,ne,ns,npp)
s1triger = Correct_Sshift(PT, s1triger,la_staS1,labelS1,ne,ns,npp)
s2triger = Correct_Sshift(PT, s2triger,la_staS2,labelS2,ne,ns,npp)

####
markP = np.zeros([ne*ns,1])
markS1 = np.zeros([ne*ns,1])
markS2 = np.zeros([ne*ns,1])
waveP = [None] * (ne*ns)
waveS1 = [None] * (ne*ns)
waveS2 = [None] * (ne*ns)
for i in range(0, ne*ns):
    markP[i] = 1; markS1[i]=1;markS2[i]=1
    try: 
        waveP[i] =read_sac(staP[i], ptriger[i] - wb - 1, ptriger[i] + wa + 1,bp_filter)
    except:
        markP[i] = 0

    try:
        waveS1[i]=read_sac(staS1[i], s1triger[i] - wb - 1, s1triger[i] + wa + 1, bp_filter)
    except:
        markS1[i] = 0

    try:
        waveS2[i] =read_sac(staS2[i], s2triger[i] - wb - 1, s2triger[i] + wa + 1, bp_filter)
    except:
        markS2[i] = 0
    
    #missing snr part
print("starts to calculate ccv\n")


for i in range(0,npp):
    event = Search_event(PT, EVE, i, ne)
    for j in range(0, ns):
        if ((markP[event[0]*ns+j] == 0) | (markP[event[1]*ns+j]==0)):
            PT[i]["pk"][2*j]["quality"] = 0
            continue
        q, cc_delta, ccv = SubccP(PT, waveP, event, i,j, wa, wf,dt,ns,cc_threshold)
        PT[i]["pk"][2*j]["quality"] = q
        PT[i]["pk"][2*j]["ccv"] = ccv
        PT[i]["pk"][2*j]["shift"] = cc_delta

for i in range(0,npp):
    event = Search_event(PT, EVE, i, ne)
    for j in range(0, ns):
        if ((markS1[event[0]*ns+j] == 0) | (markS1[event[1]*ns+j]==0)):
            PT[i]["pk"][2*j+1]["quality"] = 0
            continue
        q, cc_delta, ccv = SubccS(PT, waveS1, waveS2, event, i,j, was, wbs, wfs,dt,ns,cc_threshold)
        PT[i]["pk"][2*j+1]["quality"] = q
        PT[i]["pk"][2*j+1]["ccv"] = ccv
        PT[i]["pk"][2*j+1]["shift"] = cc_delta

output = open('dt.cc', 'w')
for i in range(0, npp):
    output.write(f"# {PT[i]['event1']}  {PT[i]['event2']}   0\n")
    for j in range(0, 2*ns):
        if ((PT[i]['pk'][j]['quality']==1) & (PT[i]['pk'][j]['ccv']>=ccv_threshold) &(PT[i]['pk'][j]['ccv']>0)):
            output.write(f"{PT[i]['pk'][j]['sta']:5s} {(PT[i]['pk'][j]['arr1']-PT[i]['pk'][j]['arr1']+PT[i]['pk'][j]['shift']):10.4f} {PT[i]['pk'][j]['ccv']:10.2f} {PT[i]['pk'][j]['phase']:.3s}\n")

output.close()