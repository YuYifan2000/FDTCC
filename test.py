import numpy as np
from obspy.signal.invsim import cosine_taper
import scipy
from obspy.signal.regression import linear_regression

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
        w = scipy.signal.hanning(window_len).astype("complex")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[half_win : len(y) - half_win]


def mwcs(sig1,sig2,dt):
    freqmin = 0.1
    freqmax = 10
    wind_len = pow(2, int(np.log(len(sig1))/np.log(2)))
    wind_len = len(sig1)
    tp = cosine_taper(wind_len, 0.05)
    minidx = 0
    slide = 2
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

        # for cc value

        fcur = scipy.fft.fft(cci, n=len(cci))[:wind_len//2]
        fref = scipy.fft.fft(cri, n=len(cri))[:wind_len//2]
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
    print(delta_t)
    delta_weight = np.array(delta_weight)

    return abs(np.sum(delta_t*delta_weight/sum(delta_weight))),ccv

t = np.linspace(0.01,2,200)
sig1 = np.sin(4*np.pi*t)
sig2 = np.sin(4*np.pi*(t+0.015))
delta, value = mwcs(sig1, sig2,0.01)

print(delta)
print(value)