			Fast double-difference cross-correlation (FDTCC)
							Authors: Min Liu & Miao Zhang
							mliu@cuhk.edu.hk & miao.zhang@dal.ca
             
Mar 10, 2026. With many people reaching out asking about FDTCC for mseed files, or interested in running python for it. I have added  two scripts as examples of how I first cut the waveforms, then calculate the cross correlations. These two files could help you processing MWCS for your data.             
June 12, 2023. Yifan Yu added cross correlation calculation in frequency cross spectrum following (Poupinet & Ellsworth, 1984).

Check Miao Zhang's Github page for original use.

If you want to use it, use 'easy_version.py'. I accelerate by mpi4py. I tried to implement it using C, had some problems, still working on it.
You can find easy_version for python version of transforming 'dt.ct', 'events.sel', 'stations.dat', 'hypoDD.pha' into a 'dt.cc' file.
