**Wednesday, 13 March 2024 11:36, written on Koen’s MacBook Air, at Westelijke Randweg 50, Geleen:**
*Thoughts:*
- Per Marijn's recommendation, tested [MNE.tools](https://mne.tools/stable/index.html). I found that MNE is mainly meant for EEG/MEG/ECG signals. Not so much for physiological signals.
- Per Hunor's recommendation, looked at [NeuroKit](https://neuropsychology.github.io/NeuroKit/functions/ppg.html). This looks much more promising, offering specific functions for PPG and EDA data.
- Started working with NeuroKit, some things I've noticed, something I noticed is that there are large differences between Empatica's HR output and NeuroKit's, e.g., 56 (NK) vs. 84 (Empatica). UPDATE: this is likely due to the fact that NK is not recognising most of the peaks. 
- Got HR and HRV data from PPG sensor data, making sure none of the calculations are based on Empatica's proprietory, closed-source datasets.
- Started looking into **rolling windows**, specically [Pandas windowing functions](https://pandas.pydata.org/pandas-docs/stable/reference/window.html#api-functions-rolling). 
  - Interesting tutorial: https://machinelearningtutorials.org/pandas-window-functions-a-comprehensive-guide-with-examples/
  - Exponential Moving Average to bias recent values.
- BACK TO BASICS: before looking into windowing and analysing etcetera, I should make sure I have combined the raw data from all sensors (PPG, EDA, TEMP, ACC) in a single dataframe with the same sampling frequency, trimmed (starting/end times) and cleaned (visual inspection, artifact detection). 
  - Note: that means I won't use IBI.csv!

*Changes:*
- Replaced `resampy` for resampling by NeuroKit2's `signal_resample` function, with "FFT" as the method. 
- Combined all raw sensor data from PPG, EDA, ACC and TEMP into one. Upsampled ACC, EDA and TEMP to 64 Hz using FFT resampling. 

*To do:*
- Find out whose data the files without "pp" in the name are.
- Match all data files with participant demographics.

---
**Thursday, 14 March 2024 09:09, written on Koen’s MacBook Air, at Geldersestraat 2, Sittard:**
- Perhaps there is a reason wrist-worn wearables, for example by Garmin, don't present HRV measures. I might need to stick to statistical measures of HR. 

*Artifact detection options:*
[Src](https://ieeexplore-ieee-org.mu.idm.oclc.org/document/7511485)
- The Discrete Wavelet Transform (DWT) enables the observation of how the frequency spectrum of a signal evolves over time. This temporal-frequency resolution capability is one of the key advantages of DWT, allowing for a more nuanced analysis of signals where the frequency content changes with time. This step helps in filtering out unwanted noise and separates the signal into its essential components (AC/DC component extraction). In the context of the PPG filter, the DC, or direct current component, represents static blood flow, while the AC or alternating current component, represents fluctuating blood flow. 
- Improving data quality EDA: https://support.mindwaretech.com/2017/12/improving-data-quality-eda/
- [FLIRT: A feature generation toolkit for wearable data](https://www-sciencedirect-com.mu.idm.oclc.org/science/article/pii/S0169260721005356?via%3Dihub) has several artifact detection options. 
- Motion artifact detection with or without acceleration data?
- [Efficient envelope-based PPG denoising algorithm](https://www.sciencedirect.com/science/article/pii/S1746809423011266) <-- seems like a low-resource, interpretable option.

*Matching anonymous data files to participants:*
  - `d1 2` - Wednesday April 12, 2023 16:16:54 (pm) - pp15
  - `d1_1` - Wednesday April 12, 2023 14:12:58 (pm) - pp13
  - `1681713254_A03F6E` - Monday April 17, 2023 08:34:14 (am) - pp16
  - `1681717717_A03F6E` - Monday April 17, 2023 09:48:37 (am) - pp17
  - `d1_3` - Tuesday April 18, 2023 15:05:05 (pm) - pp18
  - `d2_1_1` - Tuesday April 18, 2023 11:46:43 (am) - pp17
  - `d2_2` - Tuesday April 18, 2023 16:32:11 (pm) - pp16
  - `d1` - Wednesday April 19, 2023 12:59:55 (pm) - pp19
  - `d1_4` - Wednesday April 19, 2023 14:17:17 (pm) - pp20
  - `d2` - Wednesday April 19, 2023 15:43:14 (pm) - pp18
  - `d2_1` - Thursday April 20, 2023 09:20:35 (am) - pp19
  - `d2_4` - Thursday April 20, 2023 12:31:20 (pm) - pp20

Note: matching was done using the questionnaire, I still need to use the excel file for the ParticipantOverview. 

- I'm not sure how Empatica dealt with missing values.

*Changes:*
- Inspected distribution of trimmed data sizes
- Visually inspected alignment of data sources across time
- Inspected statistical properties of raw data
- Checked for duplicate data (found none)
- Cleaned PPG signal using `nk.ppg_clean`.
- Performed peak detection using `nk.ppg_findpeaks` and fixed peak placement using `nk.signal_fixpeaks`. 

- Peak detection using NeuroKit is BAD. In fact, I'm in the exact same situation as [this GitHub issue](https://github.com/neuropsychology/NeuroKit/issues/462). I'd rather not use Empatica's closed-source IBI.csv file, but I'm having trouble doing proper peak detection otherwise. NeuroKit's peak detection seems to be especially troublesome with low amplitude signals, where very few peaks are being detected. 
![output/artefact_correction.png]
![output/heartpy_ppg_peak_detection.png]
![peak_detection.png]

*To do:*
- Test packages for physiological data processing: [src for hrv Python packages](https://www.sciencedirect.com/science/article/pii/S0169260721005356)
  - Seemingly abandoned/unmaintained/irrelevant: FLIRT, hrv-analysis, PHRV, hrv. Irrelevant: PySiology. 
  - Promising: [BioSPPy](https://github.com/scientisst/BioSPPy), HeartPy, [PyPhysio](https://gitlab.com/a.bizzego/pyphysio/-/blob/master/tutorials/1-signals.ipynb)
  - For PPG: [pyPPG](https://github.com/godamartonaron/GODA_pyPPG).

Useful ChatGPT output on filtering:
```md
1. **Bandpass Filtering:** PPG/BVP signals contain valuable heart rate information typically in the 0.5 Hz to 5 Hz range for adults at rest, which can extend up to 4 Hz or above during physical activity. A bandpass filter can effectively isolate this frequency range, eliminating both higher-frequency noise (e.g., motion artifacts) and lower-frequency trends (e.g., slow vasomotor fluctuations).
2. **Notch Filtering:** To remove power line interference, commonly at 50 Hz or 60 Hz depending on the region, a notch filter can be applied specifically at these frequencies.
3. **Moving Average or Low-pass Filtering:** For smoothing the signal and further reducing high-frequency noise, a moving average filter or a low-pass filter can be used. However, the selection of cutoff frequency is critical to avoid distortion of the signal of interest.
4. **Adaptive Filtering:** For signals with varying frequencies or for removing specific types of artifacts like motion or respiration effects, adaptive filters can be beneficial. These filters adjust their parameters in real-time based on the characteristics of the input signal.
5. **Wavelet Denoising:** This technique is useful for non-stationary physiological signals like PPG/BVP, allowing for noise reduction while preserving significant signal features across different frequency bands.
```

- *Conclusion after experimentation:*
  - The signal may be too noisy to do beat detection myself. 
  - I might not have another option than use Empatica's close-source/proprietory code-generated data (e.g., IBI.csv and HR.csv). 
  - It will allow me to focus on problems more relevant to my thesis. 

**Saturday, 16 March 2024 09:13, written on Koen’s MacBook Air, at Geldersestraat 1, Sittard:**
- Let's take a step back and look at what others did to preprocess Empatica data. I've found no other Empatica-specific packages who calculate the inter-beat intervals from scratch.  
- Including the ibi data:
  - Number of folders with no IBI data: 17 / 67
- IMPORTANT: what I thought was noise is actually signal. 
- Let's try aggressive denoising
  - Tried using a variable moving average, where the smoothing window size changes with the variance in the ppg signal (for a window of 250ms). Was difficult computationally, tried using the Kaufman Adaptive Moving Average taken from finance. 
  - Realised that people have done this before me.

*Options:* ([Peng, 2014/12](zotero://select/items/2_H8SKC8XR))
- Fast Fourier transform (FFT) combining LMS (FFT-LMS) method
- https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html 

NOTE: My earlier attempt at variable moving average was a very simple implementation of the statistical approach suggested by [Pollreisz, 2022-04-01](zotero://select/items/2_HQPF2BYJ).

THIS WORKS: ([Zhang, 2/2015](zotero://select/items/2_5VJI7BHN)), specifically their advice applied to this code:
[src](https://github.com/maryambayat-code/WAVELET-PPG/blob/main/waveletcodeforPPG.ipynb)
```py
from numpy import size
coeff = pywt.wavedec(filterd2, wavelet='db2', level=3)
```

The advice:
> "For example, when fs is 25 Hz instead of 125 Hz, the number of frequency bins, N , should take about 1/5 of the original value such that other parameters’ values (e.g. ∆s and ∆) maintain the same physical measures." 

*My description of what I did:*
Motion artifact detection was performed by deconstructing the signal into two frequency bins using wavelet decomposition. The number of frequency bins was chosen based on the heuristic that the number of frequency bins should be reduced by a factor `r` when the sampling frequency of the signal is reduced by factor `r`, first introduced in the paper by Zhang et al. (2015)[1]. The script then proceeds to threshold the first two sets of wavelet coefficients, leaving coefficients within a calculated threshold unchanged and modifying those outside the threshold using a function dependent on the threshold and a calculated parameter. These thresholded coefficients are subsequently used to reconstruct the signal via an inverse wavelet transform. Finally, the reconstructed signal is smoothed using a Savitzky-Golay filter, a type of low-pass filter effective for smoothing noisy data. The original, reconstructed, and smoothed signals are all plotted for visual comparison.

### I COULD EXPAND IT WITH SPARSE SIGNAL RECONSTRUCTION

Originally from TROIKA paper
sparse signal reconstruction (SSR):
Sources describing it:
"[19] I. F. Gorodnitsky and B. D. Rao, “Sparse signal reconstruction from limited data using FOCUSS: a re-weighted minimum norm algorithm,” IEEE Trans. on Signal Processing, vol. 45, no. 3, pp. 600–616, 1997. [20] D. Donoho, “Compressed sensing,” IEEE Transactions on Information Theory, vol. 52, no. 4, pp. 1289–1306, 2006. [21] M. Elad, Sparse and Redundant Representations: From Theory to Applications in Signal and Image Processing. Springer, 2010."

Examples from Github:
- Done in Python: https://github.com/DingNingCN/OMP-algorithm-compressed-sensing/blob/main/3.OMP_Algorithm%20Reconstruction.ipynb
- Done in Matlab: https://github.com/sachin-chandani/Heart-Rate-Monitoring-PPG
- May also be relevant: https://github.com/leopoldjouffroy/Sparse-signal-reconstruction

This just might have messed up the temporal relation of the data. Tt appears that, since the final wavelet is reconstructed from final_y, a list of the two frequency bins, they are simply added back together sequentially? In other words, it fucks up the temporal association of my data?

Since I decompose the signal in 2 frequency domains.

https://github.com/DingNingCN/OMP-algorithm-compressed-sensing/blob/main/3.OMP_Algorithm%20Reconstruction.ipynb <-- to continue at!!
Or more advanced: https://www-sciencedirect-com.mu.idm.oclc.org/science/article/pii/S1746809422001549

([Gaur, 2015](zotero://select/items/2_8M9MAVK8))

**Sunday, 17 March 2024 13:31, written on Koen’s MacBook Air, at Geldersestraat 1, Sittard:**
I've been trying to implement the X-LMS algorithm: 
1. Bandpass filter with 4th order butterworth infinite impulse response filter (IIR) in the range 0.3 - 5 Hz, applied to the raw ppg signal. 
2. Singular Value Decomposition (SVD) to generate a motion artifact reference for the adaptive filter. 
3. Modified LMS adaptive filter, where the coefficients ($h(n)$) are updateda according to the least mean error ($e(n)$). An identical filter is placed in the reference signal path to adjust the weights. This adjustment is called X-LMS.

Formulas:
1. $y_c(n) = w^T(n) * u(n)$ : Output generation
2. $e(n) = d(n) - y_C(n)$ : Error calculation
3. $u_{C^*}(n) = \Sum_{i=0}^{I-1} c^*_i * u(n - i - M + 1)$
4. $w(n+1) = w(n) + \mu * u_{C^*} (n)e(n)$ : Weight updates

However, **signular value decomposition has proved to be way too computationally expensive**. In addition, Welhenge et al. stated that LMS algorithms are generally more effective than SVD-based algorithms. 

Decided to start simpler and implement an LMS algorithm with the statistical score as reference signal ([Welhenge, 2019-02-04](zotero://select/items/2_VDUMGETB)). 
([Wei, 2008-07-01](zotero://select/items/2_9QT7UHA3))

> "Signals from all the axes (x, y and z) of the accelerometer are added together to generate the acceleration signal (ak). This takes the effects of noise on all three axes. This signal is represented with a FIR model. Using an adaptive filter, the FIR coefficients are derived [11]. The output is an estimate of the noise (^nk). This signal is then subtracted from the PPG signal (dk) which is the addition of the cardiac component (sk) and the noise component (nk) to get an estimate (ek) for the cardiac component. The adaptive filter continuously adjusts the filter coefficients. The purpose of this method is to minimize the error between nk and n^k. "

For `padasip`: 
> https://github.com/matousc89/Python-Adaptive-Signal-Processing-Handbook/blob/master/notebooks/padasip_adaptive_filters_basics.ipynb


**Thursday, 21 March 2024 10:34, written on Koen’s MacBook Air, at Geldersestraat 1, Sittard:**
In order to understand how much missing data I have without already calculating the feature set, I must set the IBI data at a fixed sampling rate (64 Hz). I did that and calculated 

**Friday, 22 March 2024 16:33, written on Koen’s MacBook Air, at Bolksbeekstraat 29bis, Utrecht:**
- The question is about which dataset I'm going to use: `df['intrusion]`, `df['intrusion_nothink]`, or `df['intrusion_tnt']`. The deciding factor here is which dataset is going to have most discriminatory ability. Marijn thinks this will be `intrusion_nothink` because successful nothink trials are more different from intrusions (i.e., failed nothink trials) than intrusions are different from think trials, becuase in the latter case, both constitute a thought. However, I am inclined to think that the negative valence aspect that succesful and failed nothink trials share makes them more similar. 
- Another question is: to what degree does it work to engineer additional statistical features from domain-specific features. For example, calculating SD for SCL calculated from EDA. 
- I used NeuroKit2 to calculate EDA features and have decided that 