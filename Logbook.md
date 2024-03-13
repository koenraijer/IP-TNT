**Wednesday, 13 March 2024 11:36, written on Koen’s MacBook Air, at Westelijke Randweg 50, Geleen:**
- Per Marijn's recommendation, tested [MNE.tools](https://mne.tools/stable/index.html). I found that MNE is mainly meant for EEG/MEG/ECG signals. Not so much for physiological signals.
- Per Hunor's recommendation, looked at [NeuroKit](https://neuropsychology.github.io/NeuroKit/functions/ppg.html). This looks much more promising, offering specific functions for PPG and EDA data.
- Started working with NeuroKit, some things I've noticed, something I noticed is that there are large differences between Empatica's HR output and NeuroKit's, e.g., 56 (NK) vs. 84 (Empatica).
- Got HR and HRV data from PPG sensor data, making sure none of the calculations are based on Empatica's proprietory, closed-source datasets.
- Started looking into **rolling windows**, specically [Pandas windowing functions](https://pandas.pydata.org/pandas-docs/stable/reference/window.html#api-functions-rolling). 
  - Interesting tutorial: https://machinelearningtutorials.org/pandas-window-functions-a-comprehensive-guide-with-examples/
  - Exponential Moving Average to bias recent values.
- BACK TO BASICS: before looking into windowing and analysing etcetera, I should make sure I have combined the raw data from all sensors (PPG, EDA, TEMP, ACC) in a single dataframe with the same sampling frequency, trimmed (starting/end times) and cleaned (visual inspection, artifact detection). 
  - Note: that means I won't use IBI.csv!
- Replaced `resampy` for resampling by NeuroKit2's `signal_resample` function, with "FFT" as the method. 

*Completed:*
- Combined all raw sensor data from PPG, EDA, ACC and TEMP into one. Upsampled ACC, EDA and TEMP to 64 Hz using FFT resampling. 
*To do:*
- Find out whose data the files without "pp" in the name are. 
- Match all data files with participant demographics.