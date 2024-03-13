# General
- Conda env: tnt
- Git repo: IP-TNT (https://github.com/koenraijer/IP-TNT)

# Structure
- root:
  - `processing_empatica.ipynb`: notebook to process the Empatica data
  - `processing_inquisit.ipynb`: notebook to process the Inquisit data
  - `combining_empatica_inquisit.ipynb`: notebook to combine the Empatica and Inquisit data
  - `recalibrating_empatica_inquisit.ipynb`: notebook that uses Empatica tags/events to attempt to recalibrate the Inquisit data.
  - `empatica_helpers.py`: helper functions for processing the Empatica data
  - `inquisit_helpers.py`: helper functions for processing the Inquisit data
  - `helpers.py`: general helper functions
- input:
  - empatica: Empatica data
  - inquisit: Inquisit data
  - pilot: pilot data
- output:
  - `empatica_combined_raw.csv`: all raw ACC, HR, TEMP, EDA, BVP data. 
  - `inquisit_combined_raw.csv`: all Inquisit trials with their time, code, and response.
  - `empatica_inquisit_merged.csv`: all Empatica and Inquisit data merged on time. 


# To do
- Include Think trials in Inquisit output like: 
  -   `t_na` = think trial without intrusion rating
  -   `t_wrong` = unsuccessful think trial, i.e., one where subjects did not manage to think about the scene.
  -   `t` = think trial (successful)
  -   `nt_na` = no-think trial without intrusion rating
  -   `nt` = no-think trial (successful)
  -   `i` = intrusion, i.e., unsuccessful no-think trial where subjects thought briefly or often about the scene.