#!/bin/bash

source /work/data/haapanie/anaconda3/bin/activate plot_mhws
python /work/data/haapanie/reanalysis_smartmet_mhws.py
python /work/data/haapanie/measured_smartmet_mhws.py
python /work/data/haapanie/mhw_plots_to_ftp.py

date >> /work/data/haapanie/dates_of_mhw_updates.txt
