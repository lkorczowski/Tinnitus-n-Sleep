import numpy as np
from datetime import *
import pandas as pd
import os
PATH = os.getcwd()
import sys
sys.path.append(PATH + '/../')
import mne
from tinnsleep.config import Config

#Note for future Robin: this code is best executed by spyder.


# The following function should be included somewhere in the code files and pytested.
def process_csv(df):
    # Retrieving the timestamps of each epoch
    get_eps = df["Horodatage"].tolist()[1:]  # Getting rid of the first element which is an empty list
    # Retrieving the sleep stage of each epoch
    get_stages = df["Sommeil"].tolist()[1:]  # Getting rid of the first element which is an empty list
    # ------------- detecting missing epochs and completing sleep stages with invalid labels -------------
    # initialisation of a new sleep stages list
    sleep_stages = [get_stages[0]]
    for i in range(len(get_eps) - 1):
        # getting time difference between two consecutive epochs. It should be of 30 seconds
        diff = int((get_eps[i + 1] - get_eps[i]).total_seconds())
        # case where we pass midnight
        if diff < 0:
            diff = 3600 * 24 + int((get_eps[i + 1] - get_eps[i]).total_seconds())

        # Deals when there is one or more epochs missing
        if diff > 30:
            if diff == 60:  # in this case, we have only one epoch missing,
                # we can assume in this case that the sleep stage has not changed and we just rewrite the preceding value
                sleep_stages.append(sleep_stages[-1])
                # We add the next sleep stage to the list
                sleep_stages.append(get_stages[i + 1])
            else:
                # If we miss more than one epoch, we add several invalid labels for all missing epochs
                for j in range(int(diff / 30) - 1):
                    sleep_stages.append("Invalid")
                # We add the next sleep stage to the list
                sleep_stages.append(get_stages[i + 1])
        else:
            # We add the next sleep stage to the list
            sleep_stages.append(get_stages[i + 1])

    # ------------- finding the first sleep onset and the last awakening -------------
    flag_endor = False
    flag_rev = False
    labs = ["Éveil", "Invalid", np.nan, "eveil"]  # different cases where the patient is considered awake
    for i in range(len(sleep_stages)):
        # detecting first sleep onset
        if not labs.__contains__(sleep_stages[i]):
            if not flag_endor:
                ind_deb = i
                flag_endor = True
        # detecting last asleep moment
        if not labs.__contains__(sleep_stages[-i - 1]):

            if not flag_rev:
                ind_fin = i
                flag_rev = True
    # assessing time of sleep onset
    deb = get_eps[0] + timedelta(seconds=ind_deb * 30)
    # assessing time of the final awakening
    fin = get_eps[-1] - timedelta(seconds=ind_fin * 30)
    # Storing results for future use
    if ind_fin > 0:
        sleep_stages_cropped = sleep_stages[ind_deb:-ind_fin]
    else:  # we go until the end of slee_stages
        sleep_stages_cropped = sleep_stages[ind_deb:]
    return sleep_stages_cropped, deb.time(), fin.time()



root_folder = os.path.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.path.sep + "notebooks" + os.path.sep + "data" + os.path.sep
csv_sleep_staging = os.listdir(root_folder + "sleep_staging" + os.path.sep)
all_s_stages = {}
for file in csv_sleep_staging:
    df = pd.read_excel(root_folder + "sleep_staging" + os.path.sep + file, sep=";", encoding='latin-1')
    # print(df)
    sleep_stages, deb, fin = process_csv(df)
    all_s_stages[file[:-4] + ".edf"] = (sleep_stages, deb, fin)

df_to_save = pd.DataFrame(all_s_stages)
df_to_save.to_pickle(root_folder + "sleep_stages.pkl")

