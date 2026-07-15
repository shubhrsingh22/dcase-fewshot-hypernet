"""Post-processing of predicted events (from the DCASE baseline).

Two variants, matching the baseline repo:

* ``adaptive`` (``post_proc.py``): drop events shorter than 60% of the shortest
  support shot for that recording.
* ``fixed`` (``post_proc_new.py``): drop events shorter than 200 ms. **The
  results on the DCASE page use this fixed variant.**

Use :func:`run_post_processing` to dispatch on ``mode``.
"""

import csv
import os


def post_processing_fixed(evaluation_file, new_evaluation_file, min_dur=0.200):
    """Fixed threshold: remove events shorter than ``min_dur`` seconds (200 ms)."""
    with open(evaluation_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        results = list(reader)

    new_results = [["Audiofilename", "Starttime", "Endtime"]]
    for event in results:
        if float(event[2]) - float(event[1]) >= min_dur:
            new_results.append(event)

    with open(new_evaluation_file, "w", newline="") as f:
        csv.writer(f).writerows(new_results)


def run_post_processing(mode, val_path, evaluation_file, new_evaluation_file,
                        n_shots=5, min_dur=0.200, adaptive_frac=0.6):
    if mode == "fixed":
        post_processing_fixed(evaluation_file, new_evaluation_file, min_dur=min_dur)
    else:
        post_processing(val_path, evaluation_file, new_evaluation_file,
                        n_shots=n_shots, frac=adaptive_frac)


def post_processing(val_path, evaluation_file, new_evaluation_file, n_shots=5, frac=0.6):
    dict_duration = {}
    for folder in os.listdir(val_path):
        fdir = os.path.join(val_path, folder)
        if not os.path.isdir(fdir):
            continue
        for file in os.listdir(fdir):
            if file.endswith(".csv"):
                audiofile = file[:-4] + ".wav"
                events = []
                with open(os.path.join(fdir, file)) as csv_file:
                    for row in csv.reader(csv_file, delimiter=","):
                        if row[-1] == "POS" and len(events) < n_shots:
                            events.append(row)
                min_duration = 10000
                for event in events:
                    dur = float(event[2]) - float(event[1])
                    if dur < min_duration:
                        min_duration = dur
                dict_duration[audiofile] = min_duration

    with open(evaluation_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        results = list(reader)

    new_results = [["Audiofilename", "Starttime", "Endtime"]]
    for event in results:
        if event[0] not in dict_duration:
            continue
        min_dur = dict_duration[event[0]]
        if float(event[2]) - float(event[1]) >= frac * min_dur:
            new_results.append(event)

    with open(new_evaluation_file, "w", newline="") as f:
        csv.writer(f).writerows(new_results)
