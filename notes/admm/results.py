import os
import numpy as np
import glob
import csv
import scipy.io as sio
from admm.lasso import lasso_admm, _update_dict, init_dictionary
from sklearn.utils import check_random_state


ACTIONS = ["nopay", "loitering","wrong_direction","interaction", "etc"]

def get_ground_truth(filename='ground-truth-entrance.csv'):
    """Parse ground truth locations from a CSV file
    """
    csvfile = open(filename, 'r')
    fieldnames = ("clip_info","loitering", "nopay", "wrong_direction","interaction", "etc", "detected_loitering", "detected_no_pay", "detected_wrong", "detected_interaction", "detected_etc", "false_alarm")
    reader = csv.DictReader( csvfile, fieldnames, delimiter='\t',)
    data = [ row for row in reader ]
    # Remove the header
    data = data[1:]

    # Get the frame numbers
    clip_numbers = len(data)

    clips = []

    for item in data:
        clip_info = item["clip_info"].strip()
        clip_number, start, end = clip_info.split(" ")
        actions = []
        for key in ACTIONS:
            if item[key] != "":
                actions.append(key)

        clips.append((int(start), int(end), actions))
    
    return clips


def get_events_files(video_folder=os.path.abspath('features_subway')):
    """Get paths to .mat files and annotate them with start and end frames.
    """
    matfiles = sorted(glob.glob(os.path.join(video_folder, '*.mat')))
    
    events_files = []

    for pth in matfiles:
        data = sio.loadmat(pth)
        events_start = int(data["frame1"])
        events_end = int(data["frame2"])
        events_files.append((pth, events_start, events_end))
    return events_files


def process_events(events_files, dictionary=None, train=False, random_state=None):
    """Extract event information from a .mat file.
    
    An event is a list of tuples, with each tuple containing the following variables:
    
        event_locations
        event_cuboids
        event_descriptors
        event_cuboid_locations
        event_adjacency
    """

    random_state = check_random_state(random_state)

    components = 100
    alpha=1
    
    for start, end, actions in get_ground_truth():
        event_counter = 0
        
        # If training is selected, ignore clips with actions 
        if train and len(actions) > 0:
            continue

        # Check which actions occur on each event
        for pth, events_start, events_end in events_files:

            # Avoid calculating files outside the window of interest           
            if events_end <= start or events_start >= end:
                continue

            data = sio.loadmat(pth)
            events = data["events"]
            results = []

            for event in events:
                ((x,), (y,), (t,)), event_cuboids, event_descriptors, event_cuboid_locations, event_adjacency = event
        
                if (t >= start - 40) and (t <= end + 40):
                    sort_order = np.argsort(event_descriptors[:,2])
                    X = event_descriptors[sort_order, :]
                    
                    if dictionary is  None:
                         dictionary = init_dictionary(X, components)

                    # Update code
                    code_T, __ = lasso_admm(X.T, dictionary.T, gamma=alpha)
                    
                    code = code_T.T

                    mse = ((dictionary * code - X) ** 2).mean(axis=1)

                    results.append((((x, y, t), code, mse))

                    # Update dictionary
                    dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             return_r2=True, random_state=random_state)
                    dictionary = dictionary.T
                    
             result_pth = pth.replace(".mat", "%s-%s-result.mat" % (start, end))

             sio.savemat(result_pth, {'results': results})
             print "Saved %s" % result_pth

        print "Found %s events in clip starting on %s and ending on %s with actions %s" % (event_counter, start, end, actions)

    return dictionary
