import os
import numpy as np
import glob
import csv
import scipy.io as sio
from admm.lasso import lasso_admm, _update_dict, init_dictionary
from sklearn.utils import check_random_state
from sklearn.decomposition import MiniBatchDictionaryLearning


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


def reconstruct_events(event_folder=None,output_folder=None, dictionary=None, components=100, alpha=1, start=0, end=np.inf, actions=[], random_state=None):
    if event_folder is None:
        event_folder=os.path.abspath('/share/storage/vision/subway/features/')

    if output_folder is None:
        output_folder=os.path.abspath('/share/storage/vision/subway/reconstructed/')
   
    events_files = get_events_files(event_folder)
 
    event_counter = 0
        
    results = []

    # Check which actions occur on each event
    for pth, events_start, events_end in events_files:
        data = sio.loadmat(pth)
        events = data["events"]

        # Avoid calculating files outside the window of interest           
        if events_end <= start or events_start >= end:
            print "Skipping %s" % pth
            continue
        else:
            print "Processing %s" % pth

        intercept = None
        deviation = None
        dico = MiniBatchDictionaryLearning(n_components=components, alpha=alpha, n_iter=100)

        for event in events:
            ((x,), (y,), (t,)), event_cuboids, event_descriptors, event_cuboid_locations, event_adjacency = event
            if (t >= start - 40) and (t <= end + 40):
                sort_order = np.argsort(event_descriptors[:,2])
                X = event_descriptors[sort_order, :]

                if intercept is None:
                    intercept = np.mean(X, axis=0)

                original  = X - intercept

                if deviation is None:
                    deviation = np.std(original, axis=0)

                original /= deviation

                dictionary = dico.fit(original).components_

                dico.set_params(transform_algorithm='lars', transform_n_nonzero_coefs=5)

                code = dico.transform(original)
                error = (original - np.dot(code, dictionary)) ** 2

                results.append(((x, y, t), code, error))

                event_counter += 1
                    
    result_pth = os.path.join(output_folder, "reconstructed_events_clip-%s-%s.mat" % (start, end))

    sio.savemat(result_pth, {'results': results, 'start': start, 'end': end, 'actions': actions})
    print "%s events saved in '%s'" % (event_counter, result_pth)
    return dictionary


def process_events(events_files, dictionary=None, train=False, random_state=None):
    """Extract event information from a .mat file.
    
    An event is a list of tuples, with each tuple containing the following variables:
    
        event_locations
        event_cuboids
        event_descriptors
        event_cuboid_locations
        event_adjacency
    """

    print events_files


    for start, end, actions in get_ground_truth():
       print "Found %s events in clip starting on %s and ending on %s with actions %s" % (event_counter, start, end, actions)

    return dictionary


def process_reconstructed(folder_path='/share/storage/vision/subway/reconstructed/'):
    """Gather performance statistics from reconstructed events.
    """ 
    matfiles = sorted(glob.glob(os.path.join(folder_path, '*.mat')))
    print "Getting files from: ", folder_path

    all_events = None

    for pth in matfiles:
        print "Processing file:", pth
        data = sio.loadmat(pth)
        (start,) = data["start"]
        actions = data["actions"]
        (end, ) = data["end"]
        results = data["results"]
        print "Starts on ", start, " ends on ", end, " with ", len(results), " results"

        events_dtype = [('x', np.uint8), ('y', np.uint8), ('frame', np.uint16), ('error', np.float32), ('sparsity', np.float32)]
 
        # Create an array to hold x, y, zm  max error and sparsity for each event
        events = np.zeros((len(results), 5), dtype=events_dtype)

        for n, result in enumerate(results):

            position, code, error = result
            # This unpacking looks funny because the .mat file write/read adds another tuple wrapping the object
            ((x,y,z),) = position

            # Take the top 5% higher errors and average them.
            max_error_avg = np.average(error, weights=(error > np.percentile(error, 95)))
            # Measure how sparse the code is.
            sparsity = np.sum(code[:] != 0)*1.0 / code.size

            events[n,:] = x, y, z, max_error_avg, sparsity
        
        if all_events is not None:
            all_events= np.concatenate((all_events, events), axis=0)
        else:
            all_events = events

    # Sort by the z (time) index.
    sort_order = np.argsort(all_events[:,2], axis=0)
    
    return np.unique(all_events[sort_order, :])
