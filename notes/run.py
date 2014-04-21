from admm.results import reconstruct_events
import scipy.io as sio
import sys
from optparse import OptionParser

dictionary = sio.loadmat("/home/x/dictionary.mat")["dictionary"]

if __name__=="__main__":
    parser = OptionParser()

    parser.add_option("-s", "--start", dest="start",
                  help="Starting Frame",)
    parser.add_option("-e", "--end", dest="end",
                  help="Ending Frame")

    (options, args) = parser.parse_args()

    start, end = int(options.start), int(options.end)

    reconstruct_events(dictionary=dictionary, start=start, end=end)
