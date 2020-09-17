import argparse as ap
import h5py as h5
import io

from glob import glob
from numpy import array
from os import path
from PIL import Image

def main():

    parser = ap.ArgumentParser(description="Save JPGs from HDF5 files",
                                usage="%(prog)s <options>")

    parser.add_argument("-d", "--directory",
                        help="HDF5 file directory",
                        required=True,
                        type=str)

    arguments = parser.parse_args()

    for ifile in sorted(glob(path.join(arguments.directory, "*.hdf5"))):

        foo = 1

        with h5.File(ifile, 'r') as h5f:

            plot_dataset = h5f["/cand/detection/plot/jpg"]
            im = Image.open(io.BytesIO(plot_dataset[:]))
            im.save(path.join(path.dirname(ifile), h5f["/cand/detection/plot"].attrs["plot_name"]))

if __name__ == "__main__":
    main()