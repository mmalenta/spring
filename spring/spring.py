import argparse as ap
import asyncio
import logging
import signal

from functools import partial
from os import path

from sppipeline.pipeline import Pipeline

logger = logging.getLogger()

def main():

    """
    Main entrypoint to the post-processing pipeline

    """


    parser = ap.ArgumentParser(description="MeerTRAP real-time post-processing \
                                            pipeline",
                                usage="%(prog)s [options]",
                                epilog="For any bugs or feature requests, \
                                        please start an issue at \
                                        https://github.com/mmalenta/spring")

    parser.add_argument("-l", "--log",
                        help="Log level", 
                        required=False,
                        type=str,
                        choices=["debug", "info", "warn"],
                        default="debug")

    parser.add_argument("-d", "--directory",
                        help="Base data directory to watch",
                        required=True,
                        type=str)

    parser.add_argument("-w", "--watchers",
                        help="Number of UTC directories to watch for new \
                                candidate files",
                        required=False,
                        type=int,
                        default=3)

    parser.add_argument("-m", "--modules",
                        help="Optional modules to enable",
                        required=False,
                        # If used, require at least one extra module
                        nargs="+",
                        type=str,
                        choices=["iqrm", "zerodm", "threshold", "mask",
                                    "multibeam", "plot", "archive"])

    parser.add_argument("--model", help="Model name and model directory",
                        required=True,
                        nargs=2,
                        type=str)

    # Plots selection is input in the format:
    # [plot1][size[1],[plot2][size2][:][plot3]
    # Plots separated by a comma appear in the same row
    # Their sizes are a fraction of the full row width and can run
    # between 0.0 and 1.0 (if they do not sum up to 1.0, they are scaled
    # accordingly). Sizes are optional - if not included, all the plots
    # in the row will have the same width
    # Colon is used to indicate a new row
    # Row heights are currently set to be equal
    parser.add_argument("-p", "--plots", help="Plots to enable",
                        required=False,
                        # If used, require at least one type of plot
                        type=str)

    arguments = parser.parse_args()

    logger.setLevel(getattr(logging, arguments.log.upper()))
    # Might set a separate file handler for warning messages
    cl_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s", datefmt="%a %Y-%m-%d %H:%M:%S")
    cl_handler.setLevel(getattr(logging, arguments.log.upper()))
    cl_handler.setFormatter(formatter)
    logger.addHandler(cl_handler)

    chosen_model = arguments.model[0].upper()
    if (chosen_model not in
            ["NET1_32_64", "NET1_64_128", "NET1_128_256", "NET2", "NET3"]):
        
        logger.warning(f"Unrecognised model '{chosen_model}'! " 
                        + "Will default to NET3!")
        chosen_model = "NET3"

    chosen_dir = arguments.model[1]
    if not path.isdir(chosen_dir):
        logger.error(f"Model directory '{chosen_dir}' does not exist! "
                        + "Will quit now!")
        # So this is not ideal, but will do for now
        exit()

    # Separate into list of lists of tuples (one inner list per row)
    plots = [ [(cell[0], float(cell[1:])) for cell in row.split(",")]
                for row in arguments.plots.split(":") ]

    configuration = {
        "base_directory": arguments.directory,
        "num_watchers": arguments.watchers,
        "modules": arguments.modules,
        "model": [chosen_model, chosen_dir],
        "plots": plots,
    }

    pipeline = Pipeline(configuration)
    loop = asyncio.get_event_loop()
    # Handle CTRL + C
    loop.add_signal_handler(getattr(signal, 'SIGINT'),
                                        partial(pipeline.stop, loop))    
    loop.create_task(pipeline.run(loop))

    try:
        loop.run_forever()
        logger.info("Pipeline finished processing")
    finally:
        loop.close()
        logger.info("Processing closed successfully")

if __name__ == "__main__":
    main()