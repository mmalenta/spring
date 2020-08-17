import argparse as ap
import logging

from pipeline.pipeline import Pipeline

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
                        choices=["iqrm", "zerodm", "threshold", "mask", "multibeam",
                                    "plot", "archive"])

    arguments = parser.parse_args()

    configuration = {
        "base_directory": arguments.directory,
        "num_watchers": arguments.watchers,
        "modules": arguments.modules
    }

    logger.setLevel(getattr(logging, arguments.log.upper()))

    # Might set a separate file handler for warning messages
    cl_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s", datefmt="%a %Y-%m-%d %H:%M:%S")
    cl_handler.setLevel(getattr(logging, arguments.log.upper()))
    cl_handler.setFormatter(formatter)
    logger.addHandler(cl_handler)

    pipeline = Pipeline(configuration)
    pipeline.run()

if __name__ == "__main__":
    main()