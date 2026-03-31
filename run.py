import sys
import logging

from bigclust2 import main


if __name__ == "__main__":
    if "--debug" in sys.argv:
        # Set bigclust2 package loggers to DEBUG
        logging.getLogger("bigclust2").setLevel(logging.DEBUG)

    main()