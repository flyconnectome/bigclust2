import sys
import logging

from bigclust2 import main


if __name__ == "__main__":
    if "--debug" in sys.argv:
        logging.basicConfig(level=logging.DEBUG)

    main()