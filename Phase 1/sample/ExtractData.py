import logging
import os

import pandas as pd

import ParseConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ExtractData(object):
    def __init__(self, data_set_loc):
        self.data_set_loc = data_set_loc

    def data_extractor(self, file_name):
        file_loc = os.path.join(self.data_set_loc, file_name)
        log.info("Loading data from %s to dataframe" % file_loc)
        data_frame = pd.DataFrame.from_csv(file_loc)
        log.info("%s dataframe loaded with the index = %s, columns = %s and rows =%s" % (
        file_name, data_frame.index.name, str(data_frame.columns.values), len(data_frame.index)))
        return data_frame


if __name__ == "__main__":
    conf = ParseConfig()
    data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
    e_d = ExtractData(data_set_loc)
    e_d.data_extractor("mlmovies.csv")
