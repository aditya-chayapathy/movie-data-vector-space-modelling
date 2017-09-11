import logging
import os

import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ExtractData(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def data_extractor(self, file_name):
        file_loc = os.path.join(self.file_path, file_name)
        data_frame = pd.read_csv(file_loc)
        return data_frame


if __name__ == "__main__":
    conf = config.ParseConfig()
    data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
    extract_data = ExtractData(data_set_location)
    data_frame = extract_data.data_extractor("mlmovies.csv")
    log.info("File columns for mlmovies.csv")
    log.info("Columns = %s" % (data_frame.columns.values))
