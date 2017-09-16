import config_parser
import extractor

conf = config_parser.ParseConfig()


class Tag(object):
    def __init__(self):
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.data_extractor = extractor.DataExtractor(self.data_set_location)

    def get_combined_data(self):
        print "Obtain all relevant data with respect to this model"

    def get_weighted_tags_for_model(self, model):
        print "Obtain the weighted tags for the model passed as input"
