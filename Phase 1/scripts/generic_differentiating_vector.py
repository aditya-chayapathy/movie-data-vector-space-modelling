import config_parser
import extractor

conf = config_parser.ParseConfig()


class GenericDifferentiateTag(object):
    def __init__(self, object_id1, object_id2):
        self.object_id1 = object_id1
        self.object_id2 = object_id2
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.data_extractor = extractor.DataExtractor(self.data_set_location)

    def get_combined_data_for_objects(self):
        print "Obtain all relevant data with respect to this model"

    def get_weighted_tags_for_model(self, model):
        print "Obtain the weighted tags for the model passed as input"

    def get_combined_data_for_object(self, object_id):
        print "Obtain all relevant data with respect to this model and object"

    def get_model_value(self, movie_id, tag_of_movie, model):
        print "Obtain the row value for the parameters passed as input"
