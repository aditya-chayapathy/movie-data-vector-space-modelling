import logging

import configparser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ParseConfig(object):
    def __init__(self):
        self.Config = configparser.ConfigParser()
        self.Config.read("path.Ini")

    def config_section_mapper(self, section):
        dict1 = {}
        options = self.Config.options(section)
        for option in options:
            try:
                dict1[option] = self.Config.get(section, option)
                if dict1[option] == -1:
                    log.debug("skip: %s" % option)
            except:
                log.error("exception on %s!" % option)
                dict1[option] = None
        return dict1
