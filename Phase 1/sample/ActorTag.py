import logging
import os
from collections import Counter

import pandas as pd

import config
import extractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config.ParseConfig()

class ActorTag(object):
    def __init__(self):
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.extract_data = extractor.ExtractData(self.data_set_location)

    def get_movie_actor_data(self):
        movie_actor = self.extract_data.data_extractor("movie-actor.csv")
        return movie_actor.reset_index()

    def get_mltags_data(self):
        mltags = self.extract_data.data_extractor("mltags.csv")
        return mltags.reset_index()

    def get_genome_tags_data(self):
        genome_tags = self.extract_data.data_extractor("genome-tags.csv")
        return genome_tags.reset_index()

    def get_imdb_actor_info_data(self):
        imdb_actor_info = self.extract_data.data_extractor("imdb-actor-info.csv")
        return imdb_actor_info.reset_index()

    def assign_tf_weight(self, data_frame):
        counter = Counter()
        for each in data_frame.tag:
            counter[each] += 1
        total = sum(counter.values())
        for each in counter:
            counter[each] = (counter[each] / total) * 100
        return dict(counter)

    def assign_weight(self, data_frame):
        data_frame = data_frame.drop(["userid", "movieid"], axis=1)
        actor_tag_dict = {}
        tf_weight_dict = self.assign_tf_weight(data_frame)
        groupby_actor_dict = data_frame.groupby("name")
        for actorid, tag_df in groupby_actor_dict:
            if tag_df.tag.isnull().all():
                actor_tag_dict[actorid] = "No tag data found"
            else:
                tag_df_len = len(tag_df.index)
                tag_df = tag_df.sort_values("timestamp", ascending=False)
                tag_df = tag_df.reset_index()
                tag_df["value"] = pd.Series(
                    [(((index / tag_df_len) * 100) + tf_weight_dict[tag] - rank) for index, tag, rank
                     in zip(tag_df.index, tag_df.tag, tag_df.actor_movie_rank)], index=tag_df.index)
                tag_df["total"] = tag_df.groupby(['tag'])['value'].transform('sum')
                tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
                actor_tag_dict[actorid] = dict(zip(tag_df.tag, tag_df.total))
        actor_tag_data_frame = pd.DataFrame.from_dict(list(actor_tag_dict.items()))
        return actor_tag_data_frame

    def merge_movie_actor_and_tag(self):
        mov_act = self.get_movie_actor_data()
        ml_tag = self.get_mltags_data()
        genome_tag = self.get_genome_tags_data()
        actor_info = self.get_imdb_actor_info_data()
        actor_movie_info = mov_act.merge(actor_info, how="left", on="actorid")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = actor_movie_info.merge(tag_data_frame, how="left", on="movieid")
        actor_tag_data_frame = self.assign_weight(merged_data_frame)
        actor_tag_data_frame.columns = ['actor_name', 'tags']
        actor_tag_data_frame.to_csv(os.path.join(self.data_set_location, "ACT_TAG.csv"), index=False)


if __name__ == "__main__":
    obj = ActorTag()
    obj.merge_movie_actor_and_tag()
