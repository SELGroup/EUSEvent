import numpy as np
import pandas as pd
from os.path import exists
from utils import preprocess_sentence, SBERT_embed
import pickle

def get_embedding(df,name):
    data_path = f'/home/lipu/dataset_streamSED/{name}/'
    if not exists(data_path+'embeddings.pkl'):
        processed_text = [preprocess_sentence(s) for s in
                          df['text'].values]  # hastags are kept (with '#' removed). RTs are removed.
        if name == "2012":
            embeddings = SBERT_embed(processed_text, language='English')
        else:
            embeddings = SBERT_embed(processed_text, language='French')
        with open(data_path+'embeddings.pkl', 'wb') as fp:
            pickle.dump(embeddings, fp)
        print('SBERT embeddings stored.')




def preprocess(name):
    if name =="2012":
        p_part1 = '/home/lipu/smed/datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = '/home/lipu/smed/datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part2.npy'
        df_np_part1 = np.load(p_part1, allow_pickle=True)
        df_np_part2 = np.load(p_part2, allow_pickle=True)
        df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
        df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
            "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities",
            "words", "filtered_words", "sampled_words"])
    else:
        df_np = np.load('/home/lipu/smed/datasets/Twitter_2018/All_French.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
                   "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"])



    df = df.sort_values(by='created_at').reset_index()
    df['date'] = [d.date() for d in df['created_at']]
    distinct_dates = df.date.unique()

    ini_df = df.loc[df['date'].isin(distinct_dates[7:])]  # fit top 7 dates
    # ini_df = df
    datacount = []
    for d in distinct_dates:
        datacount.append(len(ini_df.loc[ini_df['date'].isin([d])]))
    datacount = datacount[6:]
    np.save(f"/home/lipu/dataset_streamSED/{name}/datacount.npy", datacount)
    ini_df_np = ini_df.to_numpy()
    np.save(f"/home/lipu/dataset_streamSED/{name}/df.npy", ini_df_np)
    get_embedding(ini_df,name)
    


if __name__ == "__main__":
    for name in ["2018"]:
        preprocess(name)
    # preprocess_event2018()
