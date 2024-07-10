from Game_SE import GAME_SE
from utils import evaluate, decode
from datetime import datetime
import numpy as np
import pickle
import pandas as pd
from itertools import combinations

def get_att_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))



def search_stable_point(corr_matrix,game):
    step = 0.01
    step_set =  np.arange(0.3, 0.6, step)

    all_1dSEs = []
    for s in step_set:
        matrix = corr_matrix.copy()
        mask = matrix - round(s, 2) <= 1e-8
        matrix[mask] = 0
        s_SE =  game.SE_1D(matrix)
        all_1dSEs.append(s_SE)


    average = sum(all_1dSEs) / len(all_1dSEs)
    nearest = min(all_1dSEs, key=lambda x: abs(x - average))
    point_a = round(step_set[all_1dSEs.index(nearest)], 2)

    # print("\npoint via average all SE:", point_a)
    return point_a

def fit_edge(matrix, threshold,att):
    att_indices = np.array(get_att_edges(att))
    mask =  matrix - threshold <= 1e-8
    row_indices = np.arange(matrix.shape[0])
    max_indices = np.argmax(matrix, axis=1)

    save_indices_1 = np.concatenate((row_indices, max_indices, att_indices[:,0], att_indices[:,1]))
    save_indices_2 = np.concatenate((max_indices, row_indices, att_indices[:,1], att_indices[:,0]))
    mask[save_indices_1, save_indices_2] = False
    matrix[mask] = 0
    matrix[matrix <= 1e-8] = 0

    return matrix

def print_scores(scores):
    line = [' ' * 4] + [f'   M{i:02d} ' for i in range(1,len(scores)+1)]
    print("".join(line))

    score_names = ['NMI', 'AMI', 'ARI']
    for n in score_names:
        line = [f'{n} '] + [f'  {s[n]:1.3f}' for s in scores]
        print("".join(line))
    print('\n', flush=True)

    # line = [' ' * 3]+ [f'   M{i:02d}' for i in range(1, len(scores)+1)]
    # print("".join(line))
    #
    # score_names = ['NMI', 'AMI', 'ARI']
    # for n in score_names:
    #     line = [f'{n:3}'] + [f'  {s[n]:4.2f}' for s in scores]
    #     print("".join(line))
    # print('\n', flush=True)

def split_value_into_chunks(total_value, chunk_size=2000):
    full_chunks = total_value // chunk_size

    remainder = total_value % chunk_size

    result_list = [chunk_size] * full_chunks

    if remainder > 0:
        result_list.append(int(remainder))
    return result_list

def run(name,save_path,fix_length):
    if name == 'Events2012':
        Columns = ["original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
                    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities",
                    "words", "filtered_words", "sampled_words", "date"]
    if name == 'Events2018':
        Columns = ["original_index", "tweet_id", "user_id", "text", "time", "event_id", "user_mentions", \
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words", "date"]

    # datacount = np.load(save_path+"datacount.npy",allow_pickle=True)
    datacount = np.load(save_path+"all_datacount.npy",allow_pickle=True)
    datacount[0] = 0

    if fix_length:
        msg_count = np.sum(datacount)
        datacount = split_value_into_chunks(msg_count)

    block_range = []
    s = 0
    for i,c in enumerate(datacount):
        if i == 0 : continue
        block_range.append([s,s+c])
        s +=c
    # with open(save_path+"embeddings.pkl", 'rb') as f:
    with open(save_path+"all_embeddings.pkl", 'rb') as f:
        all_embeddings = pickle.load(f)
    # df_np = np.load(save_path+"df.npy", allow_pickle=True)
    df_np = np.load(save_path+"all_df.npy", allow_pickle=True)
    all_df = pd.DataFrame(data=df_np, columns=Columns)
    num = 0
    all_detection_time = 0.
    all_cos = []
    game = GAME_SE()

    for block in range(1,len(datacount)):
        print('====================================================')
        print('block: ', block)
        # print(datetime.now().strftime("%H:%M:%S"))
        # start = datetime.now().strftime("%H:%M:%S")
        num += datacount[block]
        df = all_df.iloc[0:num]


        all_node_features = [[str(u)] + \
                             [str(each) for each in um] + \
                             [h.lower() for h in hs] + \
                             e  \
                             for u, um, hs, e,  in \
                             zip(df['user_id'], df['user_mentions'], df['hashtags'], df['entities'])]

        embeddings = all_embeddings[:num]

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        point = search_stable_point(corr_matrix,game)

        all_cos.append(point)
        print(f"we use Cos> {point}")
        corr_matrix = fit_edge(corr_matrix, point, all_node_features)

        # print(f"Edge num: {int(np.count_nonzero(corr_matrix[corr_matrix>=0])/2)}")
        print(datetime.now().strftime("%H:%M:%S"))
        start = datetime.now().strftime("%H:%M:%S")



        game.partition_init(corr_matrix, datacount[block])
        game.gaming(max_item=10,patience=1, fix_window=2 ,verbose=False)
        division = game.get_clusters()

        end = datetime.now().strftime("%H:%M:%S")

        start_time = datetime.strptime(start, "%H:%M:%S")
        end_time = datetime.strptime(end, "%H:%M:%S")
        time_diff = end_time - start_time
        seconds = time_diff.total_seconds()

        print(datetime.now().strftime("%H:%M:%S"))
        print(f"time consuming:{seconds}")
        all_detection_time +=seconds
        prediction = decode(division)
        local_score = []
        labels_true = df['event_id'].tolist()
        all_nmi, all_ami, all_ari = evaluate(labels_true, prediction)
        print(f'ALL Block, nmi:{all_nmi}, ami:{all_ami}, ari:{all_ari}')
        for blk in range(block):
            pred = prediction[block_range[blk][0]:block_range[blk][1]]
            true = labels_true[block_range[blk][0]:block_range[blk][1]]
            # n_clusters = len(list(set(true)))
            # print(f'blk{blk+1} ,#gt: {n_clusters}, $pred: {len(set(pred))}')
            nmi, ami, ari = evaluate(true, pred)
            local_score.append({'NMI': nmi, 'AMI': ami, 'ARI': ari})
        print_scores(local_score)

    print(f"all_time: {all_detection_time}")
    print(f"all_cos: {all_cos}")
def run_incre(name,save_path,fix_length):
    if name == 'Events2012':
        Columns = ["original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
                    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities",
                    "words", "filtered_words", "sampled_words", "date"]
    if name == 'Events2018':
        Columns = ["original_index", "tweet_id", "user_id", "text", "time", "event_id", "user_mentions", \
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words", "date"]

    datacount = np.load(save_path+"all_datacount.npy",allow_pickle=True)

    datacount[0] = 0
    if fix_length:
        msg_count = np.sum(datacount)
        datacount = split_value_into_chunks(msg_count)

    block_range = []
    s = 0
    for i, c in enumerate(datacount):
        if i == 0: continue
        block_range.append([s, s + c])
        s += c
    scores = []
    with open(save_path+"all_embeddings.pkl", 'rb') as f:
        all_embeddings = pickle.load(f)
    # df_np = np.load(save_path+"df.npy", allow_pickle=True)
    df_np = np.load(save_path+"all_df.npy", allow_pickle=True)
    all_df = pd.DataFrame(data=df_np, columns=Columns)
    num = 0


    for block in range(1,len(datacount)):
        game = GAME_SE()
        print('====================================================')
        print('block: ', block)
        first_ind = num
        num += datacount[block]
        df = all_df.iloc[first_ind:num]


        all_node_features = [[str(u)] + \
                             [str(each) for each in um] + \
                             [h.lower() for h in hs] + \
                             e  \
                             for u, um, hs, e,  in \
                             zip(df['user_id'], df['user_mentions'], df['hashtags'], df['entities'])]

        embeddings = all_embeddings[first_ind :num]

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        point = search_stable_point(corr_matrix,game)

        print(f"we use Cos> {point}")
        corr_matrix = fit_edge(corr_matrix, point, all_node_features)

        print(f"Edge num: {int(np.count_nonzero(corr_matrix[corr_matrix>=0])/2)}")
        print(datetime.now().strftime("%H:%M:%S"))
        start = datetime.now().strftime("%H:%M:%S")



        game.partition_init(corr_matrix, datacount[block])
        game.gaming(max_item=10,patience=1, fix_window=2 ,verbose=False)
        division = game.get_clusters()

        end = datetime.now().strftime("%H:%M:%S")

        start_time = datetime.strptime(start, "%H:%M:%S")
        end_time = datetime.strptime(end, "%H:%M:%S")
        time_diff = end_time - start_time
        seconds = time_diff.total_seconds()

        print(datetime.now().strftime("%H:%M:%S"))
        print(f"time consuming:{seconds}")
        prediction = decode(division)

        prediction = prediction

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate(labels_true, prediction)
        scores.append({'NMI':nmi,'AMI':ami,'ARI':ari})
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)

    print_scores(scores)

def read_ture_cluster(path):
    cmty = []
    keep_nodes = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            nodes = line.rstrip('\n').split('\t')
            nodes = [int(n) for n in nodes]
            if len(nodes)>2:
                cmty.append(nodes)
                keep_nodes += nodes
    return cmty, set(keep_nodes)

all_cos= [0.45, 0.47, 0.46, 0.46, 0.46, 0.46, 0.45, 0.46, 0.45, 0.45, 0.44, 0.45, 0.45, 0.44, 0.44, 0.44, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
if __name__ == "__main__":


    set_name= "Events2018" #or Events2018

    path = '/home/lipu/dataset_streamSED/2018/' #or '/home/lipu/his_data/E_2018/'

    fix_length = True
    run(set_name, path, fix_length)
    # run_incre(set_name, path,fix_length)



