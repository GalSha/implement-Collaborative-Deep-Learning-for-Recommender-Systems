import numpy as np
import pickle
import tensorflow as tf

#init random seed
np.random.seed(5)
print(tf.__version__)
print("#### load matrix from pickle")
print()
print("#### build item information matrix of citeulike-a by bag of word")
# find vocabulary_size = 8000
tag_id_to_index = {}
with open(r"last.fm/tags.dat", encoding="latin1") as tag_file:
    lines = tag_file.readlines()[1:]
    tag_size = len(lines)
    for index,line in enumerate(lines):
        tag_id = int(line.strip().split("\t")[0])
        tag_id_to_index[tag_id] = index

# find item_size = 16980
artist_id_to_index = {}
with open(r"last.fm/artists.dat", encoding="latin1") as artists_file:
    lines = artists_file.readlines()[1:]
    artist_size = len(lines)
    for index,line in enumerate(lines):
        artist_id = int(line.strip().split("\t")[0])
        artist_id_to_index[artist_id] = index

# initialize item_infomation_matrix (16980 , 8000)
artist_infomation_matrix = np.zeros((artist_size, tag_size))

# build item_infomation_matrix
with open(r"last.fm/user_taggedartists.dat", encoding="latin1") as artist_tag_file:
    lines = artist_tag_file.readlines()[1:]

    for line in lines:
        artist_id = int(line.strip().split("\t")[1])
        tag_id = int(line.strip().split("\t")[2])
        if artist_id in artist_id_to_index:
            artist_infomation_matrix[artist_id_to_index[artist_id]][tag_id_to_index[tag_id]] += 1

print("#### build rating matrix last.fm")


with open(r"last.fm/user_artists.dat", encoding="latin1") as rating_file:
    rating_file_new = open("last.fm/ratings.dat", "w", encoding="latin1")
    lines = rating_file.readlines()[1:]
    for line in lines:
        user = int(line.strip().split("\t")[0])
        artist_id = int(line.strip().split("\t")[1])
        if artist_id in artist_id_to_index:
            rating_file_new.write("{user}::{artist}::{rate}::{timestamp}\n"
                                    .format(user=user, artist=artist_id_to_index[artist_id], rate=1, timestamp=0))
            rating_file_new.flush()
    rating_file_new.close()

# clean rating
with open(r"last.fm/ratings.dat", encoding="latin1") as rating_file:
    clean_rating_file = open("last.fm/clean_ratings.dat", "w", encoding="latin1")
    _user = "x"
    _user_count = 0
    write_lines = []
    lines = rating_file.readlines()
    _user_index = 0
    for line in lines:
        cur_user = line.split("::")[0]
        if cur_user == _user:
            _user_count += 1

            split_line = line.split("::")
            split_line[0] = str(_user_index)
            line = split_line[0] + "\t" + split_line[1] + "\t" + split_line[2] + "\t" + \
                   split_line[3]
            write_lines += [line]
        else:
            print(cur_user, _user, _user_count)
            if _user_count >= 20:
                _user_index += 1
                np.random.shuffle(write_lines)
                clean_rating_file.write("".join(write_lines))
                clean_rating_file.flush()
            else:
                None
            _user = cur_user
            _user_count = 1
            split_line = line.split("::")
            split_line[0] = str(_user_index)
            line = split_line[0] + "\t" + split_line[1] + "\t" + split_line[2] + "\t" + \
                   split_line[3]
            write_lines = [line]
    if _user_count >= 20:
        np.random.shuffle(write_lines)
        clean_rating_file.write("".join(write_lines))
        clean_rating_file.flush()
    clean_rating_file.close()
    user_size = _user_index + 1

# split train \ val rating
for p in [1, -10]:
    with open(r"last.fm/clean_ratings.dat", encoding="latin1") as clean_rating_file:
        clean_rating_tr_file = open("last.fm/clean_ratings_tr_p{p}.dat".format(p=p), "w", encoding="latin1")
        clean_rating_val_file = open("last.fm/clean_ratings_val_p{p}.dat".format(p=p), "w", encoding="latin1")
        clean_rating_neg_file = open("last.fm/clean_ratings_neg_p{p}.dat".format(p=p), "w", encoding="latin1")
        lines = clean_rating_file.readlines()
        _user = lines[0].split("\t")[0]
        write_lines = []
        negs = np.arange(artist_size)
        np.random.shuffle(negs)
        negs = list(negs)
        for line in lines:
            cur_user = line.split("\t")[0]
            if cur_user == _user:
                if int(line.split("\t")[1]) in negs:
                    write_lines += [line]
                    negs.remove(int(line.split("\t")[1]))
            else:
                print(cur_user, _user)
                neg_line = "".join(["{}\t".format(cur_user)] + ["{}\t".format(neg) for neg in negs[:99]] + [
                    "{}\n".format(negs[100])])
                clean_rating_tr_file.write("".join(write_lines[:-p]))
                clean_rating_tr_file.flush()
                clean_rating_val_file.write("".join(write_lines[-p:]))
                clean_rating_val_file.flush()
                clean_rating_neg_file.write(neg_line)
                clean_rating_neg_file.flush()
                _user = cur_user
                write_lines = [line]
                negs = np.arange(artist_size)
                np.random.shuffle(negs)
                negs = list(negs)
        clean_rating_tr_file.write("".join(write_lines[:-p]))
        clean_rating_tr_file.flush()
        clean_rating_tr_file.close()
        clean_rating_val_file.write("".join(write_lines[-p:]))
        clean_rating_val_file.flush()
        clean_rating_val_file.close()
        neg_line = "".join(
            ["{}\t".format(cur_user)] + ["{}\t".format(neg) for neg in negs[:99]] + ["{}\n".format(negs[100])])
        clean_rating_neg_file.write(neg_line)
        clean_rating_neg_file.flush()
        clean_rating_neg_file.close()

    rating_matrix = np.zeros((user_size, artist_size))
    with open(r"last.fm/clean_ratings_tr_p{p}.dat".format(p=p), encoding="latin1") as clean_rating_tr_file:
        lines = clean_rating_tr_file.readlines()
        for line in lines:
            u_i = int(line.split("\t")[0])
            m_i = int(line.split("\t")[1])
            rating_matrix[u_i][m_i] = 1
    with open(r'last.fm/rating_matrix_p{p}.pickle'.format(p=p), 'wb') as handle:
        pickle.dump(rating_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'last.fm/item_infomation_matrix.pickle', 'wb') as handle:
    pickle.dump(artist_infomation_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

from shutil import copyfile

copyfile("last.fm/clean_ratings_tr_p1.dat",  "last.fm/last.fm.hr-ndcg.train.rating")
copyfile("last.fm/clean_ratings_val_p1.dat", "last.fm/last.fm.hr-ndcg.test.rating")
copyfile("last.fm/clean_ratings_neg_p1.dat", "last.fm/last.fm.hr-ndcg.test.negative")

copyfile("last.fm/clean_ratings_tr_p-10.dat",  "last.fm/last.fm.precision-recall.train.rating")
copyfile("last.fm/clean_ratings_val_p-10.dat", "last.fm/last.fm.precision-recall.test.rating")
copyfile("last.fm/clean_ratings_neg_p-10.dat", "last.fm/last.fm.precision-recall.test.negative")
