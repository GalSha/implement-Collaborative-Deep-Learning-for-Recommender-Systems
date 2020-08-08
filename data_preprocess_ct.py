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
with open(r"ctrsr_datasets/citeulike-a/vocabulary.dat") as vocabulary_file:
    vocabulary_size = len(vocabulary_file.readlines())

# find item_size = 16980
with open(r"ctrsr_datasets/citeulike-a/mult.dat") as item_info_file:
    item_size = len(item_info_file.readlines())

# initialize item_infomation_matrix (16980 , 8000)
item_infomation_matrix = np.zeros((item_size, vocabulary_size))

# build item_infomation_matrix
with open(r"ctrsr_datasets/citeulike-a/mult.dat") as item_info_file:
    sentences = item_info_file.readlines()

    for index, sentence in enumerate(sentences):
        words = sentence.strip().split(" ")[1:]
        for word in words:
            vocabulary_index, number = word.split(":")
            item_infomation_matrix[index][int(vocabulary_index)] = number

print("#### build rating matrix citeulike-a")

with open(r"ctrsr_datasets/citeulike-a/users.dat") as rating_file:
    rating_file_new = open("ctr/ratings.dat", "w", encoding="latin1")
    lines = rating_file.readlines()
    for index,line in enumerate(lines):
        items = line.strip().split(" ")
        for item in items:
            rating_file_new.write("{user}::{item}::{rate}::{timestamp}\n"
                                  .format(user=index, item=item, rate=1, timestamp=0))
            rating_file_new.flush()
    rating_file_new.close()

# clean rating
with open(r"ctr/ratings.dat", encoding="latin1") as rating_file:
    clean_rating_file = open("ctr/clean_ratings.dat", "w", encoding="latin1")
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
    with open(r"ctr/clean_ratings.dat", encoding="latin1") as clean_rating_file:
        clean_rating_tr_file = open("ctr/clean_ratings_tr_p{p}.dat".format(p=p), "w", encoding="latin1")
        clean_rating_val_file = open("ctr/clean_ratings_val_p{p}.dat".format(p=p), "w", encoding="latin1")
        clean_rating_neg_file = open("ctr/clean_ratings_neg_p{p}.dat".format(p=p), "w", encoding="latin1")
        lines = clean_rating_file.readlines()
        _user = lines[0].split("\t")[0]
        write_lines = []
        negs = np.arange(item_size)
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
                negs = np.arange(item_size)
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

    rating_matrix = np.zeros((user_size, item_size))
    with open(r"ctr/clean_ratings_tr_p{p}.dat".format(p=p), encoding="latin1") as clean_rating_tr_file:
        lines = clean_rating_tr_file.readlines()
        for line in lines:
            u_i = int(line.split("\t")[0])
            m_i = int(line.split("\t")[1])
            rating_matrix[u_i][m_i] = 1
    with open(r'ctr/rating_matrix_p{p}.pickle'.format(p=p), 'wb') as handle:
        pickle.dump(rating_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'ctr/item_infomation_matrix.pickle', 'wb') as handle:
    pickle.dump(item_infomation_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

from shutil import copyfile

copyfile("ctr/clean_ratings_tr_p1.dat",  "ctr/ctr.hr-ndcg.train.rating")
copyfile("ctr/clean_ratings_val_p1.dat", "ctr/ctr.hr-ndcg.test.rating")
copyfile("ctr/clean_ratings_neg_p1.dat", "ctr/ctr.hr-ndcg.test.negative")

copyfile("ctr/clean_ratings_tr_p-10.dat",  "ctr/ctr.precision-recall.train.rating")
copyfile("ctr/clean_ratings_val_p-10.dat", "ctr/ctr.precision-recall.test.rating")
copyfile("ctr/clean_ratings_neg_p-10.dat", "ctr/ctr.precision-recall.test.negative")
