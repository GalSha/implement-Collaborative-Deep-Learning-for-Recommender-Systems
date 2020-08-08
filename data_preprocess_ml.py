import numpy as np
import pickle

#init random seed
np.random.seed(5)

genres = [
	"Action",
	"Adventure",
	"Animation",
	"Children's",
	"Comedy",
	"Crime",
	"Documentary",
	"Drama",
	"Fantasy",
	"Film-Noir",
	"Horror",
	"Musical",
	"Mystery",
	"Romance",
	"Sci-Fi",
	"Thriller",
	"War",
	"Western",
	]

print("#### load matrix from pickle")
print()
print("#### build movie infomation matrix of movie by genre")
genres_size = len(genres)

print("genres_size=",genres_size)

# find item_size = 16980
movie_size = 3952
print("movie_size=",movie_size)

movie_infomation_matrix = np.zeros((movie_size, genres_size))

# build item_infomation_matrix
with open(r"ml-1m/movies.dat",encoding="latin1") as movie_genres_file:
    movies = movie_genres_file.readlines()

    for movie in movies:
        index = int(movie.split("::")[0])-1
        movie_genres = movie.strip().split("::")[2].split("|")
        for genre in movie_genres:
            movie_infomation_matrix[index][genres.index(genre)] = 1

print("#### build rating matrix ml-1m")

#initialize rating_matrix (5551 , 16980)
import numpy as np

#clean rating
with open(r"ml-1m/ratings.dat",encoding="latin1") as rating_file:
    clean_rating_file = open("ml-1m/clean_ratings.dat","w",encoding="latin1")
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
            line = split_line[0] + "\t" + "{}".format(int(split_line[1])-1) + "\t" + split_line[2] + "\t" + split_line[3]
            write_lines += [line]
        else:
            print(cur_user,_user,_user_count)
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
            line = split_line[0] + "\t" + "{}".format(int(split_line[1])-1) + "\t" + split_line[2] + "\t" + split_line[3]
            write_lines = [line]
    if _user_count >= 20:
        np.random.shuffle(write_lines)
        clean_rating_file.write("".join(write_lines))
        clean_rating_file.flush()
    clean_rating_file.close()
    user_size = _user_index + 1

#split train \ val rating
for p in [1,-10]:
    with open(r"ml-1m/clean_ratings.dat",encoding="latin1") as clean_rating_file:
        clean_rating_tr_file = open("ml-1m/clean_ratings_tr_p{p}.dat".format(p=p),"w",encoding="latin1")
        clean_rating_val_file = open("ml-1m/clean_ratings_val_p{p}.dat".format(p=p),"w",encoding="latin1")
        clean_rating_neg_file = open("ml-1m/clean_ratings_neg_p{p}.dat".format(p=p),"w",encoding="latin1")
        lines = clean_rating_file.readlines()
        _user = lines[0].split("\t")[0]
        write_lines = []
        negs = np.arange(movie_size)
        np.random.shuffle(negs)
        negs = list(negs)
        for line in lines:
            cur_user = line.split("\t")[0]
            if cur_user == _user:
                if int(line.split("\t")[1]) in negs:
                    write_lines += [line]
                    negs.remove(int(line.split("\t")[1]))
            else:
                print(cur_user,_user)
                neg_line = "".join(["{}\t".format(cur_user)]+["{}\t".format(neg) for neg in negs[:99]]+["{}\n".format(negs[100])])
                clean_rating_tr_file.write("".join(write_lines[:-p]))
                clean_rating_tr_file.flush()
                clean_rating_val_file.write("".join(write_lines[-p:]))
                clean_rating_val_file.flush()
                clean_rating_neg_file.write(neg_line)
                clean_rating_neg_file.flush()
                _user = cur_user
                write_lines = [line]
                negs = np.arange(movie_size)
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

    rating_matrix = np.zeros((user_size , movie_size))
    with open(r"ml-1m/clean_ratings_tr_p{p}.dat".format(p=p),encoding="latin1") as clean_rating_tr_file:
        lines = clean_rating_tr_file.readlines()
        for line in lines:
            u_i = int(line.split("\t")[0])
            m_i = int(line.split("\t")[1])
            rating_matrix[u_i][m_i]=1
    with open(r'ml-1m/rating_matrix_p{p}.pickle'.format(p=p), 'wb') as handle:
        pickle.dump(rating_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("#### save matrix by pickle")
with open(r'ml-1m/movie_infomation_matrix.pickle', 'wb') as handle:
    pickle.dump(movie_infomation_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

from shutil import copyfile
copyfile("ml-1m/clean_ratings_tr_p1.dat", "ml-1m/ml-1m.hr-ndcg.train.rating")
copyfile("ml-1m/clean_ratings_val_p1.dat", "ml-1m/ml-1m.hr-ndcg.test.rating")
copyfile("ml-1m/clean_ratings_neg_p1.dat", "ml-1m/ml-1m.hr-ndcg.test.negative")

copyfile("ml-1m/clean_ratings_tr_p-10.dat", "ml-1m/ml-1m.precision-recall.train.rating")
copyfile("ml-1m/clean_ratings_val_p-10.dat", "ml-1m/ml-1m.precision-recall.test.rating")
copyfile("ml-1m/clean_ratings_neg_p-10.dat", "ml-1m/ml-1m.precision-recall.test.negative")