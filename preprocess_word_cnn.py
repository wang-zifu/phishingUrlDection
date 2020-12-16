
import collections
import numpy as np
import re


def shuffle_all(features, labels):
    '''
    Inputs:
        features: a list of list of feautres
        labels: a list of all the labels
    Output: (shuffled features, shuffled labels)
    '''

    indices = np.array(range(len(labels)))

    np.random.shuffle(indices)

    features = features[indices, :]

    labels = labels[indices]
    return features, labels


def convert_urls_to_vector(file_names, is_phishing):

    def count_words(file_names):
        #all_text = ""
        sum_word_list = []
        for file_name in file_names:
            file = open(file_name, 'r')
            #all_text += file.read()
            for line in file:
                line = line.strip().replace('"', "")  # remove " " that surrounds some URLs
                url = line.strip('\n')
                url_word_split = re.split(regular, url)
                while '' in url_word_split:
                    url_word_split.remove('')
                sum_word_list += url_word_split
            file.close()
        word_counts = collections.Counter(sum_word_list)
        return word_counts
    def sum_unique_words(word_counts):
        temp=word_counts.copy()
        for key in list(temp.keys()):
            if temp[key] < 100:
                del temp[key]
        return len(temp)

    regular = re.compile(r'[-|,|;|\.|!|\?|"|\/|\\|\||_|@|#|\$|%|\^|&|\*|~|\+|`|=|<|>|\[|\]|\(|\)|\{|\}]')
    word_counts = count_words(file_names)

    unique_words=sum_unique_words(word_counts)

    print(word_counts)

    vector_length = 300
    word_to_id = {}
    url_vectors = []
    labels = []
    word_to_id["<PAD>"] = 0
    word_to_id["<UNK>"] = 1
    id = 2
    for file_id, file_name in enumerate(file_names):
        num_urls_in_file = 0
        f = open(file_name, 'r')
        for line in f:
            num_urls_in_file += 1
            line = line.strip().replace('"', "")  # remove " " that surrounds some URLs
            url = line.strip('\n')

            url_list = []

            print("url:", url)
            # vector_length=300
            url_word_split = re.split(regular, url)

            while '' in url_word_split:
                url_word_split.remove('')
            url_vec = np.full(vector_length, 0)  # Fill with <PAD>


            for i in range(min(vector_length, len(url_word_split))):
                print("len(url_word_split):", len(url_word_split))
                word = url_word_split[i]
                print("word:", word)
                if word_counts[word] >= 100:
                    print('word_counts[word]:', word_counts[word])
                    if word not in word_to_id:
                        word_to_id[word] = id
                        id += 1
                    url_vec[i] = word_to_id[word]

                else:
                    url_vec[i] = 1  # <UNK>

                # if char_counts[char] >= 100:
                #     print("char_counts[char]:",char_counts[char])
                #     if char not in char_to_id:
                #         char_to_id[char] = id
                #         id += 1
                #     url_vec[i] = char_to_id[char]
                # else:
                #     url_vec[i] = 1 # <UNK>
            url_vectors.append(url_vec)

        # Generate labels
        if is_phishing[file_id]:
            labels += [1] * num_urls_in_file
        else:
            labels += [0] * num_urls_in_file

            # Shuffle
    url_vectors, labels = shuffle_all(np.array(url_vectors), np.array(labels))

    # Train-test split
    # train_ratio = 0.8
    # num_urls = len(url_vectors)
    # split_index = int(train_ratio * num_urls)
    #
    # train_data = np.array(url_vectors[0:split_index])
    # train_labels = np.array(labels[0:split_index])
    #
    # test_data = np.array(url_vectors[split_index:])
    # test_labels = np.array(labels[split_index:])
    # #检测数据集中是否出现了脏数据
    # f = open(r"E:\daima-sx\test2\result\word_detection.csv", "w+")
    # for temp in url_vectors:
    #     f.write(str(temp))
    #     f.write('\n')



    return url_vectors, labels,unique_words


def main():
    # urls_file_name =
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]

    convert_urls_to_vector(file_names, is_phishing)

    # save vector representations
    # write_to = urls_file_name.replace(".txt", "") + ".csv"
    # np.savetxt(write_to, url_vectors, fmt='%s', delimiter=",")
    # print("Finished writing to", write_to)


if __name__ == "__main__":
    main()













