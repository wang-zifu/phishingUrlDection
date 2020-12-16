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


    vector_length = 300
    url_vectors = []
    labels = []
    regular = re.compile(r'(-|,|;|\.|!|\?|"|\/|\\|\||_|@|#|\$|%|\^|&|\*|~|\+|`|=|<|>|\[|\]|\(|\)|\{|\}|account|admin|administrator'
                         r'|auth|bank|client|confirm|cmd|email|host|login|password|pay|private|registed|safe|secure|security|'
                         r'sign|service|signin|submit|user|update|validation|verification|webscr)')
    # 字符映射编码
    char_to_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
                  'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
                  'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'A': 27, 'B': 28,
                  'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33,
                  'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44,
                  'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49,
                  'X': 50, 'Y': 51, 'Z': 52, '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, '7': 60,
                  '8': 61, '9': 62, '-': 63, ',': 64, ';': 65,
                  '.': 66, '!': 67, '?': 68, '"': 69, '/': 70, '\\': 71, '|': 72, '_': 73, '@': 74, '#': 75, '$': 76,
                  '%': 77, '^': 78,'&': 79, '*': 80, '~': 81, '+': 82, '`': 83, '=': 84, '<': 85, '>': 86,
                  '(': 87, ')': 88, '[': 89, ']': 90, '{': 91, '}': 92, '<PAD>': 93, '<UNK>': 94,'account':95,
                  'admin':96,'administrator':97,'auth':98,'bank':99,'client':100,'confirm':101,'cmd':102,'email':103,
                  'host':104,'login':105,'password':106,'pay':107,'private':108,'registed':109,'safe':110,'secure':111,
                  'security':112 ,'sign':113,'service':114,'signin':115,'submit':116,'user':117,'update':118,
                  'validation':119,'verification':120,'webscr':121}

    for file_id, file_name in enumerate(file_names):
        num_urls_in_file = 0
        f = open(file_name, 'r')
        for line in f:
            num_urls_in_file += 1
            line = line.strip().replace('"', "")  # remove " " that surrounds some URLs
            url = line.strip('\n')

            url_list=[]
            url_word_split = re.split(regular, url)

            while '' in url_word_split:
                url_word_split.remove('')

            for i in range(len(url_word_split)):
                if url_word_split[i] not in char_to_id:
                    for j in range(len(url_word_split[i])):
                        url_list+=url_word_split[i][j]
                else:
                    url_list+=url_word_split[i]

            url_vec = np.full(vector_length, 93)  # Fill with <PAD>
            for i in range(min(vector_length, len(url_list))):
                print("len(url_list):", len(url_list))
                char = url_list[i]
                print("char:", char)
                if char not in char_to_id:

                    url_vec[i] = 94  # <UNK>
                else:
                    url_vec[i] = char_to_id[char]


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

    return url_vectors, labels


def main():

    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]

    convert_urls_to_vector(file_names, is_phishing)

    # save vector representations
    # write_to = urls_file_name.replace(".txt", "") + ".csv"
    # np.savetxt(write_to, url_vectors, fmt='%s', delimiter=",")
    # print("Finished writing to", write_to)


if __name__ == "__main__":
    main()