from __future__ import unicode_literals  # compatible with python3 unicode
import codecs
import collections
import sys
from sys import argv


def Generate_dictionary(path):
    input_file = codecs.open(path, 'r', 'utf-8')
    output_file = codecs.open("dictionary.txt", 'w', 'utf-8')

    for sentence in input_file:
        words = sentence.strip().split(" ")
        if words[0] != "":
            wordlist.append(words[0])  # word

    word_counts = collections.Counter(wordlist)
    word_counts_top = word_counts.most_common(497)
    # 统计出出现频率最高的前497个词，方便和句首、句尾、其他构成500维的One-Hot
    for word in word_counts_top:
        output_file.write(word[0]+"\n")

    input_file.close()
    output_file.close()


if __name__ == "__main__":
    wordlist = []
    training_path = "../../source/Dataset/People_Daily/example.train"
    Generate_dictionary(training_path)
    print("Generate dictionary finished")
