import hashlib

from music21 import *


def main():
    song = converter.parse("/Users/konradsbuss/Documents/Uni/bak/dataset/lmd_full/5/5922bdbe0120c36c2a7dbaf1f305ee4d.mid")
    song.show("text")
    # output_file_path = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filenamecopy.txt"
    # sanitized_output_file_path = str(output_file_path) + "_sanitized.txt"
    # input_file_path = str(output_file_path)
    # completed_lines_hash = set()
    # sanitized_output_file = open(sanitized_output_file_path, "w")
    # for line in open(input_file_path, "r"):
    #     hashValue = hashlib.md5(line.rstrip().encode('utf-8')).hexdigest()
    #     if hashValue not in completed_lines_hash:
    #         sanitized_output_file.write(line)
    #         completed_lines_hash.add(hashValue)
    # sanitized_output_file.close()


main()