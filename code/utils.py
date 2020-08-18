""" general utility functions used throughput codebase """

import os


def ensure_dir_exists(d):
    """ creates given dir if it does not already exist """
    if not os.path.isdir(d):
        os.makedirs(d)


def load_lines(fn):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines
