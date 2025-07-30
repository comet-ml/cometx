# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2023 Cometx Development
#      Team. All rights reserved.
# ****************************************

import os
import random
import tempfile

## Randomize large files


def merge_files(temp_files, filename_out):
    with open(filename_out, "w") as fp_out:
        for temp_file in temp_files:
            with open(temp_file.name) as fp:
                line = fp.readline()
                while line:
                    fp_out.write(line)
                    line = fp.readline()


def shuffle_in_memory(filename_in, filename_out):
    # Shuffle a file, line-by-line
    with open(filename_in) as fp:
        lines = fp.readlines()
    # Randomize them in place:
    random.shuffle(lines)
    # Write the new order out:
    with open(filename_out, "w") as fp:
        fp.writelines(lines)


def shuffle(
    filename_in, filename_out, memory_limit, file_split_count, depth=0, debug=False
):
    if os.path.getsize(filename_in) < memory_limit:
        if debug:
            print(" " * depth, f"Level {depth + 1}", "Shuffle in memory...")
        shuffle_in_memory(filename_in, filename_out)
    else:
        if debug:
            print(
                " " * depth,
                f"Level {depth + 1}",
                f"{os.path.getsize(filename_in)} is too big;",
                f"Split into {file_split_count} files...",
            )
        # Split the big file into smaller files
        temp_files = [
            tempfile.NamedTemporaryFile("w+", delete=False)
            for i in range(file_split_count)
        ]
        for line in open(filename_in):
            random_index = random.randint(0, len(temp_files) - 1)
            temp_files[random_index].write(line)

        # Now we shuffle each smaller file
        for temp_file in temp_files:
            temp_file.close()
            shuffle(
                temp_file.name,
                temp_file.name,
                memory_limit,
                file_split_count,
                depth + 1,
                debug,
            )

        # And merge back in place of the original
        if debug:
            print(" " * depth, f"Level {depth + 1}", "Merge files...")
        merge_files(temp_files, filename_out)
