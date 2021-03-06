#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import csv


def prepare_vocal_multiple_option(path="data/questions_vocal_single_option.csv"):
    """
    Load questions and answer formats for single-option vocal questions
    :param path: csv file to load from
    :return: ordered dictionary where keys are questions and values are answer format that the subject must say out loud
    """
    question_templates = OrderedDict()

    with open(path, encoding='utf-8') as out:
        reader = csv.reader(out)

        headers = next(reader, [])

        if len(headers) != 2 or headers[0] != "question" or headers[1] != "answer_format":
            raise Exception("prepare_vocal_single_options: Bad csv format")

        for row in reader:
            if len(row) != 5:
                raise Exception("prepare_vocal_single_options: Bad csv format")

            question_templates[row[0]] = [row[1],row[2],row[3],row[4]]

    return question_templates


def assoc_array_to_list_multiple(sv_od):
    return [item for kvp in list(sv_od.items()) for item in kvp]
