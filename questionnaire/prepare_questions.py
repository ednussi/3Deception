#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import csv

CATCH_ITEMS_FREQ = 3
CATCH_ITEMS_QUESTION = "נא להשיב כן לשאלה הזאת?"


def prepare_slt(path="data/questions_slt.csv"):
    question_templates = OrderedDict()

    with open(path, encoding='utf-8') as out:
        reader = csv.reader(out)

        headers = next(reader, [])

        if len(headers) != 3 or headers[0] != "question" or headers[1] != "answer_format" \
           or headers[2] != "answer":
            raise Exception("prepare_slt: Bad csv format - bad headers")

        num_row = 1
        for row in reader:
            if len(row) != 3:
                raise Exception("prepare_slt: Bad csv format in row {}".format(row))

            if num_row % CATCH_ITEMS_FREQ == 0:
                question_templates["catch_item"] = CATCH_ITEMS_QUESTION

            # Format truthful and deceptive answers
            question_templates[row[0]] = row[1].format(row[2])

    return question_templates


def prepare_vocal_single_option(path="data/questions_vocal_single_option.csv"):
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
            raise Exception("prepare_vocal_single_options: Bad csv format1")

        for row in reader:
            if len(row) != 2:
                raise Exception("prepare_vocal_single_options: Bad csv format2")

            question_templates[row[0]] = row[1]

    return question_templates


def assoc_array_to_list(sv_od):
    return [item for kvp in list(sv_od.items()) for item in kvp]
