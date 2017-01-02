#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import csv

CATCH_ITEMS_FREQ = 3
CATCH_ITEMS_QUESTION = "נא להשיב כן לשאלה הזאת?"

POSSIBLE_ANSWERS_CIT = {

}


def prepare_slt(path="data/questions_slt.csv"):
    """
    Load questions and answer formats for single-option vocal questions - SLT
    :param path: csv file to load from
    :return: ordered dictionary where keys are question type and values are
    dicts with 'true' and 'false' questions.
    """
    question_templates = OrderedDict()

    with open(path, encoding='utf-8') as out:
        reader = csv.reader(out)

        headers = next(reader, [])

        if len(headers) != 4 or headers[0] != "question" or headers[1] != "answer_format" \
           or headers[2] != "true_answer" or headers[3] != "false_answer":
            raise Exception("prepare_slt: Bad csv format - bad headers")

        num_row = 1
        for row in reader:
            if len(row) != 4:
                raise Exception("prepare_slt: Bad csv format in row {}".format(row))

            if num_row % CATCH_ITEMS_FREQ == 0:
                question_templates["catch_item"] = CATCH_ITEMS_QUESTION

            # Format truthful and deceptive answers
            question_templates[row[0]] = {
                'true': row[1].format(row[2]),
                'false': row[1].format(row[3])
            }

    return question_templates


def prepare_cit(path="data/questions_cit.csv"):
    """
    Load questions and answer formats for single-option vocal questions - CIT
    :param path: csv file to load from
    :return: ordered dictionary where keys are questions and values are answer format that the subject must say out loud
    """
    question_templates = OrderedDict()

    with open(path, encoding='utf-8') as out:
        reader = csv.reader(out)

        headers = next(reader, [])

        if len(headers) != 7 or headers[0] != "question" or headers[1] != "answer_format" \
           or headers[2] != "true_answer":
            raise Exception("prepare_cit: Bad csv format - headers")

        num_row = 1
        for row in reader:
            if len(row) != 7:
                raise Exception("prepare_cit: Bad csv format in row {}".format(row))

            if num_row % CATCH_ITEMS_FREQ == 0:
                question_templates["catch_item"] = CATCH_ITEMS_QUESTION

            # Format truthful and deceptive answers
            question_templates[row[0] + "_true"] = (POSSIBLE_ANSWERS_CIT.get(row[2], '0'), row[1].format(row[2]))
            question_templates[row[0] + "_false_1"] = (POSSIBLE_ANSWERS_CIT.get(row[3], '0'), row[1].format(row[3]))
            question_templates[row[0] + "_false_2"] = (POSSIBLE_ANSWERS_CIT.get(row[4], '0'), row[1].format(row[4]))
            question_templates[row[0] + "_false_3"] = (POSSIBLE_ANSWERS_CIT.get(row[5], '0'), row[1].format(row[5]))
            question_templates[row[0] + "_false_4"] = (POSSIBLE_ANSWERS_CIT.get(row[6], '0'), row[1].format(row[6]))

    return question_templates


def assoc_array_to_list(sv_od):
    return [item for kvp in list(sv_od.items()) for item in kvp]
