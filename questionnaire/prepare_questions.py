#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from io import StringIO
import csv
import random
import os

CATCH_ITEMS_FREQ = 3
CATCH_ITEMS_QUESTION = "נא להשיב כן לשאלה הזאת?"

AUDIO_FLAGS = {
    'אסף': 'voice/qtype/1.mp3‏',
    'יהודה': 'voice/qtype/1.mp3‏',
    'דוד': 'voice/qtype/1.mp3‏',
    'תומר': 'voice/qtype/1.mp3‏',
    'רותם': 'voice/qtype/1.mp3‏',
    'אבי': 'voice/qtype/1.mp3‏',
    'יונתן': 'voice/qtype/1.mp3‏',
    'מארק': 'voice/qtype/1.mp3‏',
    'עודד': 'voice/qtype/1.mp3‏',
    'יואל': 'voice/qtype/1.mp3‏',
    'יערה': 'voice/qtype/1.mp3‏',
    'דפנה': 'voice/qtype/1.mp3‏',
    'יעל': 'voice/qtype/1.mp3‏',
    'נופר': 'voice/qtype/1.mp3‏',
    'סיוון': 'voice/qtype/1.mp3‏',
    'יוליה': 'voice/qtype/1.mp3‏',
    'יובל': 'voice/qtype/1.mp3‏',
    'אבישג': 'voice/qtype/1.mp3‏',
    'לילך': 'voice/qtype/1.mp3‏',
    'נטלי': 'voice/qtype/1.mp3‏',
    'כהן': 'voice/qtype/1.mp3‏',
    'לוי': 'voice/qtype/1.mp3‏',
    'גל': 'voice/qtype/1.mp3‏',
    'שוורץ': 'voice/qtype/1.mp3‏',
    'אוחיון': 'voice/qtype/1.mp3‏',
    'מרקוביץ': 'voice/qtype/1.mp3‏',
    'בן סימון': 'voice/qtype/1.mp3‏',
    'פריד': 'voice/qtype/1.mp3‏',
    'דביר': 'voice/qtype/1.mp3‏',
    'גרוסמן': 'voice/qtype/1.mp3‏',
    'איטליה': 'voice/qtype/1.mp3‏',
    'יוון': 'voice/qtype/1.mp3‏',
    'קרואטיה': 'voice/qtype/1.mp3‏',
    'איסלנד': 'voice/qtype/1.mp3‏',
    'מולדובה': 'voice/qtype/1.mp3‏',
    'הונגריה': 'voice/qtype/1.mp3‏',
    'מרוקו': 'voice/qtype/1.mp3‏',
    'יפן': 'voice/qtype/1.mp3‏',
    'טורקיה': 'voice/qtype/1.mp3‏',
    'מצרים': 'voice/qtype/1.mp3‏',
    'ינואר': 'voice/qtype/1.mp3‏',
    'פברואר': 'voice/qtype/1.mp3‏',
    'מרץ': 'voice/qtype/1.mp3‏',
    'אפריל': 'voice/qtype/1.mp3‏',
    'מאי': 'voice/qtype/1.mp3‏',
    'יוני': 'voice/qtype/1.mp3‏',
    'יולי': 'voice/qtype/1.mp3‏',
    'אוגוסט': 'voice/qtype/1.mp3‏',
    'ספטמבר': 'voice/qtype/1.mp3‏',
    'אוקטובר': 'voice/qtype/1.mp3‏',
    'נובמבר': 'voice/qtype/1.mp3‏',
    'דצמבר': 'voice/qtype/1.mp3‏',
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
    :return: ordered dictionary where keys are question types and values are 
        {question, true answer, list of false answers}
    """
    question_templates = OrderedDict()

    with open(path, encoding='utf-8') as out:
        reader = csv.reader(out)

        headers = next(reader, [])

        if len(headers) != 8 or headers[0] != "question" or headers[1] != "answer_format" \
           or headers[2] != "true_answer":
            raise Exception("prepare_cit: Bad csv format - headers")

        num_row = 1
        for row in reader:
            if len(row) != 8:
                raise Exception("prepare_cit: Bad csv format in row {}".format(row))

            if num_row % CATCH_ITEMS_FREQ == 0:
                question_templates["catch_item"] = CATCH_ITEMS_QUESTION

            # Format truthful and deceptive answers
            question_templates[row[0]] = {
                "question": (AUDIO_FLAGS.get(row[2], 'no_audio'), row[1]),
                "true": (AUDIO_FLAGS.get(row[2], 'no_audio'), row[2]),
                "false": [(AUDIO_FLAGS.get(row[i], 'no_audio'), row[i]) for i in range(3, 8)]
            }

    return question_templates


def parse_form_record_cit(record, is_male = True):
    """
    Parse line from LIES007 form to CIT csv file
    :param record: comma separated values, 
        fields are in the following order and formats:
            1 שם פרטי    first_name    text
            2 שם משפחה    surname    text
            3 שם אם    mother_name    text
            4 ארץ לידה    birth_country    text
            5 חודש לידה    birth_month    enum
            6 איזה מהשמות (פרטיים) הבאים הם בעלי משמעות עבורך?    important_names    csv
            7 איזה מהשמות (משפחה) הבאים הם בעלי משמעות עבורך?    important_surnames    csv
            8 איזה מהשמות (ארצות) הבאים הם בעלי משמעות עבורך?    important_countries    csv
            9 איזה מהחודשים הבאים הם בעלי משמעות עבורך?    important_months    csv
            10 כתובת אימייל    subject_id
    """
    str_record = StringIO(record)
    reader = csv.reader(str_record, delimiter='\t')
    reader = [y for x in reader for y in x]

    if not os.path.isdir('data/{}'.format(reader[10])):
        os.mkdir('data/{}'.format(reader[10]))
    
    if os.path.exists('data/{}/questions_cit.csv'.format(reader[10])):
        os.remove('data/{}/questions_cit.csv'.format(reader[10]))

    f = "data/{}/questions_cit.csv".format(reader[10])

    male_names = {'אסף‏', 'יהודה‏', 'דוד‏', 'תומר‏', 'רותם‏', 'אבי‏', 'יוני‏', 'מארק‏', 'עודד‏', 'יואל'}
    female_names = {'יערה‏', 'דפנה‏', 'יעל‏', 'נופר‏', 'סיוון‏', 'יוליה‏', 'יובל‏', 'אבישג‏', 'לילך‏', 'נטלי'}
    surnames = {'כהן‏', 'לוי‏', 'גל‏', 'שוורץ‏', 'אוחיון‏', 'מרקוביץ‏', 'בן סימון‏', 'פריד‏', 'דביר‏', 'גרוסמן'}
    countries = {'איטליה‏', 'יוון‏', 'קרואטיה‏', 'איסלנד‏', 'מולדובה‏', 'הונגריה‏', 'מרוקו‏', 'יפן‏', 'טורקיה‏', 'מצרים'}
    months = {'ינואר‏', 'פברואר‏', 'מרץ‏', 'אפריל‏', 'מאי‏', 'יוני‏', 'יולי‏', 'אוגוסט‏', 'ספטמבר‏', 'אוקטובר‏', 'נובמבר‏', 'דצמבר'}

    qtypes = {
        'שם פרטי': 'first_name‏',
        'שם משפחה': 'surname‏',
        'שם אם': 'mother_name‏',
        'ארץ לידה': 'birth_country‏',
        'חודש לידה': 'birth_month'
    }

    formats = {
        'first_name': 'מה הוא השם הפרטי שלך?‏‏',
        'surname': 'מה הוא שם המשפחה שלך?‏‏',
        'mother_name': 'מה הוא השם של אמא שלך?‏‏',
        'birth_country': 'מה היא ארץ הלידה שלך?‏‏',
        'birth_month': 'מה הוא חודש הלידה שלך?‏'
    }

    with open(f, 'w‏', newline='') as out:
        wr = csv.writer(out)

        header = [
            "question",
            "answer_format",
            "true_answer",
            "false_answer_1",
            "false_answer_2",
            "false_answer_3",
            "false_answer_4",
            "false_answer_5"
        ]
        wr.writerow(header)

        false_answers = None

        # first_name
        if is_male:
            false_answers = list(male_names - set([n+'‏' for n in reader[0][6].split(', ')]))
        else:
            false_answers = list(female_names - set([n+'‏' for n in reader[0][6].split(', ')]))

        random.shuffle(false_answers)
        false_answers = false_answers[:5]
        wr.writerow([
            'first_name‏',  # question type
            formats['first_name'],  # question format
            reader[1],  # true
            *false_answers
        ])

        # surname
        false_answers = list(surnames - set(reader[0][7].split(', ')))
        random.shuffle(false_answers)
        false_answers = false_answers[:5]
        wr.writerow([
            'surname‏',  # question type
            formats['surname'],  # question format
            reader[2],  # true
            *false_answers
        ])

        # mother_name
        false_answers = list(female_names - set(reader[0][6].split(', ')))
        random.shuffle(false_answers)
        false_answers = false_answers[:5]
        wr.writerow([
            'mother_name‏',  # question type
            formats['mother_name'],  # question format
            reader[3],  # true
            *false_answers
        ])

        # birth_country
        false_answers = list(countries - set(reader[0][8].split(', ')))
        random.shuffle(false_answers)
        false_answers = false_answers[:5]
        wr.writerow([
            'birth_country‏',  # question type
            formats['birth_country'],  # question format
            reader[4],  # true
            *false_answers
        ])

        # birth_month
        false_answers = list(months - set(reader[0][9].split(', ')))
        random.shuffle(false_answers)
        false_answers = false_answers[:5]
        wr.writerow([
            'birth_month‏',  # question type
            formats['birth_month'],  # question format
            reader[5],  # true
            *false_answers
        ])


def assoc_array_to_list(sv_od):
    return [item for kvp in list(sv_od.items()) for item in kvp]
