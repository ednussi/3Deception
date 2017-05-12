#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from io import StringIO
import csv
import random
import os
from constants import QUESTION_TYPES

CATCH_ITEMS_FREQ = 3
CATCH_ITEMS_QUESTION = "נא להשיב כן לשאלה הזאת?"

AUDIO_FLAGS = {
    
    'מצרים': 'voice/cit/country_a/egypt.mp3',
    'צרפת': 'voice/cit/country_a/france.mp3',
    'יוון': 'voice/cit/country_a/greece.mp3',
    'הונגריה': 'voice/cit/country_a/hungary.mp3',
    'איסלנד': 'voice/cit/country_a/iceland.mp3',
    'ישראל': 'voice/cit/country_a/israel.mp3',
    'איטליה': 'voice/cit/country_a/italy.mp3',
    'יפן': 'voice/cit/country_a/japan.mp3',
    'קרואטיה': 'voice/cit/country_a/kroatia.mp3',
    'הולנד': 'voice/cit/country_a/holand.mp3',
    'מרוקו': 'voice/cit/country_a/maroko.mp3',
    'מולדובה': 'voice/cit/country_a/moldova.mp3',
    'רוסיה': 'voice/cit/country_a/russia.mp3',
    'טורקיה': 'voice/cit/country_a/turkey.mp3',
    'אוקראינה': 'voice/cit/country_a/ukraine.mp3',
    'ארצות הברית': 'voice/cit/country_a/usa.mp3',

    'עדי': 'voice/cit/female_a/adi.mp3',
    'אריאל': 'voice/cit/female_a/ariel.mp3',
    'אבישג': 'voice/cit/female_a/avishag.mp3',
    'אביטל': 'voice/cit/female_a/avital.mp3',
    'דפנה': 'voice/cit/female_a/dafna.mp3',
    'דבורה': 'voice/cit/female_a/debura.mp3',
    'אפרת': 'voice/cit/female_a/efrat.mp3',
    'אלנה': 'voice/cit/female_a/elena.mp3',
    'ורדה': 'voice/cit/female_a/varda.mp3',
    'גפן': 'voice/cit/female_a/gefen.mp3',
    'לאה': 'voice/cit/female_a/lea.mp3',
    'לילך': 'voice/cit/female_a/lilach.mp3',
    'נעמי': 'voice/cit/female_a/naomi.mp3',
    'נטלי': 'voice/cit/female_a/natali.mp3',
    'נופר': 'voice/cit/female_a/nofar.mp3',
    'אושרה': 'voice/cit/female_a/oshra.mp3',
    'ריבקה': 'voice/cit/female_a/rivka.mp3',
    'רותם': 'voice/cit/female_a/rotem.mp3',
    'שרה': 'voice/cit/female_a/sara.mp3',
    'שושנה': 'voice/cit/female_a/shushana.mp3',
    'סיון': 'voice/cit/female_a/sivan.mp3',
    'סיוון': 'voice/cit/female_a/sivan.mp3',
    'סופיה': 'voice/cit/female_a/sofia.mp3',
    'יערה': 'voice/cit/female_a/yaara.mp3',
    'יעל': 'voice/cit/female_a/yael.mp3',
    'ירדן': 'voice/cit/female_a/yarden.mp3',
    'יוליה': 'voice/cit/female_a/yulia.mp3',
    'יובל': 'voice/cit/female_a/yuval.mp3',
    'סיגלית': 'voice/cit/female_a/sigalit.mp3',

    'birth_country_female': 'voice/cit/female_q/birth_country.mp3',
    'birth_month_female': 'voice/cit/female_q/birth_month.mp3',
    'first_name_female': 'voice/cit/female_q/first_name.mp3',
    'mother_name_female': 'voice/cit/female_q/mother_name.mp3',
    'surname_female': 'voice/cit/female_q/surname.mp3',

    'בן חמו': 'voice/cit/lastname_a/ben_hamu.mp3',
    'בן סימון': 'voice/cit/lastname_a/ben_simun.mp3',
    'כהן': 'voice/cit/lastname_a/cohen.mp3',
    'דביר': 'voice/cit/lastname_a/dvir.mp3',
    'פריד': 'voice/cit/lastname_a/frid.mp3',
    'גל': 'voice/cit/lastname_a/gal.mp3',
    'גרוסמן': 'voice/cit/lastname_a/grosman.mp3',
    'גרוס': 'voice/cit/lastname_a/gross.mp3',
    'יעקובוביץ': 'voice/cit/lastname_a/jakobovitz.mp3',
    'קרו': 'voice/cit/lastname_a/karu.mp3',
    'קליין': 'voice/cit/lastname_a/klain.mp3',
    'קליין סלע': 'voice/cit/lastname_a/klainsela.mp3',
    'לוי': 'voice/cit/lastname_a/levi.mp3',
    'מרקוביץ': 'voice/cit/lastname_a/markovitz.mp3',
    'ניסים': 'voice/cit/lastname_a/nisim.mp3',
    'אוחיון': 'voice/cit/lastname_a/ohayon.mp3',
    'פינדיק': 'voice/cit/lastname_a/pindik.mp3',
    'שוורץ': 'voice/cit/lastname_a/shwartz.mp3',
    'טל': 'voice/cit/lastname_a/tal.mp3',
    'תורג\'מן': 'voice/cit/lastname_a/torgeman.mp3',
    'נוסינוביץ': 'voice/cit/lastname_a/nussinovitch.mp3',
    'קצלניק': 'voice/cit/lastname_a/kazelnik.mp3',
    'פסטרנק': 'voice/cit/lastname_a/pasternak.mp3',
    'אורנשטין': 'voice/cit/lastname_a/orenstein.mp3',

#    'עדי': 'voice/cit/male_a/adi.mp3',
    'אלון': 'voice/cit/male_a/alon.mp3',
#    'אריאל': 'voice/cit/male_a/ariel.mp3',
    'ארתור': 'voice/cit/male_a/arthur.mp3',
    'אסף': 'voice/cit/male_a/asaf.mp3',
    'אבי': 'voice/cit/male_a/avi.mp3',
    'דוד': 'voice/cit/male_a/david.mp3',
    'אהוד': 'voice/cit/male_a/ehud.mp3',
    'אליסף': 'voice/cit/male_a/elyasaf.mp3',
    'ליאב': 'voice/cit/male_a/liav.mp3',
    'מאור': 'voice/cit/male_a/maor.mp3',
    'מארק': 'voice/cit/male_a/mark.mp3',
    'מתן': 'voice/cit/male_a/matan.mp3',
    'מוטי': 'voice/cit/male_a/moti.mp3',
    'עודד': 'voice/cit/male_a/oded.mp3',
    'רועי': 'voice/cit/male_a/roi.mp3',
    'עמרי': 'voice/cit/male_a/omri.mp3',
#    'רותם': 'voice/cit/male_a/rotem.mp3',
    'תומר': 'voice/cit/male_a/tomer.mp3',
    'יהודה': 'voice/cit/male_a/yehuda.mp3',
    'יואל': 'voice/cit/male_a/yoel.mp3',
    'יהונתן': 'voice/cit/male_a/yonatan.mp3',
    'צבי': 'voice/cit/male_a/zvi.mp3',
    'ערן': 'voice/cit/male_a/eran.mp3',
    'גרגורי': 'voice/cit/male_a/gregory.mp3',
    'רפאל': 'voice/cit/male_a/refael.mp3',
    'יפים': 'voice/cit/male_a/yafim.mp3',
    'יואב': 'voice/cit/male_a/yoav.mp3',

    'birth_country_male': 'voice/cit/male_q/birth_country.mp3',
    'birth_month_male': 'voice/cit/male_q/birth_month.mp3',
    'first_name_male': 'voice/cit/male_q/first_name.mp3',
    'mother_name_male': 'voice/cit/male_q/mother_name.mp3',
    'surname_male': 'voice/cit/male_q/surname.mp3',

    'ינואר': 'voice/cit/month_a/1.mp3',
    'אוקטובר': 'voice/cit/month_a/10.mp3',
    'נובמבר': 'voice/cit/month_a/11.mp3',
    'דצמבר': 'voice/cit/month_a/12.mp3',
    'פברואר': 'voice/cit/month_a/2.mp3',
    'מרץ': 'voice/cit/month_a/3.mp3',
    'אפריל': 'voice/cit/month_a/4.mp3',
    'מאי': 'voice/cit/month_a/5.mp3',
    'יוני': 'voice/cit/month_a/6.mp3',
    'יולי': 'voice/cit/month_a/7.mp3',
    'אוגוסט': 'voice/cit/month_a/8.mp3',
    'ספטמבר': 'voice/cit/month_a/9.mp3'

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


def prepare_cit(path="data/questions_cit.csv", male=True):
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
            rowname = row[0][:-1] if row[0].endswith('\u200f') else row[0]
            question_templates[row[0]] = {
                "type": QUESTION_TYPES[rowname],
                "question": (AUDIO_FLAGS.get(rowname+'_male', 'no_audio')
                             if male else AUDIO_FLAGS.get(rowname+'_female', 'no_audio'), row[1]),
                "true": (AUDIO_FLAGS.get(row[2][:-1], 'no_audio'), row[2]),
                "false": [(AUDIO_FLAGS.get(row[i][:-1], 'no_audio'), row[i]) for i in range(3, 8)]
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
