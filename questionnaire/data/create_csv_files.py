#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv, sys
import pandas as pd
import os


# Before running need to change manually the <responses>.xlsx files header to:
#timestamp	first_name	surname	mother_name	birth_country	birth_month	email	age

def main():
	excel_path = sys.argv[1]

	df = pd.read_excel(excel_path)
	for index,row in df.iterrows():
		writeUserCsv(row)
		
	#df.loc[:, 'answer_true']

def writeUserCsv(row):

	#params:
	
	timestamp = row[0]
	first_name = row[1]
	last_name = row[2]
	mother_name = row[3]
	birth_country = row[4]
	birth_month = row[5]
	mail = row[6]
	age = row[7]
	sex = row[8]
	new_folder_name = mail
	if not os.path.exists(new_folder_name):
	    os.makedirs(new_folder_name)

	fid = open(new_folder_name+'\questions_slt.csv','w')
	fid.write('question,answer_format,true_answer,false_answer')
	
	false_answer = 'אריאל‏' if first_name != 'אריאל‏' else 'דניאל‏‏'
	fid.write('first_name,האם השם הפרטי שלך הוא \{\}?‏,'+first_name+','+false_answer)
	
	false_answer = 'פרידמן‏' if last_name !=  'פרידמן‏' else  'שפיגל‏'
	fid.write('surname,האם שם המשפחה שלך הוא \{\}?‏,'+last_name+','+false_answer)

	false_answer = 'תמר‏' if mother_name != 'תמר‏' else 'אביגיל‏'
	fid.write('mother_name,האם השם של אמא שלך הוא \{\}?‏,'+mother_name+','+false_answer)

	false_answer = 'איטליה‏' if birth_country !=  'איטליה‏' else 'אוסטרליה‏'
	fid.write('birth_country,האם ארץ הלידה שלך היא \{\}?‏,'+birth_country+','+false_answer)

	false_answer = 'אפריל‏' if birth_month !=  'אפריל‏' else 'ספטמבר‏'
	fid.write('birth_month,האם חודש הלידה שלך הוא \{\}?‏,'+birth_month+','+false_answer)
	fid.close()


if __name__ == "__main__":
    main()
