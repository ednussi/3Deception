for au in 51
do
	for fe in 20 80
	do
		for pc in 7 20
	  	do
	  		echo "MEGARUNNER iteration: AU=$au FE=$fe PCA=$pc"
	  		/cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/alon/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax &

	  		/cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/alon/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax &
	  	done
	done
done
