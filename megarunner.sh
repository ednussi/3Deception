for au in 51
do
	for fe in 20 25 30 35 40 45 50 55 60 65 70 75 80
	do
		for pc in 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	  	do
	  		echo "MEGARUNNER iteration: AU=$au FE=$fe PCA=$pc"
	  		srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/alon/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/arielt/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/asaph/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/inbar/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/liav/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/lilach/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/maayan/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/omri/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/yuval/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lQSP -nmax

	  		srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/alon/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/arielt/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/asaph/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/inbar/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/liav/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/lilach/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/maayan/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/omri/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  		# srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner2.py -i questionnaire/data/yuval/fs_shapes.not_quantized.csv -atop -A$au -fall -F$fe -pglobal -P$pc -lSP -nmax
	  	done
	done
done