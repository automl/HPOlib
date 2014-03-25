

directory=`ls | grep "^smac_2_06_01-dev\>"`
if [ -a "${directory}" ]
    then
    echo "Looking for SMAC (smac_2_06_01-dev):"

    printf "%5s | %4s(%5s) | %5s | %10s | %10s\n" "Seed" "#run" "crash" "#iter" "Performance" "Test-Perf"
    for i in `seq 1000 1000 10000`
    do
      directory=`ls | grep "^smac_2_06_01-dev_${i}_"`
      if [ -f "${directory}/smac_2_06_01-dev.out" ]
      then
        it=`cat ${directory}/smac*.out | grep "Model/Iteration used:" | tail -1`
        it=`echo $it | cut -d' ' -f3`

        per=`cat ${directory}/smac*.out | grep "Performance of the Incumbent:" | tail -1`
        per=`echo $per | cut -d' ' -f5`

        num=`cat ${directory}/smac*.out | grep "Algorithm Runs used:" | tail -1`
        num=`echo $num | cut -d' ' -f4`

        numC=`ls ${directory}/ | grep 'instance.out$' | wc -l`
        numC=$(($numC - 1))

        test_error=999
        test_file=`ls "$directory/" | grep "_test_run.out"`
        if [ -f "${directory}/$test_file" ]
        then
          test_error=`cat "$directory/$test_file" | grep "Result for ParamILS: SAT" | tail -1`
          test_error=`echo "$test_error" | cut -d' ' -f7| cut -d',' -f1`
        fi
        printf "%5s | %4s(%5s) | %5s | %10f | %10f\n" "$i" "$num" "$numC" "$it" "$per" "$test_error"
      fi

    done
fi


directory=`ls | grep "^partial_smac\>"`
if [ -a "${directory}" ]
then

    echo "Looking for SMAC (smac-v2_06_02-partial38):"

    printf "%5s | %4s(%5s) | %5s | %10s | %10s\n" "Seed" "#run" "crash" "#iter" "Performance" "Test-Perf"
    for i in `seq 1000 1000 10000`
    do
      directory=`ls | grep "^partial_smac_${i}_"`
      if [ -f "${directory}/partial_smac.out" ]
      then
        it=`cat ${directory}/partial_*.out | grep "Model/Iteration used:" | tail -1`
        it=`echo $it | cut -d' ' -f3`

        per=`cat ${directory}/partial_*.out | grep "Performance of the Incumbent:" | tail -1`
        per=`echo $per | cut -d' ' -f5`

        num=`cat ${directory}/partial_*.out | grep "Algorithm Runs used:" | tail -1`
        num=`echo $num | cut -d' ' -f4`

        numC=`ls ${directory}/ | grep 'instance.out$' | wc -l`
        numC=$(($numC - 1))

        test_error=999
        test_file=`ls "$directory/" | grep "_test_run.out"`
        if [ -f "${directory}/$test_file" ]
        then
          test_error=`cat "$directory/$test_file" | grep "Result for ParamILS: SAT" | tail -1`
          test_error=`echo "$test_error" | cut -d' ' -f7| cut -d',' -f1`
        fi
        printf "%5s | %4s(%5s) | %5s | %10f | %10f\n" "$i" "$num" "$numC" "$it" "$per" "$test_error"
      fi

    done
fi


directory=`ls | grep "^hyperopt_august2013_mod\>"`
if [ -a "${directory}" ]
then
    echo "Looking for TPE:"

    printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
    for i in `seq 1000 1000 10000`
    do
      directory=`ls | grep "^hyperopt_august2013_mod_${i}_"`
      if [ -a "${directory}/hyperopt_august2013_mod.out" ]
      then
        num=`cat ${directory}/hyperopt_august2013_mod.out | grep " -----------------------RUNNING RUNSOLVER" | wc -l`

        per=`cat ${directory}/hyperopt_august2013_mod.out | grep "Result for ParamILS:" | sort -r | tail -1`
	per=`echo $per | cut -d' ' -f9`
	per=`echo $per | sed 's/,//'`

        numC=`ls ${directory}/ | grep 'instance.out$' | wc -l`
        numC=$(($numC - 1))

        test_error=999
        test_file=`ls -1 "$directory" | grep "_test_run.out"`
        printf "%5s | %4s(%5s) | %10f \n" "$i" "$num" "$numC" "$per"
      fi
    done
fi


directory=`ls | grep "^random_hyperopt_august2013_mod\>"`
if [ -a "${directory}" ]
then
    echo "Looking for RandomTPE:"

    printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
    for i in `seq 1000 1000 10000`
    do
      directory=`ls | grep "^random_hyperopt_august2013_mod_${i}_"`
      if [ -a "${directory}/random_hyperopt_august2013_mod.out" ]
      then
        num=`cat ${directory}/random_hyperopt_august2013_mod.out | grep " -----------------------RUNNING RUNSOLVER" | wc -l`

        per=`cat ${directory}/random_hyperopt_august2013_mod.out | grep "Result for ParamILS:" | sort -r | tail -1`
        per=`echo $per | cut -d' ' -f9`
        per=`echo $per | sed 's/,//'`

        numC=`ls ${directory}/ | grep 'instance.out$' | wc -l`
        numC=$(($numC - 1))

        test_error=999
        test_file=`ls -1 "$directory" | grep "_test_run.out"`
        printf "%5s | %4s(%5s) | %10f \n" "$i" "$num" "$numC" "$per"
      fi
    done
fi


directory=`ls | grep "^spearmint_april2013_mod\>"`
if [ -a "${directory}" ]
    then
    echo "Looking for Spearmint:"

    printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
    for i in `seq 1000 1000 10000`
    do
      directory=`ls | grep "^spearmint_april2013_mod_${i}_"`
      if [ -a "${directory}/spearmint_april2013_mod.out" ]
      then
        num=`cat ${directory}/spearmint_april2013_mod.out | grep " pending   " | tail -1`
        num=`echo $num | cut -d' ' -f5`
        per=`cat ${directory}/spearmint_april2013_mod.out | grep "best:" | tail -1`
        per=`echo $per | cut -d' ' -f3`

        numC=`ls ${directory}/ | grep 'instance.out$' | wc -l`
        numC=$(($numC - 1))

        test_error=999
        test_file=`ls -1 "$directory" | grep "_test_run.out"`
        if [ -f "$directory/$test_file" ]
        then
          test_error=`cat "$directory/$test_file" | grep "Result for ParamILS: SAT" | tail -1`
          test_error=`echo "$test_error" | cut -d' ' -f7| cut -d',' -f1`
        fi
        printf "%5s | %4s(%5s) | %10f | %10f\n" "$i" "$num" "$numC" "$per" "$test_error"
      fi
    done
fi
