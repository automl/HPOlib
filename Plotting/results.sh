#!/bin/sh

echo "Looking for SMAC:"

printf "%5s | %4s(%5s) | %5s | %10s | %10s\n" "Seed" "#run" "crash" "#iter" "Performance" "Test-Perf"
for i in `seq 1000 1000 10000`
do
  directory=`ls | grep "^smac_${i}_"`
  if [ -f "${directory}/smac.out" ]
  then
    it=`cat ${directory}/smac.out | grep "Model/Iteration used:" | tail -1`
    it=`echo $it | cut -d' ' -f3`
   
    per=`cat ${directory}/smac.out | grep "Performance of the Incumbent:" | tail -1`
    per=`echo $per | cut -d' ' -f5`

    num=`cat ${directory}/smac.out | grep "Algorithm Runs used:" | tail -1`
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

echo "Looking for TPE:"

printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
for i in `seq 1000 1000 10000`
do
  directory=`ls | grep "^tpe_${i}_"`
  if [ -a "${directory}/tpe.out" ]
  then
    num=`cat ${directory}/tpe.out | grep "Result:" | wc -l`

    per=`cat ${directory}/tpe.out | grep "Result:" | sort -r | tail -1`
    per=`echo $per | cut -d' ' -f2`

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

echo "Looking for Spearmint:"

printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
for i in `seq 1000 1000 10000`
do
  directory=`ls | grep "^spearmint_${i}_"`
  if [ -a "${directory}/spearmint.out" ]
  then
    num=`cat ${directory}/spearmint.out | grep " pending   " | tail -1`
    num=`echo $num | cut -d' ' -f5` 
    per=`cat ${directory}/spearmint.out | grep "best:" | tail -1`
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

echo "Looking for Random:"

printf "%5s | %4s(%5s) | %10s | %10s\n" "Seed" "#run" "crash" "Performance" "Test-Perf"
for i in `seq 1000 1000 10000`
do
  directory=`ls | grep "^randomtpe_${i}_"`
  if [ -a "${directory}/randomtpe.out" ]
  then
    num=`cat ${directory}/randomtpe.out | grep "Result:" | wc -l`

    per=`cat ${directory}/randomtpe.out | grep "Result:" | sort -r | tail -1`
    per=`echo $per | cut -d' ' -f2`

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

