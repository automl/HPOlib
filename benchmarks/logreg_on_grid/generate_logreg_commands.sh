rm logreg_commands.txt

for optimizer in "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_06_01-dev" "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_08_00-master" "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/tpe/hyperopt_august2013_mod" "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/tpe/random_hyperopt_august2013_mod" "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/spearmint/spearmint_april2013_mod"
do
  for seed in `seq 1000 1000 10000`
  do
    echo "HPOlib-run -o $optimizer -s $seed" >> logreg_commands.txt
  done
done


rm init_logreg.txt

echo 'source ~/virtualenvs/development/bin/activate' >> init_logreg.txt
echo 'export PATH=$PATH:/home/feurerm/mhome/HPOlib/Software/bin_cluster' >> init_logreg.txt
echo 'export PYTHONPATH=$PYTHONPATH:/home/feurerm/mhome/HPOlib/Software/HPOlib' >> init_logreg.txt
echo 'export PATH=$PATH:/home/feurerm/mhome/HPOlib/Software/HPOlib/scripts' >> init_logreg.txt
