#!/usr/bin/env bash

# Authors: Katharina Eggensperger and Matthias Feurer
# License: 3-clause BSD License
# Contact: "automl.org"

#minTimeTrace=0
#maxTimeTrace=0
#minErrorTrace=0
#maxErrorTrace=0
#minTrace=0
#maxTrace=0
#minBox=0
#maxBox=0
#specArgs=""

#LDA
#minTimeTrace=-1
#maxTimeTrace=16
#minErrorTrace=1260
#maxErrorTrace=1350
#minTrace=1000
#maxTrace=4000
#minBox=1260
#maxBox=1350
#specArgs=""

#SVM
#minTimeTrace=-1
#maxTimeTrace=12
#minErrorTrace=0.24
#maxErrorTrace=0.26
#minTrace=0.24
#maxTrace=0.3
#minBox=0.24
#maxBox=0.26
#specArgs=""

#Har6
#minTimeTrace=-1
#maxTimeTrace=12
#minErrorTrace=0
#maxErrorTrace=4
#minTrace=0
#maxTrace=4
#minBox=0
#maxBox=4
#specArgs=" -j"

#Branin
#minTimeTrace=-1
#maxTimeTrace=12
#minErrorTrace=0
#maxErrorTrace=1.5
#minTrace=0
#maxTrace=2
#minBox=0
#maxBox=2
#specArgs=" -b"


#Net convex
#minTimeTrace=-1
#maxTimeTrace=12
#minErrorTrace=0.15
#maxErrorTrace=0.5
#minTrace=0.15
#maxTrace=0.5
#minBox=0.15
#maxBox=0.3
#specArgs=""

#Net MRBI
minTimeTrace=-1
maxTimeTrace=12
minErrorTrace=0.45
maxErrorTrace=1
minTrace=0.45
maxTrace=1
minBox=0.45
maxBox=0.7
specArgs=""

#AutoWEKA
#minTimeTrace=-1
#maxTimeTrace=12
#minErrorTrace=20
#maxErrorTrace=50
#minTrace=20
#maxTrace=50
#minBox=20
#maxBox=50
#specArgs=""

#Logistic Regression
#minTimeTrace=0
#maxTimeTrace=0
#minErrorTrace=0.06
#maxErrorTrace=0.26
#minTrace=0.08
#maxTrace=0.28
#minBox=0.08
#maxBox=0.12
#specArgs=""

echo $1

smac="smac $1/smac_*_*/smac.pkl"
tpe="tpe $1/tpe_*_*/tpe.pkl"
spearmint="spearmint $1/spearmint_*_*/spearmint.pkl"
random="random $1/random_*_*/random.pkl"

#smac="smac $1/smac_*.pkl"
#tpe="tpe $1/tpe_*.pkl"
#spearmint="spearmint $1/spearmint_*.pkl"
#random="random $1/random_*.pkl"

# build arguments
argu=""
if [ -d "$1/smac" ]
then
    argu="$argu $smac"
fi

if [ -d "$1/tpe" ]
then
    argu="$argu $tpe"
fi

if [ -d "$1/spearmint" ]
then
    argu="$argu $spearmint"
fi

if [ -d "$1/random" ]
then
    argu="$argu $random"
fi

output="$1/Evaluation"
mkdir $output
echo "#########################################################################"
echo "Generating basic statistics"
python statistics.py $argu > "$output/statistics.txt"

echo "#########################################################################"
echo "Gather validation and test results"

now=`pwd`
cd $1
echo `LANG=en_us_8859_1 bash "$now/results.sh" .` > "$output/results.txt"
cd $now

echo "#########################################################################"
echo "Plotting the time trace of the optimizer"
python plotTimeTrace.py $argu -s "$output/TimeTraceLog.png" --min $minTimeTrace --max $maxTimeTrace
python plotTimeTrace.py $argu -s "$output/TimeTraceLog.pdf" --min $minTimeTrace --max $maxTimeTrace
python plotTimeTrace.py $argu -l -s "$output/TimeTrace.png" --min $minTimeTrace --max $maxTimeTrace
python plotTimeTrace.py $argu -l -s "$output/TimeTrace.pdf" --min $minTimeTrace --max $maxTimeTrace

echo "#########################################################################"
echo "Plot the error trace with standard deviation"
python plotTraceWithStd.py $argu -a -s "$output/ErrorTraceLog.png" --min $minErrorTrace --max $maxErrorTrace $specArgs
python plotTraceWithStd.py $argu -a -s "$output/ErrorTraceLog.pdf" --min $minErrorTrace --max $maxErrorTrace $specArgs
python plotTraceWithStd.py $argu --nolog -a -s "$output/ErrorTrace.png" --min $minErrorTrace --max $maxErrorTrace $specArgs
python plotTraceWithStd.py $argu --nolog -a -s "$output/ErrorTrace.pdf" --min $minErrorTrace --max $maxErrorTrace $specArgs

echo "#########################################################################"
echo "Plot the distribution of hyperparameters together with their response value"
mkdir $output/Params
python plot.py $tpe -s "$output/Params/TPE_para_"
python plot.py $smac -s "$output/Params/SMAC_para_"
python plot.py $spearmint -s "$output/Params/Spearmint_para_"
python plot.py MIX $1/smac_*_*/smac.pkl $1/tpe_*_*/tpe.pkl $1/spearmint_*_*/spearmint.pkl -s $output/Params/MIX_para_

echo "#########################################################################"
echo "Plot an optimization trace for every single optimization"
idx=0
opt=""

for pkl in $argu
do
    if [ -f $pkl ]
    then
      opt=`echo $pkl | rev| cut -d"/" -f1 | rev | cut -d"." -f1`
      #python plotTrace.py $opt $pkl -s "$output/Trace_${opt}_$idx.pdf" --min $minTrace --max $maxTrace $specArgs
      #python plotTrace.py $opt $pkl -s "$output/Trace_${opt}_$idx.png" --min $minTrace --max $maxTrace $specArgs
    fi
    idx=$((idx+1))
done

for i in `seq 1000 1000 10000`
do
    python plotTrace.py smac $1/smac_${i}_*/smac.pkl tpe $1/tpe_${i}_*/tpe.pkl spearmint $1/spearmint_${i}_*/spearmint.pkl -s "$output/Trace_all_$i.png" --min $minTrace --max $maxTrace $specArgs
    python plotTrace.py smac $1/smac_${i}_*/smac.pkl tpe $1/tpe_${i}_*/tpe.pkl spearmint $1/spearmint_${i}_*/spearmint.pkl -s "$output/Trace_all_$i.pdf" --min $minTrace --max $maxTrace $specArgs
    #python plotTrace.py smac $1/smac_${i}.pkl tpe $1/tpe_${i}.pkl spearmint $1/spearmint_${i}.pkl -s "$output/Trace_all_$i.png" --min $minTrace --max $maxTrace $specArgs
    #python plotTrace.py smac $1/smac_${i}.pkl tpe $1/tpe_${i}.pkl spearmint $1/spearmint_${i}.pkl -s "$output/Trace_all_$i.pdf" --min $minTrace --max $maxTrace $specArgs
done

echo "#########################################################################"
echo "Plot a box-whisker plot"
python plotBoxWhisker.py $argu -s "$output/BoxWhisker.png" --min $minBox --max $maxBox
python plotBoxWhisker.py $argu -s "$output/BoxWhisker.pdf" --min $minBox --max $maxBox






