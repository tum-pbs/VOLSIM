export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/" #manually set correct cuda version
scriptPath="demos/varied_advDiff.py"
seeds=$(seq 0 23)
dataset="Train"
#seeds=$(seq 0 2)
#dataset="Test"

modes=$(seq 0 18) # only change when adding more varied parameters

for seed in $seeds;
do
    for mode in $modes;
    do
        python $scriptPath $mode $seed $dataset
    done
done