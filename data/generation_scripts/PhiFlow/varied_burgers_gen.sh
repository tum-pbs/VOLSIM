export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/" #manually set correct cuda version
scriptPath="demos/varied_burgers.py"
seeds=$(seq 0 26)

modes=$(seq 0 16) # only change when adding more varied parameters

for seed in $seeds;
do
    for mode in $modes;
    do
        python $scriptPath $mode $seed
    done
done