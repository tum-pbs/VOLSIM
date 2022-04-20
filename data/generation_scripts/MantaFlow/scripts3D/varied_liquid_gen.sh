scriptPath="scripts3D/varied_liquid.py"
seeds=$(seq 0 9)
dataset="Train"
#seeds=$(seq 0 1)
#dataset="Test"

modes=$(seq 0 14) # only change when adding more varied parameters
amounts=$(seq 0 10) # always create a reference and 10 variations

for seed in $seeds;
do
    for mode in $modes;
    do
        for amount in $amounts;
        do
            build/manta $scriptPath $mode $amount $seed $dataset
        done
    done
done