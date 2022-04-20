scriptPath="scripts3D/varied_smoke.py"
seeds=$(seq 0 9)

modes=$(seq 0 15) # only change when adding more varied parameters
amounts=$(seq 0 10) # always create a reference and 10 variations

for seed in $seeds;
do
    for mode in $modes;
    do
        for amount in $amounts;
        do
            build/manta $scriptPath $mode $amount $seed
        done
    done
done