scriptPath="scripts3D/varied_shapes.py"
seeds=$(seq 0 14)
dataset="Shapes"
modes=$(seq 0 3) # 4 varied parameters for shapes

#seeds=$(seq 0 29)
#dataset="Waves"
#modes=$(seq 0 1) # 2 varied parameters for waves

for seed in $seeds;
do
    for mode in $modes;
    do
        build/manta $scriptPath $mode $seed $dataset
    done
done