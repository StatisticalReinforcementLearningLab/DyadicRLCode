 #!/bin/bash

# Slurm parameters
PART=murphy,shared  	       # Partition names
MEMO=40960                     # Memory required (10G)
TIME=1:00:00                   # Time required (1h)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -p "$PART" --time="$TIME

LOGS=logs
mkdir -p $LOGS

for seed in {1..20}; do
for one in {2,4,6,8}; do
for two in {2,4,6,8}; do
if [ $two -gt $one ] || [ $two -eq $one ]
then
    # Script to be run
    SCRIPT="script_dyad.sh $seed $one $two"
    # Define job name
    JOBN="job_seed"$seed"_"$one"_"$two
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"
    # Assemble slurm order for this job
    ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
    # Print order
    echo $ORD
    # Submit order
    $ORD
    #./$SCRIPT
fi
done
done
done


  
