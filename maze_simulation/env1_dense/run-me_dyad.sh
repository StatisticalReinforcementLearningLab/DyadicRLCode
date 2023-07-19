 #!/bin/bash

# Slurm parameters
PART=murphy,shared  	       # Partition names
MEMO=40960                     # Memory required (10G)
TIME=1:00:00                   # Time required (1h)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -p "$PART" --time="$TIME

LOGS=logs
mkdir -p $LOGS

for seed in {1..200}; do
    # Script to be run
    SCRIPT="script_dyad.sh $seed"
    # Define job name
    JOBN="job_seed"$seed
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"
    # Assemble slurm order for this job
    ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
    # Print order
    echo $ORD
    # Submit order
    $ORD
    #./$SCRIPT
done


  
