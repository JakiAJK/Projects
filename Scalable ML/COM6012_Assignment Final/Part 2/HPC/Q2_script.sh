#!/bin/bash
#$ -l h_rt=1:00:00  #time needed
#$ -pe smp 5 #5 cores
#$ -l rmem=5G #5GB memery
#$ -P rse-com6012 # a com6012-reserved node
#$ -q rse-com6012.q # com6012 queue
#$ -o ../Output/Q2_Output.txt  # output and errors logs.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jkalla1@shef.ac.uk #Notify by email
#$ -m ea #Email when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 5g --executor-memory 5g --master local[5] ../Code/Q2_code.py