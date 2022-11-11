#!/bin/bash
#$ -l h_rt=1:00:00  #time needed
#$ -pe smp 2 #2 cores
#$ -l rmem=10G #10GB memery
#$ -P rse-com6012 # a com6012-reserved node
#$ -q rse-com6012.q # com6012 queue
#$ -o ../Output/Q1_Output.txt  # output and errors logs.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jkalla1@shef.ac.uk #Notify by email
#$ -m ea #Email when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 10g --executor-memory 10g --master local[2] ../Code/Q1_code.py