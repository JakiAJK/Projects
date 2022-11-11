#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 3 #number of cores
#$ -l rmem=15G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o ../Output/P2Q2_Output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jkalla1@shef.ac.uk #Notify by email
#$ -m ea #Email when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 15g --executor-memory 15g --master local[3] ../Code/P2Q2.py