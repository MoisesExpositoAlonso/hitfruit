#!/bin/bash

##  Reserve X CPUs for this job
##$ -pe parallel 1
#
# Request  RAM
#$ -l h_vmem=1G
#
#Request not to send it to the biggest node
#$ -l mem_total_lt=600G
#
#  Request it to run this long HH:MM:SS
#$ -l h_rt=00:01:00
#
#  Use /bin/bash to execute this script
#$ -S /bin/bash
#
#  Run job from current working directory
#$ -cwd
#
#  Redirect outputs
#$ -o ../skeletons/logs
#$ -e ../skeletons/logs
#
###  Send email when the job begins, ends, aborts, or is suspended
##$ -m beas

s=$1
NAME=${s##*/}

./runhitfruit.py $* > '../skeletons/logs/'$NAME'.log'
