#!/bin/bash



train=-1
dev=-1
test=-1

datasets=(spanish german dutch)
models=(linear latent_character)
prefix=(false true)

for (( d=0; d<${#datasets[@]}; d++  )) do
	dataset=${datasets[$d]}	
	for (( r=0; r<${#prefix[@]}; r++  )) do
		ucf=${prefix[$r]}
		for (( i=0; i<${#models[@]}; i++  )) do
			model=${models[$i]}
			logFile=logs/${model}_${dataset}_prefix_${ucf}.log
			java -cp statnlp-ner-1.0.jar org.statnlp.example.linear_ne.EMain --train_num ${train} --dev_num ${dev} --test_num ${test} -t 40 --dataset ${dataset} \
			                                                                 -it 8000 -ucf ${ucf} -mt ${model} > ${logFile} 2>&1
		done
	done
done
