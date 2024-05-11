dir="sbatch_log"
job_File="run_fednoro.sh" 
robust_method=$"fednoro"
dataset=$"cifar100"
architecture=$"resnet34"
alpha=$"5"
epchs=$"300"

for seed in 1 2 3
do
    for noisy_client_ratio in 0.4 0.7 1.0
    do 
        for minimum_noise in 0.2 0.5
        do
            EXPT="$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise"
            STD=$dir/STD_"$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise".out
            ERR=$dir/STD_"$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise".err
            export noisy_client_ratio;
            export minimum_noise;
            export robust_method;
            export dataset;
            export architecture;
            export alpha;
            export seed;
            export epochs;
            sbatch -J $EXPT -o $STD -t 1-00:00:00 -e $ERR $job_File
        done;
    done;

    for noisy_client_ratio in 0.0
    do
        for minimum_noise in 0.0
        do
            EXPT="$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise"
            STD=$dir/STD_"$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise".out
            ERR=$dir/STD_"$robust_method"_"$dataset"_"$noisy_client_ratio"_"$minimum_noise".err
            export noisy_client_ratio;
            export minimum_noise;
            export robust_method;
            export dataset;
            export architecture;
            export alpha;
            export seed;
            export epochs;
            sbatch -J $EXPT -o $STD -t 1-00:00:00 -e $ERR $job_File
        done;
    done;   
done