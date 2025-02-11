mkdir -p ./save
mkdir -p ./trainlogs

method=sdmgrad
seed=0
rho=0.1
niter=1

nohup python -u trainer.py --method=$method --seed=$seed --rho=$rho --niter=$niter > trainlogs/sdmgrad-lambda$lamda-sd$seed.log 2>&1 &
