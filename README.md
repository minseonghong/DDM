# DDM


# Installation

Package is tested in Ubuntu20.04, working both cpu and gpu, please install cuda toolkit before gpu training, Ubuntu system recommended for wandb and raytune hyperparameter training.

It is recommended to create a new conda environment:


  conda create --name fthd python=3.10

  conda activate fthd

To install FTHD:


  git clone https://github.com/minseonghong/DDM.git
  cd fthd/
  pip install -e .

# Dataset

Real data:

  fthd/data/IAC_EKF_DATA/Putnam_park2023_run4_2.csv

MORAI simulation train data:

  fthd/data/SIMULATION_DATA/simulation_train_dataset.csv

MORAI simulation evaluation data:

  fthd/data/SIMULATION_DATA/simulation_double_lane_change_eval.csv

# Data preprocessing

cd tools

Real data:

    for i in {1..20}; do
      python csv_fthd_timeverify.py ../data/SIMULATION_DATA/Putnam_park2023_run4_2.csv $i
    done
  
MORAI simulation data:

  for i in {1..20}; do
      python csv_fthd_timeverify.py ../data/SIMULATION_DATA/simulation_train_dataset.csv $i
  done

npz file will be stored in fthd/data/SIMULATION_DATA/{csv file name}_{1..20}RNN_Val.npz

# Train

To run the training code:

  cd train

  python fthd_ekf.py {absolute path to yaml} {absolute path to npz}

For example:

  python fthd_ekf.py ../FTHD-master/fthd/cfgs/fthd_iac_ekf.yaml ../FTHD-master/fthd/data/SIMULATION_DATA/simulation_train_dataset_19RNN_Val.npz

The trained model will be stored in fthd/output/fthd_iac_ekf/supervised_test/{YYYY}-{MM}-{DD}_{HH}_{MM}_{SS}/finetuned_model.pth

# Evaluation

To evaluation the model in double lane change scenario:

  cd real_report_data
  cd benchmark
  
  python speed_qualified_eval.py 

The estimated tire model's parameters will be stored in fthd/real_report_data/benchmark/output/Best_Vehicle_Parameters.txt.



# Reference paper

@ARTICLE{FangFTHD2024,

AUTHOR = "Shiming Fang and Kaiyan Yu",

TITLE = "{Fine-Tuning Hybrid Physics-Informed Neural Networks for Vehicle Dynamics Model Estimation}",

JOURNAL = "arXiv preprint arXiv:2409.19647",

YEAR = "2024"

}
