import yaml
import os
from functools import partial
import torch
import numpy as np

from fthd.model.models_supervised import string_to_dataset, string_to_model
from fthd.train.fthd_train import train

from ray import tune
import pickle
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import os
from datetime import datetime

def main(model_cfg, log_wandb):
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
    dir_name = os.path.split(os.path.dirname(__file__))[0]
    
    model_cfg_path = os.path.join(dir_name,"cfgs",model_cfg)

    config = {
        "layers" : tune.choice(range(2,10)),
        "neurons" : tune.randint(4, 256),
        "batch_size": tune.choice([32,64,128]),
        "lr" : tune.uniform(1e-4, 1e-2),
        "horizon": tune.choice(range(1,20)),
        "gru_layers": tune.choice(range(0,5))
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=100,
        grace_period=100,
    )
    result = tune.run(
        partial(tune_hyperparams, model_cfg=model_cfg_path, log_wandb=log_wandb,formatted_time=formatted_time),
        metric='loss',
        mode='min',
        search_alg=OptunaSearch(),
        resources_per_trial={"cpu": 3, "gpu": 3/6},
        config=config,
        num_samples=200,
        max_concurrent_trials=1,
        scheduler=scheduler,
        storage_path=os.path.join(dir_name,"ray_results"),
        
        stop={"training_iteration":100}
        # checkpoint_at_end=True
    )

def tune_hyperparams(hyperparam_config, model_cfg, log_wandb,formatted_time):
    dir_name = os.path.split(os.path.dirname(__file__))[0]
    ##denoised 데이터를 파싱한 1~20 horizon의 npz 파일 가져옴
    dataset_file = os.path.join(dir_name,"output","fthd_iac_ekf","2024-10-15_13_55_08","9layers_85neurons_128batch_0.006984lr_19horizon_3gru","denoised_csv_{}RNN_hong_val.npz".format(hyperparam_config["horizon"]))
    
    project_name = "opensource_test_IAC"

    with open(model_cfg, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
    
    experiment_name = "%dlayers_%dneurons_%dbatch_%flr_%dhorizon_%dgru" % (hyperparam_config["layers"], 
                                                                           hyperparam_config["neurons"], 
                                                                           hyperparam_config["batch_size"], 
                                                                           hyperparam_config["lr"], 
                                                                           hyperparam_config["horizon"], 
                                                                           hyperparam_config["gru_layers"])
    if not os.path.exists(os.path.join(dir_name,'output')):
        os.mkdir(os.path.join(dir_name,'output'))
    if not os.path.exists(os.path.join(dir_name,"output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0]))):
        os.mkdir(os.path.join(dir_name,"output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0])))
    output_dir = os.path.join(dir_name,"output/%s/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0], formatted_time))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
            
    
    if not os.path.exists("%s/%s" % (output_dir,experiment_name)):
        os.makedirs("%s/%s" % (output_dir,experiment_name))
        
    output_dir = os.path.join(output_dir,experiment_name)
    
    param_dict["MODEL"]["LAYERS"] = []
    if hyperparam_config["gru_layers"]:
        layer = dict()
        layer["GRU"] = None
        layer["OUT_FEATURES"] = hyperparam_config["horizon"] ** 2
        layer["LAYERS"] = hyperparam_config["gru_layers"]
        param_dict["MODEL"]["LAYERS"].append(layer)
    for i in range(hyperparam_config["layers"]):
        layer = dict()
        layer["DENSE"] = None
        layer["OUT_FEATURES"] = hyperparam_config["neurons"]
        layer["ACTIVATION"] = "Mish"
        param_dict["MODEL"]["LAYERS"].append(layer)
    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = hyperparam_config["batch_size"]
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = hyperparam_config["lr"]
    param_dict["MODEL"]["HORIZON"] = hyperparam_config["horizon"]
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    data_npy = np.load(dataset_file)

    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"],
                                                            data_npy["times_features"],data_npy["times"])
    
    # val_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](val_data_npy["features"], val_data_npy["labels"],
    #                                                         val_data_npy["times_features"],val_data_npy["times"])
    
    train_dataset, _ = dataset.split(0.05)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparam_config["batch_size"], shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=hyperparam_config["batch_size"], shuffle=False, drop_last=True)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset.scaler, f)
    train(model, train_data_loader, val_data_loader, 
          experiment_name, log_wandb, output_dir, 
          project_name, use_ray_tune=True)
if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("--log_wandb", action='store_true', help="Log experiment in wandb")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    
    main(argdict["model_cfg"], argdict["log_wandb"])
