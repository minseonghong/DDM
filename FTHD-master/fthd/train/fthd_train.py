import wandb
from torch.autograd import grad
from ray import train as raytrain
from fthd.model.models_supervised import string_to_model,string_to_dataset
import torch
import numpy as np
import os
import yaml
import pickle
import os
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_usecuda_use")
else:
    device = torch.device("cpu")
    print("cpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_usecpu_use")

def train(model, train_data_loader, val_data_loader, experiment_name, log_wandb, output_dir, project_name=None, use_ray_tune=False):
    print("Starting experiment: {}".format(experiment_name))
    parameters_ls = list([*(list(p.keys())[0] for p in model.param_dict["PARAMETERS"])])
    error_threshold = 1e-12
    best_coeff = dict()

    if log_wandb:
        if model.is_rnn:
            architecture = "RNN"
            gru_layers = model.param_dict["MODEL"]["LAYERS"][0]["LAYERS"]
            hidden_layer_size = model.param_dict["MODEL"]["LAYERS"][1]["OUT_FEATURES"]
            hidden_layers = len(model.param_dict["MODEL"]["LAYERS"]) - 2
        else:
            architecture = "FFNN"
            gru_layers = 0
            hidden_layer_size = model.param_dict["MODEL"]["LAYERS"][0]["OUT_FEATURES"]
            hidden_layers = len(model.param_dict["MODEL"]["LAYERS"]) - 1
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name = experiment_name,
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": model.param_dict["MODEL"]["OPTIMIZATION"]["LR"],
            "hidden_layers" : hidden_layers,
            "hidden_layer_size" : hidden_layer_size,
            "batch_size" : model.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"],
            "timestamps" : model.param_dict["MODEL"]["HORIZON"],
            "architecture": architecture,
            "gru_layers": gru_layers
            }
        )
        wandb.watch(model, log='all')
    valid_loss_min = torch.inf
    qualified_loss_min = torch.inf
    
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    weights = torch.tensor([1.0, 1.0, 1.0]).to(device)

    loss_items = torch.zeros(2,1)

    def df_error(output, acc_output,time_):

        vxyt_sum = output[:,:2].mean()
        # vyt_sum = output[:,1].mean()
        uvxy_t = grad(vxyt_sum, time_, grad_outputs=torch.ones_like(vxyt_sum), only_inputs=True, retain_graph=True)[0]
        
        uvxy_t -= acc_output[:,:2].mean()

        return uvxy_t
    
    for i in range(model.epochs):
        train_steps = 0
        train_loss_accum = 0.0
        alpha_item = torch.zeros(2,1).to(device)
        
        if model.is_rnn:
            h = model.init_hidden(model.batch_size)

        for features, labels, time_feature, times, norm_inputs in train_data_loader:

            model.optimizer.zero_grad()

            features,labels = features.to(device),labels.to(device)
            times, time_feature,norm_inputs = times.to(device),time_feature.to(device),norm_inputs.to(device)

            if model.is_rnn:
                h = h.data
            if model.is_rnn:
                output, h, acc_output = model(features, norm_inputs, h, time_feature, times)
            else:
                output, _, acc_output = model(features, norm_inputs ,None, time_feature, times)
                
            loss = model.weighted_mse_loss(output, labels, weights).mean()
            
            train_loss_accum += loss.item()
            
            train_steps += 1
            loss.backward()

            model.optimizer.step()
        model.eval()
        val_steps = 0
        val_loss_accum = 0.0
        val_vx_accum = 0.0
        val_vy_accum = 0.0
        val_yawr_accum = 0.0
        max_val_vx = -torch.inf
        max_val_vy = -torch.inf
        max_val_yaw = -torch.inf
        if model.is_rnn:
            val_h = model.init_hidden(model.batch_size)

        for val_f, val_la, val_timef, val_times, val_normin in val_data_loader:

            val_f, val_la, val_times = val_f.to(device), val_la.to(device), val_times.to(device)
            val_timef = val_timef.to(device)
            val_normin = val_normin.to(device)
            if model.is_rnn:
                val_h = val_h.data
                out, val_h, _ = model(val_f, val_normin, val_h,val_timef, val_times)
            else:
                out, _, _ = model(val_f, val_normin,None, val_timef, val_times)

            val_loss = model.weighted_mse_loss(out, val_la, weights)
            val_vx_accum += val_loss[:,0].mean().item()/weights[0].item()
            val_vy_accum += val_loss[:,1].mean().item()/weights[1].item()
            val_yawr_accum += val_loss[:,2].mean().item()/weights[2].item()
            
            
            max_vx, _ = val_loss[:,0].max(dim=0)
            max_vy, _ = val_loss[:,1].max(dim=0)
            max_yaw, _ = val_loss[:,2].max(dim=0)
            
            max_vx /= weights[0].item()
            max_vy /= weights[1].item()
            max_yaw /= weights[2].item()
            
            
            if torch.sqrt(max_vx) > max_val_vx:
                max_val_vx=torch.sqrt(max_vx)
                
            if torch.sqrt(max_vy) > max_val_vy:
                max_val_vy=torch.sqrt(max_vy)
                
            if torch.sqrt(max_yaw) > max_val_yaw:
                max_val_yaw=torch.sqrt(max_yaw)
            
            min_val, min_index = val_loss[1].min(dim=0)

            if min_val < qualified_loss_min:

                qualified_loss_min = min_val
                model.get_min_coeff(min_index)
            val_loss = val_loss.mean()

            val_loss_accum += val_loss.item()
            val_steps += 1
            
        # print(train_steps)
        mean_train_loss = train_loss_accum / train_steps
        mean_val_loss = val_loss_accum / val_steps
        rmse_val_vx = np.sqrt(val_vx_accum / val_steps)
        rmse_val_vy = np.sqrt(val_vy_accum / val_steps)
        rmse_val_yawr = np.sqrt(val_yawr_accum / val_steps)
        
        if log_wandb:
            wandb.log({"train_loss": mean_train_loss })
            wandb.log({"val_loss": mean_val_loss})
            wandb.log({"finetuning epoch": 0})
            wandb.log({"training epoch": i+1})
        if mean_val_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,mean_val_loss))
            torch.save(model.state_dict(), "%s/best_before_model.pth" % (output_dir))
            valid_loss_min = mean_val_loss
            best_coeff = model.get_trained_coeff()
            print("best DDM val loss:",mean_val_loss)
            if log_wandb:
                wandb.log({"best_DDM_val_loss" : mean_val_loss})
                wandb.log({"best_val_loss" : mean_val_loss})
                
                coeff = []
                for i in range(len(parameters_ls)):
                    coeff.append(best_coeff[parameters_ls[i]])
                    # wandb.log({"%s:" %(parameters_ls[i]): best_coeff[parameters_ls[i]]})
                wandb.log({"DDM param list":coeff})
                wandb.log({"DDM rmse vx":rmse_val_vx})
                wandb.log({"DDM max vx":max_val_vx})
                wandb.log({"DDM rmse vy":rmse_val_vy})
                wandb.log({"DDM max vy":max_val_vy})
                wandb.log({"DDM rmse yawRate":rmse_val_yawr})
                wandb.log({"DDM max yawRate":max_val_yaw})
                
                wandb.log({"param list":coeff})
                wandb.log({"rmse vx":rmse_val_vx})
                wandb.log({"max vx":max_val_vx})
                wandb.log({"rmse vy":rmse_val_vy})
                wandb.log({"max vy":max_val_vy})
                wandb.log({"rmse yawRate":rmse_val_yawr})
                wandb.log({"max yawRate":max_val_yaw})
        print("Epoch: {}/{}...".format(i+1, model.epochs),
            "Loss: {:.6f}...".format(mean_train_loss),
            "Val Loss: {:.6f}".format(mean_val_loss),
            "loss1: {:.6f}, loss2: {:.6f}, alpha1: {:.6f}, alpha2: {:.6f}".format(loss_items[0].item(),loss_items[1].item(),alpha_item[0].item(),alpha_item[1].item()))
        
        if i % 100 ==0:
            trained_coeff = model.get_trained_coeff()
            # torch.save(model.state_dict(), "%s/epoch_%s.pth" % (output_dir, i+1))
            for i in range(len(parameters_ls)):
                print("trained_coeff %s is %f :" %(parameters_ls[i], trained_coeff[parameters_ls[i]]))
        if use_ray_tune:
            checkpoint_data = {
                "epoch": i,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
            }
            raytrain.report(
                {"loss": mean_train_loss},
            )
        if np.isnan(mean_train_loss):
            break

        if mean_val_loss <= error_threshold:
            break
        model.train()
        
    qualified_loss_min = torch.inf
    
    model.load_state_dict(torch.load("%s/best_before_model.pth" % (output_dir)))
    for param in model.parameters():
        param.requires_grad = True
        
    freeze_layers_num = int(len(model.feed_forward)*3/4)
    for i in range(freeze_layers_num):
        if i % 2 == 0:
            for param in model.feed_forward[i].parameters():
                param.requires_grad = False
    print("freeze layers num:",freeze_layers_num)
    for i in range(freeze_layers_num):
        for param in model.feed_forward[i].parameters():
            param.requires_grad = False
    
    model.update_optimizer(dlr = 1)
        
    additional_epoches = int(model.epochs)
    model.train()
    for i in range(additional_epoches):
        train_steps = 0
        train_loss_accum = 0.0
        alpha_item = torch.zeros(2,1).to(device)
            
        if model.is_rnn:
            h = model.init_hidden(model.batch_size)

        for features, labels, time_feature, times, norm_inputs in train_data_loader:

            model.optimizer.zero_grad()
            features,labels = features.to(device),labels.to(device)
            times, time_feature,norm_inputs = times.to(device),time_feature.to(device),norm_inputs.to(device)
            times = times.requires_grad_(True)
            time_feature = time_feature.requires_grad_(True)
            if model.is_rnn:
                h = h.data
            if model.is_rnn:
                output, h, acc_output = model(features, norm_inputs, h, time_feature, times)
            else:
                output, _, acc_output = model(features, norm_inputs ,None, time_feature, times)

            loss1 = model.weighted_mse_loss(output, labels, weights)
            
            loss1 = loss1.mean()
            loss2 = (df_error(output, acc_output,times)**2).mean()
            times.detach()
            loss = 0.999975* loss1 + 0.000025*loss2
            train_loss_accum += loss.item()
            train_steps += 1
            loss.backward()
            # del output, acc_output, loss1, loss2
            # gc.collect()
            model.optimizer.step()
        model.eval()
        val_steps = 0
        val_loss_accum = 0.0
        val_vx_accum = 0.0
        val_vy_accum = 0.0
        val_yawr_accum = 0.0
        max_val_vx = -torch.inf
        max_val_vy = -torch.inf
        max_val_yaw = -torch.inf
        if model.is_rnn:
            val_h = model.init_hidden(model.batch_size)
        
        for val_f, val_la, val_timef, val_times, val_normin in val_data_loader:
            

            val_f, val_la, val_times = val_f.to(device), val_la.to(device), val_times.to(device)
            val_timef = val_timef.to(device)
            val_normin = val_normin.to(device)
            if model.is_rnn:
                val_h = val_h.data
                out, val_h, _ = model(val_f, val_normin, val_h,val_timef, val_times)
            else:
                out, _, _ = model(val_f, val_normin,None, val_timef, val_times)

            val_loss = model.weighted_mse_loss(out, val_la, weights)
            val_vx_accum += val_loss[:,0].mean().item()/weights[0].item()
            val_vy_accum += val_loss[:,1].mean().item()/weights[1].item()
            val_yawr_accum += val_loss[:,2].mean().item()/weights[2].item()
            
            max_vx, _ = val_loss[:,0].max(dim=0)
            max_vy, _ = val_loss[:,1].max(dim=0)
            max_yaw, _ = val_loss[:,2].max(dim=0)
            
            max_vx /= weights[0].item()
            max_vy /= weights[1].item()
            max_yaw /= weights[2].item()
            
            if torch.sqrt(max_vx) > max_val_vx:
                max_val_vx=torch.sqrt(max_vx)
                
            if torch.sqrt(max_vy) > max_val_vy:
                max_val_vy=torch.sqrt(max_vy)
                
            if torch.sqrt(max_yaw) > max_val_yaw:
                max_val_yaw=torch.sqrt(max_yaw)
            
            min_val, min_index = val_loss[1].min(dim=0)

            if min_val < qualified_loss_min:

                qualified_loss_min = min_val
                model.get_min_coeff(min_index)
            val_loss = val_loss.mean()
            val_loss_accum += val_loss.item()
            val_steps += 1
        mean_train_loss = train_loss_accum / train_steps
        mean_val_loss = val_loss_accum / val_steps
        rmse_val_vx = np.sqrt(val_vx_accum / val_steps)
        rmse_val_vy = np.sqrt(val_vy_accum / val_steps)
        rmse_val_yawr = np.sqrt(val_yawr_accum / val_steps)
        
        if log_wandb:
            wandb.log({"train_loss": mean_train_loss })
            wandb.log({"val_loss": mean_val_loss})
            wandb.log({"training epoch": model.epochs})
            wandb.log({"finetuning epoch": i+1})
        if mean_val_loss < valid_loss_min:
            print('Fine Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,mean_val_loss))
            torch.save(model.state_dict(), "%s/finetuned_model.pth" % (output_dir))
            valid_loss_min = mean_val_loss
            best_coeff = model.get_trained_coeff()
            print("best val loss:",mean_val_loss)
            if log_wandb:
                wandb.log({"best_val_loss" : mean_val_loss})
                coeff = []
                for i in range(len(parameters_ls)):
                    coeff.append(best_coeff[parameters_ls[i]])
                    # wandb.log({"%s:" %(parameters_ls[i]): best_coeff[parameters_ls[i]]})
                wandb.log({"param list":coeff})
                
                wandb.log({"rmse vx":rmse_val_vx})
                wandb.log({"max vx":max_val_vx})
                wandb.log({"rmse vy":rmse_val_vy})
                wandb.log({"max vy":max_val_vy})
                wandb.log({"rmse yawRate":rmse_val_yawr})
                wandb.log({"max yawRate":max_val_yaw})

            if coeff:
                txt_file_path = os.path.join(output_dir, "best_vehicle_parameters.txt")
                with open(txt_file_path, 'w') as f:
                    f.write("Best Vehicle Parameters (validation loss: {:.9f}):\n".format(valid_loss_min))
                    
                    # 리스트일 경우 인덱스와 값으로 처리
                    for idx, param_value in enumerate(coeff):
                        f.write(f"Parameter {idx}: {param_value}\n")
                
                print(f"______________________Best vehicle parameters saved to_______________________ {txt_file_path}")


        print("Fine Epoch: {}/{}...".format(i+1, additional_epoches),
            "Loss: {:.6f}...".format(mean_train_loss),
            "Val Loss: {:.6f}".format(mean_val_loss),
            "loss1: {:.6f}, loss2: {:.6f}, alpha1: {:.6f}, alpha2: {:.6f}".format(loss_items[0].item(),loss_items[1].item(),alpha_item[0].item(),alpha_item[1].item()))
        
        if i % 100 ==0:
            trained_coeff = model.get_trained_coeff()
            for i in range(len(parameters_ls)):
                print("trained_coeff %s is %f :" %(parameters_ls[i], trained_coeff[parameters_ls[i]]))
        if use_ray_tune:
            checkpoint_data = {
                "epoch": i,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
            }
            raytrain.report(
                {"loss": mean_train_loss},
            )
        if np.isnan(mean_train_loss):
            break

        if mean_val_loss <= error_threshold:
            break
        model.train()
        
    if best_coeff:
        txt_file_path = os.path.join(output_dir, "best_vehicle_parameters.txt")
        with open(txt_file_path, 'w') as f:
            f.write("Best Vehicle Parameters (validation loss: {:.9f}):\n".format(valid_loss_min))
            for param_name, param_value in best_coeff.items():
                f.write(f"{param_name}: {param_value}\n")
        
        print(f"Best vehicle parameters saved to {txt_file_path}")   


    wandb.finish()
    for i in range(len(parameters_ls)):
        print("trained_coeff %s is %f :" %(parameters_ls[i], best_coeff[parameters_ls[i]]))
    print("val loss : {:.6f}".format(valid_loss_min))
    
if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Train a deep dynamics model.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset_file", type=str, help="Dataset file for model")
    
    dirname = os.path.split(os.path.dirname(__file__))[0]
    
    print(dirname)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    experiment_name = "fthd_test"

    dataset_file = argdict["dataset_file"]
    
    data_npy = np.load(dataset_file)

    with open(argdict["model_cfg"], 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    
    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"],
                                                            data_npy["times_features"],data_npy["times"])
    
    # val_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](val_data_npy["features"], val_data_npy["labels"],
    #                                                         val_data_npy["times_features"],val_data_npy["times"])
    
    if not os.path.exists("../output"):
        os.mkdir("../output")
    if not os.path.exists("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0])):
        os.mkdir("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]))
    output_dir = "../output/%s/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0], experiment_name)
    if not os.path.exists(output_dir):
         os.mkdir(output_dir)
         
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
    if not os.path.exists("%s/%s" % (output_dir,formatted_time)):
        os.makedirs("%s/%s" % (output_dir,formatted_time))
        
    output_dir = os.path.join(output_dir,formatted_time)
    train_dataset, _ = dataset.split(0.9)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size, shuffle=False, drop_last=True)

    error_threshold = 1e-6
    best_coeff = dict()
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset.scaler, f)

    train(model, train_data_loader, val_data_loader, experiment_name, False,output_dir,None,use_ray_tune=True)
        

    
