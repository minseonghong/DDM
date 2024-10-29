from torch import nn
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import random
from fthd.model.build_network import build_network, string_to_torch, create_module
from abc import abstractmethod

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, features, labels,time_feature, time,scaler=None):
        self.X_data = torch.from_numpy(features).float().to(device)
        self.y_data = torch.from_numpy(labels).float().to(device)
        self.time_feature = torch.from_numpy(time_feature).float().to(device)

        self.time_line = torch.from_numpy(time).float().to(device)

        self.X_norm = torch.zeros(features.shape)
        num_instances, num_time_steps, num_features = features.shape
        train_data = features.reshape((-1, num_features))
        if scaler is None:
            self.scaler = StandardScaler()
            norm_train_data = self.scaler.fit_transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
        else:
            self.scaler = scaler
            norm_train_data = self.scaler.transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
    def __len__(self):
        return(self.X_data.shape[0])
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        time = self.time_line[idx]
        time_feature = self.time_feature[idx]
        x_norm = self.X_norm[idx]

        return x, y, time_feature, time, x_norm
    def split(self, percent):
        split_id = int(len(self)* percent)
        torch.manual_seed(0)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])

class FTHDDataset(DatasetBase):
    def __init__(self, features, labels,time_feature, time, scalers=None):
        super().__init__(features, labels,time_feature, time, scalers)

class ModelBase(nn.Module):
    def __init__(self, param_dict, output_module, eval=False):
        super().__init__()
        self.param_dict = param_dict
        layers = build_network(self.param_dict)
        self.batch_size = self.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        self.trained_coeff_ls = dict()
        self.trained_coeff = dict()
        self.dropout = nn.Dropout(0.25)
        
        self.P_vx = torch.ones(self.batch_size).numpy()
        self.P_vy = torch.ones(self.batch_size).numpy()
        self.P_yawRate = torch.ones(self.batch_size).numpy()
        
        if self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = self.param_dict["MODEL"]["HORIZON"]
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False
        self.horizon = self.param_dict["MODEL"]["HORIZON"]
        layers.extend(output_module)
        self.feed_forward = nn.ModuleList(layers)
        if eval:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]](reduction='none')
        else:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=self.param_dict["MODEL"]["OPTIMIZATION"]["LR"])
        self.epochs = self.param_dict["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(self.param_dict["STATE"])
        self.actions = list(self.param_dict["ACTIONS"])
        self.sys_params = list([*(list(p.keys())[0] for p in self.param_dict["PARAMETERS"])])
        self.vehicle_specs = self.param_dict["VEHICLE_SPECS"]

    @abstractmethod
    def differential_equation(self, x, output, time_feature, times):
        pass
    
    @abstractmethod
    def ekf_filter(self, x_label, x_pred, K):
        pass

    @abstractmethod
    def finite_difference(self, x, output, time_feature, times, epsilon=1e-4):
        pass
    
    def update_optimizer(self,dlr):
        self.optimizer = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](filter(lambda p: p.requires_grad, self.parameters()), lr=dlr*self.param_dict["MODEL"]["OPTIMIZATION"]["LR"])


    def forward(self, x, x_norm, x_label, h0_=None, time_feature=None, times=None):
        for i in range(len(self.feed_forward)):
            if i == 0:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0_ = self.feed_forward[0](x_norm, h0_)
                else:
                    ff = self.feed_forward[i](torch.reshape(x_norm, (len(x), -1)))
            else:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0_ = self.feed_forward[0](ff, h0_)
                else:
                    ff = self.feed_forward[i](ff)
                    # ff = self.dropout(ff)
        # ff = self.feed_forward[-1](ff)
        o, d_o, K , mass, self.sys_param_dict = self.differential_equation(x, ff, time_feature, times)
        x_upd, noise = self.ekf_filter(x_label, o, K)
        # o, d_o, jac = self.finite_difference(x,ff,time_feature,times)
        # print("x_upd:",x_upd)
        return x_upd, o, noise, h0_, d_o , mass , self.sys_param_dict


    def unpack_sys_params(self, o):
        sys_params_dict = dict()
        self.trained_coeff_ls.clear()
        for i in range(len(self.sys_params)):
            sys_params_dict[self.sys_params[i]] = o[:,i]
            self.trained_coeff_ls[self.sys_params[i]] = o[:,i]
        ground_truth_dict =  dict()
        for p in self.param_dict["PARAMETERS"]:
            ground_truth_dict.update(p)
        self.sys_params_dict = sys_params_dict
        return sys_params_dict, ground_truth_dict

    def unpack_state_actions(self, x):
        state_action_dict = dict()
        global_index = 0
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:,-1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:,-1, global_index]
            global_index += 1
        return state_action_dict

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
    
    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2)
    
    def get_trained_coeff(self):
        return self.trained_coeff
    
    def get_min_coeff(self,idx):
        for i in range(len(self.sys_params)):
            self.trained_coeff[self.sys_params[i]] = self.trained_coeff_ls[self.sys_params[i]][idx]

    
class FTHDModel(ModelBase):
    def __init__(self, param_dict, eval=False):

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation="Sigmoid")
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output

        
        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, time_feature, times):
            Ts = times - time_feature
            #print(Ts)
            sys_param_dict, _ = self.unpack_sys_params(output)
            state_action_dict = self.unpack_state_actions(x)
            steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
            throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
            mass = 1490
            self.sys_param_dict = sys_param_dict
            alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
            alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
            Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"]**2)*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
            Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
            Fry = sys_param_dict["Svr"] +sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
            dxdt = torch.zeros(len(x), 3).to(device)
            dxdt[:,0] = 1/sys_param_dict["Mass"] * (Frx + Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
            dxdt[:,1] = 1/sys_param_dict["Mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
            dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
            F_vx_vx = torch.ones(self.batch_size).to(device)
            dt = Ts.squeeze(1).data
            F_vx_vy = dt * state_action_dict["YAW_RATE"]
            F_vx_yawRate = dt * state_action_dict["VY"]
            F_vy_vy = torch.ones(self.batch_size).to(device)
            F_vy_vx = -dt * state_action_dict["YAW_RATE"]  # Updated
            F_vy_yawRate = -dt * state_action_dict["VX"]  # Updated
            F_yawRate_yawRate = torch.ones(self.batch_size).to(device)
            
            P_i_vx = torch.from_numpy(self.P_vx).to(device)
            P_i_vy = torch.from_numpy(self.P_vy).to(device)
            P_i_yawRate = torch.from_numpy(self.P_yawRate).to(device)
            
            P_vx = F_vx_vx * P_i_vx * F_vx_vx + F_vx_vy * P_i_vy * F_vx_vy + F_vx_yawRate * P_i_yawRate * F_vx_yawRate + sys_param_dict["Qvx"]
            P_vy = F_vy_vy * P_i_vy * F_vy_vy + F_vy_vx * P_i_vx * F_vy_vx + F_vy_yawRate * P_i_yawRate * F_vy_yawRate + sys_param_dict["Qvy"]
            P_yawRate = F_yawRate_yawRate * P_i_yawRate * F_yawRate_yawRate + sys_param_dict["QyawRate"]
            
            self.P_vx = P_vx.detach().cpu().numpy()
            self.P_vy = P_vy.detach().cpu().numpy()
            self.P_yawRate = P_yawRate.detach().cpu().numpy()
            # print("mass",sys_param_dict["Mass"])
            K_vx = P_vx / (P_vx + sys_param_dict["Rvx"])
            K_vy = P_vy / (P_vy + sys_param_dict["Rvy"])
            K_yawRate = P_yawRate / (P_yawRate + sys_param_dict["RyawRate"])
            
            ax = dxdt
            dxdt = dxdt * Ts
            return x[:,-1,:3] + dxdt, ax, [K_vx,K_vy,K_yawRate] , sys_param_dict["Mass"] , self.sys_param_dict

    
    def ekf_filter(self, x_label, x_pred, K):       
        # print("x pred shape:",x_pred_diag.shape)
        # print("x label shape:",x_label_diag.shape)
        residual = x_label - x_pred
        vx_res = residual[:,0]
        vy_res = residual[:,1]
        yawRate_res = residual[:,2]
        
        vx_update = x_pred[:,0] + K[0] * vx_res
        vy_update = x_pred[:,1] + K[1] * vy_res
        yawRate_update = x_pred[:,2] + K[2] * yawRate_res
        
        vx_noise = K[0] * vx_res
        vy_noise = K[1] * vy_res
        yawRate_noise = K[2] * yawRate_res
        
        
        # vx_update = x_pred[:,0]
        # vy_update = x_pred[:,1]
        # yawRate_update = x_pred[:,2]
        
        self.P_vx = (1 - K[0].detach().cpu().numpy()) * self.P_vx
        self.P_vy = (1 - K[1].detach().cpu().numpy()) * self.P_vy
        self.P_yawRate = (1 - K[2].detach().cpu().numpy()) * self.P_yawRate
        
        return torch.stack((vx_update,vy_update,yawRate_update),dim=1),torch.stack((vx_noise,vy_noise,yawRate_noise),dim=1)
    
    def stable_matrix_inverse(self, matrix):
    # Use Cholesky decomposition for matrix inversion if applicable
        try:
            L = torch.linalg.cholesky(matrix).mH
            inv_matrix = torch.cholesky_inverse(L)
            return inv_matrix
        except RuntimeError as e:
            # Fallback to torch.inverse if Cholesky decomposition fails
            return torch.inverse(matrix)

string_to_model = {
    "FTHD" : FTHDModel,
    "FTHDIAC" : FTHDModel
}

string_to_dataset = {
    "FTHD" : FTHDDataset,
    "FTHDIAC" : FTHDDataset

}
