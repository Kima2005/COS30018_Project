import time as time2
from source.model import HierarchialVAE, Denoise_net, Encoder_Block, Decoder_Block, DiffusionProcess
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error

app = FastAPI()

class Filename(BaseModel):
    filename: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_length):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_length

    def __getitem__(self, index):
        input_data = self.data.iloc[index: index + self.sequence_length]
        input_list = input_data.values.tolist()
        input = torch.Tensor(input_list)

        output = self.data.loc[index + self.sequence_length : index + self.sequence_length + self.prediction_length-1, 'Close Normalized'].values.tolist()
        output = torch.Tensor(output)

        return input, output
    
# Define the directory containing the checkpoint files
checkpoint_dir = './source'
checkpoint_dir2 = './source2'
# Define lists to hold the loaded models and dataloaders
vae_checkpoints = {}
denoise_checkpoints = {}
dataloaders = {}

# List of sources to loop through
sources = ['amazon', 'apple', 'gg', 'meta', 'netflix']



# Load VAE and denoise checkpoints
for source in sources:
    vae_path = os.path.join(checkpoint_dir, f'vae_checkpoint_{source}.pt')
    denoise_path = os.path.join(checkpoint_dir, f'denoise_checkpoint_{source}.pt')
    
    vae_checkpoints[source] = torch.load(vae_path)
    denoise_checkpoints[source] = torch.load(denoise_path)

# Load entire dataloaders
for source in sources:
    dataloader_path = os.path.join(checkpoint_dir, f'entire_dataloader_{source}.pth')
    dataloaders[source] = torch.load(dataloader_path)


VAE = HierarchialVAE(Encoder_Block = Encoder_Block, Decoder_Block = Decoder_Block , latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36,
                 feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12)
Diffusion_Process = DiffusionProcess(num_diff_steps = 10, vae = VAE, beta_start = 0.01, beta_end = 0.1, scale = 0.5)
Denoise_Net = Denoise_net(in_channels = 16,dim = 16, size = 5)


# Load WGAN model ########################3
Wgan_model ={}
index_test_load = {}
index_train_load = {}
# X_scaler = {}
X_test_load = {}
X_train_load = {}
y_scaler_load = {}
y_test_load = {}
y_train_load = {}

# Load 
for source in sources:
    wgan_path = os.path.join(checkpoint_dir2, f'WGAN_{source}.keras')
    print(wgan_path)
    Wgan_model[source] =  tf.keras.models.load_model(wgan_path)

for source in sources:
    index_test_path = os.path.join(checkpoint_dir2, f'index_test_{source}.npy')
    index_test_load[source] = np.load(index_test_path, allow_pickle=True)
    
    index_train_path = os.path.join(checkpoint_dir2, f'index_train_{source}.npy')
    index_train_load[source] = np.load(index_train_path, allow_pickle=True)
    
    x_test_path = os.path.join(checkpoint_dir2, f'X_test_{source}.npy')
    X_test_load[source] = np.load(x_test_path, allow_pickle=True)
    
    x_train_path = os.path.join(checkpoint_dir2, f'X_train_{source}.npy')
    X_train_load[source] = np.load(x_train_path, allow_pickle=True)
    
    y_test_path = os.path.join(checkpoint_dir2, f'y_test_{source}.npy')
    y_test_load[source] = np.load(y_test_path, allow_pickle=True)
    
    y_train_path = os.path.join(checkpoint_dir2, f'y_train_{source}.npy')
    y_train_load[source] = np.load(y_train_path, allow_pickle=True)

    
    # X_scaler_path = os.path.join(checkpoint_dir2, f'X_scaler{source}.npy')
    # X_scaler[source] = load(open(X_scaler_path, 'rb'))

        
    y_scaler_path = os.path.join(checkpoint_dir2, f'y_scaler_{source}.pkl')
    y_scaler_load[source] = load(open(y_scaler_path, 'rb'))








@app.get("/")
def read_root():
    return {"message": "stock Price Predictor"}

# CORS: *
num_diff_steps = 10
file_map = {
    "Amazon": "./data/Amazon_final.csv",
    "Apple": "./data/Apple_final.csv",
    "Google": "./data/Google_final.csv",
    "Microsoft": "./data/Microsoft_final.csv",
    "Netflix": "./data/Netflix_final.csv"
}



@app.post("/predict")
async def predict(input: Filename):
    start_time = time2.time()
    if input.filename not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file name.")


    vae_state_dict = vae_checkpoints.get(input.filename.lower())
    denoise_state_dict = denoise_checkpoints.get(input.filename.lower())
    VAE.load_state_dict(vae_state_dict)
    Denoise_Net.load_state_dict(denoise_state_dict)
    VAE.eval()
    Denoise_Net.eval()
    entire_dataloader = dataloaders.get(input.filename.lower())

    stock = pd.read_csv(file_map[input.filename])
    predicted_seq=[]
    inp_seq=[]
    tar=[]

    for i,(x,y) in enumerate(entire_dataloader):
        if(x.size(0)!=16):
            break
        vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
        diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
        for time in range(1,num_diff_steps + 1):
            output, y_noisy = Diffusion_Process.diffuse(x,y,time)
            vae_out[:,:,time-1] = output
            diff_out[:,:,time-1] = y_noisy
        y_nn=vae_out[:,:,:]
        E = Denoise_Net(y_nn).sum()
        grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0]
        mean_vae = torch.mean(vae_out, dim = 2)
        inp_seq.append(x)
        predicted_seq.append(mean_vae - 0.1*torch.mean(grad_x,dim=2))
        tar.append(y)

    # print(predicted_seq[:10])
    target_sequence=[]
    for i in range (0,len(tar)):
        for j in range(0,16):
            target_sequence.append(tar[i][j])
    pred_sequence=[]
    for i in range(0,len(predicted_seq)):
        for j in range(0,16):
            pred_sequence.append(predicted_seq[i][j])
    # print(pred_sequence[:10])

    tarcont_seq=[]
    for i in range(0,len(target_sequence)):
        if(i%5==0):
            tarcont_seq.append(target_sequence[i])
    tarcont_seq = [item.item() for sublist in tarcont_seq for item in sublist]
    predcont_seq=[]
    for i in range(0,len(pred_sequence)):
        if(i%5==0):
            predcont_seq.append(pred_sequence[i])
    predcont_seq = [item.item() for sublist in predcont_seq for item in sublist]
    
    # denorm_tar = stock['close'][6+5:len(tarcont_seq)+11]*tarcont_seq
    denorm_pred = stock['Close'][6+5:len(tarcont_seq)+11]*predcont_seq
    print("--- %s seconds ---" % (time2.time() - start_time))
    print(denorm_pred)
    return {"predictions": denorm_pred}
            # "target": denorm_tar}



@app.post("/predict2")
async def predict2(input: Filename):
    start_time = time2.time()
    if input.filename not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    G_model = Wgan_model.get(input.filename.lower())
    y_test = y_test_load.get(input.filename.lower())
    X_test = X_test_load.get(input.filename.lower())
    X_train = X_train_load.get(input.filename.lower())
    y_train = y_train_load.get(input.filename.lower())
    y_scaler = y_scaler_load.get(input.filename.lower())
    index_test = index_test_load.get(input.filename.lower())
    index_train = index_train_load.get(input.filename.lower())
    print("FUCKKKKKKKKKKKKKKKKKK")
    print(y_test)
    print(y_test_load)
    print(input.filename.lower())
    # Set output steps
    output_dim = y_test.shape[1]

    # Get predicted data
    y_predicted = G_model(X_test)
    y_train_predicted = G_model(X_train)
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)

    rescaled_real_y_train = y_scaler.inverse_transform(y_train)
    rescaled_predicted_y_train = y_scaler.inverse_transform(y_train_predicted)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_price"],
                                 index=index_test[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    ## Real price
    real_price = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_price"], index=index_test[i:i + output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    predict_result_train = pd.DataFrame()
    for i in range(rescaled_predicted_y_train.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y_train[i], columns=["predicted_price"], index=index_train[i:i+output_dim])
        predict_result_train = pd.concat([predict_result_train, y_predict], axis=1, sort=False)

    real_price_train = pd.DataFrame()
    for i in range(rescaled_real_y_train.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y_train[i], columns=["real_price"], index=index_train[i:i+output_dim])
        real_price_train = pd.concat([real_price_train, y_train], axis=1, sort=False)

    predict_result_train['predicted_mean'] = predict_result_train.mean(axis=1)
    real_price_train['real_mean'] = real_price_train.mean(axis=1)

    predict_final = pd.concat([predict_result_train['predicted_mean'], predict_result['predicted_mean']], axis=0)
    # real_final = pd.concat([real_price_train['real_mean'], real_price['real_mean']], axis=0)


    print(predict_final)
    


    return {"predictions": predict_final.tolist()}
            # "target": denorm_tar}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)