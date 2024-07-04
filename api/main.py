import sys
sys.path.insert(0, 'D:\Swinburne_study\Year2\semester3\COS30018\Project\source')

from source import HierarchialVAE, Denoise_net, Encoder_Block, Decoder_Block, DiffusionProcess


import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from fastapi.middleware.cors import CORSMiddleware

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

# entire_dataloader = torch.load('entire_dataloader.pth')
# vae_checkpoint = torch.load('../model/vae_checkpoint_amazon.pt')
# denoise_checkpoint = torch.load('../model/denoise_checkpoint_amazon.pt')

# Define the directory containing the checkpoint files
checkpoint_dir = '../source'

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


# VAE.load_state_dict(vae_checkpoint)
# Denoise_Net.load_state_dict(denoise_checkpoint)

# VAE.eval()
# Denoise_Net.eval()


@app.get("/")
def read_root():
    return {"message": "stock Price Predictor"}

# CORS: *
num_diff_steps = 10

@app.post("/predict")
async def predict(input: Filename):
    file_map = {
        "Amazon": "./Amazon_final.csv",
        "Apple": "./Apple_final.csv",
        "Google": "./Google_final.csv",
        "Microsoft": "./Microsoft_final.csv",
        "Netflix": "./Netflix_final.csv"
    }
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

    target_sequence=[]
    for i in range (0,len(tar)):
        for j in range(0,16):
            target_sequence.append(tar[i][j])
    pred_sequence=[]
    for i in range(0,len(predicted_seq)):
        for j in range(0,16):
            pred_sequence.append(predicted_seq[i][j])


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
    denorm_pred = stock['close'][6+5:len(tarcont_seq)+11]*predcont_seq
    return {"predictions": denorm_pred}
            # "target": denorm_tar}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)