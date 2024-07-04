# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
# import torch

# # Load the saved model weights
# vae_checkpoint = torch.load('vae_checkpoint_amazon.pt')
# denoise_checkpoint = torch.load('denoise_checkpoint_amazon.pt')

# # Create API instance
# app = FastAPI()


from model import HierarchialVAE, Denoise_net, Encoder_Block, Decoder_Block, Diffusion_Process

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

app = FastAPI()

class StockData(BaseModel):
    open: float
    high: float
    close: float
    volume: float
    close_normalized: float

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

        output = self.data.iloc[index + self.sequence_length: index + self.sequence_length + self.prediction_length]['close_normalized'].values.tolist()
        output = torch.Tensor(output)

        return input, output

# # Load your pre-trained model (assuming you have it saved as 'model.pth')
# model = torch.load('model.pth')
# model.eval()

# Load the saved model weights
vae_checkpoint = torch.load('vae_checkpoint_amazon.pt')
denoise_checkpoint = torch.load('denoise_checkpoint_amazon.pt')

VAE = HierarchialVAE(Encoder_Block = Encoder_Block, Decoder_Block = Decoder_Block , latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36,
                 feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12)
Denoise_Net = Denoise_net(in_channels = 16,dim = 16, size = 5)


VAE.load_state_dict(vae_checkpoint)
Denoise_Net.load_state_dict(denoise_checkpoint)

VAE.eval()
Denoise_Net.eval()


@app.get("/")
def read_root():
    return {"message": "Car Price Predictor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the CSV file. Please check the file content.")

    # Ensure the required columns are present
    required_columns = ['open', 'high', 'close', 'volume']
    if not all(column in df.columns for column in required_columns):
        raise HTTPException(status_code=400, detail="Missing required columns")

    # Normalize the 'close' column
    df['close_normalized'] = (df['close'] - df['close'].mean()) / df['close'].std()

    # Define sequence and prediction lengths
    sequence_length = 12
    prediction_length = 5

    # Create an instance of the StockDataset
    sequenced_data = StockDataset(df, sequence_length, prediction_length)

    # Create a DataLoader for batching
    dataloader = DataLoader(sequenced_data, batch_size=1, shuffle=False)

    predictions = []
    for inputs, _ in dataloader:
        with torch.no_grad():
            vae_out = torch.zeros((inputs.size(0), inputs.size(1), num_diff_steps))
            diff_out = torch.zeros((inputs.size(0), inputs.size(1), num_diff_steps))
            for time in range(1, num_diff_steps + 1):
                output, y_noisy = Diffusion_Process.diffuse(inputs, inputs, time)  # Dummy diffusion function
                vae_out[:, :, time - 1] = output
                diff_out[:, :, time - 1] = y_noisy

            mean_vae = torch.mean(vae_out, dim=2)
            predictions.append(mean_vae.numpy().tolist())

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)