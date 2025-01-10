import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim),
        #     nn.Linear(encoding_dim, encoding_dim//2),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim//2, encoding_dim),
        #     nn.Linear(encoding_dim, input_dim),
        # )

        #shallower + sigmond
        self.encoder = nn.Sequential(
           nn.Linear(input_dim, encoding_dim),
           nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, input_dim),
        )
    
        # self.encoder = nn.Sequential(
        #    nn.Linear(input_dim, encoding_dim),
        #    nn.Linear(encoding_dim, encoding_dim*2),
        #    nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #    nn.Linear(encoding_dim*2, encoding_dim),
        #    nn.Linear(encoding_dim, input_dim),
        # )
    def forward(self, x):
        # TODO: 5%
        # Hint: a forward pass includes one pass of encoder and decoder
        coder = self.encoder(x)
        res = self.decoder(coder)
        return res
        # raise NotImplementedError
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        # Hint: a regular pytorch training includes:
        # 1. define optimizer
        # 2. define loss function
        # 3. define number of epochs
        # 4. define batch size
        # 5. define data loader
        # 6. define training loop
        # 7. record loss history 
        # Note that you can use `self(X)` to make forward pass.

        learning_rate = 0.005
        loss_func = nn.MSELoss()
        # optimizer
        # Adam
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # SGD
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.7)
        X_fit = torch.tensor(X).clone().detach().float()

        # data loader
        data_pre = torch.utils.data.TensorDataset(X_fit)
        data_load = torch.utils.data.DataLoader(data_pre, batch_size=batch_size, shuffle=True)


        loss_history = []

        # training loor
        for _ in tqdm(range(epochs), desc="fitting"):
            loss_num = 0
            k = 0
            for item in data_load:
                x = item[0]

                optimizer.zero_grad()
                gx = self.forward(x)
                loss = loss_func(gx, x)
                loss.backward()
                optimizer.step()

                loss_num += loss.item()
                k += 1
            loss_history.append(loss_num/k)

        # x_plot = range(1, epochs + 1)
        # plt.plot(x_plot , loss_history)
        # plt.xlabel('iterations')
        # plt.ylabel('Averaged Loss')
        # plt.title('Autoencoder')
        # plt.show()
        # raise NotImplementedError
    
    def transform(self, X):
        #TODO: 2%
        #Hint: Use the encoder to transofrm X
        X_tra = torch.tensor(X).clone().detach().float()
        temp = self.encoder(X_tra)
        result = temp.detach().numpy()
        return result
        # raise NotImplementedError
    
    def reconstruct(self, X):
        #TODO: 2%
        #Hint: Use the decoder to reconstruct transformed X
        X_rec = torch.tensor(X).clone().detach().float()
        temp = self.forward(X_rec)
        result = temp.detach().numpy()
        return result
        # raise NotImplementedError


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        #Hint: Generate Gaussian noise with noise_factor
        
        noise = self.noise_factor * torch.randn(*x.shape)
        noise = torch.clamp(noise, -1, 1)
        return noise
        # raise NotImplementedError
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        #Hint: Follow the same procedure above but remember to add_noise before training.
        learning_rate = 0.005
        loss_func = nn.MSELoss()
        # optimizer
        # Adam
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # SGD
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.7)
        X_fit = torch.tensor(X).clone().detach().float()

        # data loader
        data_pre = torch.utils.data.TensorDataset(X_fit)
        data_load = torch.utils.data.DataLoader(data_pre, batch_size=batch_size, shuffle=True)

        loss_history = []
        for _ in tqdm(range(epochs), desc="Denoising-fitting"):
            loss_num = 0
            k = 0
            for item in data_load:
                x = item[0]
                noise = self.add_noise(x) + x

                optimizer.zero_grad()
                gx = self.forward(noise)
                loss = loss_func(gx, x)
                loss.backward()
                optimizer.step()

                loss_num += loss.item()
                k += 1
            loss_history.append(loss_num / k)

        x_plot = range(1, epochs + 1)
        # plt.plot(x_plot, loss_history)
        # plt.xlabel('iterations')
        # plt.ylabel('Averaged Loss')
        # # plt.title('DenoisingAutoencoder')
        # #plt.title('DenoisingAutoencoder-Adam')
        # plt.title('DenoisingAutoencoder-SGD')
        # plt.show()       
        
        # raise NotImplementedError