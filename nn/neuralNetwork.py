import torch
import torch.nn as nn
import numpy as np
from data_featured.dataset_featured import DatasetFeatures
import matplotlib.pyplot as plt

class NN(object):
    def __init__(self, data:DatasetFeatures, utils, FPrate, discount, att_type):
        self.data = data
        self.utils = utils # function
        self.FPrate = FPrate
        self.discount = discount
        self.att_type = att_type

        self.BATCH_SIZE = 2 * 64
        self.LR_G = 0.0001  # learning rate for generator
        self.LR_D = 0.0001  # learning rate for discriminator
        self.latent = 10  # think of this as number of ideas for generating an art work (Generator)
        self.dim = len(self.data.features)  # it could be total point G can draw in the canvas


    def sample_data(self):     # painting from the famous artist (real target)
        samples = self.data.data.sample(self.BATCH_SIZE)
        samples = samples.as_matrix()
        # samples = samples[:, np.newaxis]
        return torch.from_numpy(samples).float()

    def solve(self):
        plt.ion() # something with continuous presentation of the plots

        G = nn.Sequential(                      # Generator
            nn.Linear(self.latent, 128),            # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(128, self.dim),
            nn.Sigmoid(),     # making a painting from these random ideas
        )

        D = nn.Sequential(                      # Discriminator
            nn.Linear(self.dim, 256),     # receive art work either from the famous artist or a newbie like G
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),                       # tell the probability that the art work is made by artist
        )

        self.reset_model(D)
        self.reset_model(G)

        opt_D = torch.optim.Adam(D.parameters(), lr=self.LR_D)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.LR_G)

        fig=plt.figure()
        for step in range(10000):
            real_samples = self.sample_data()           # real painting from artist
            latent_samples = torch.rand(self.BATCH_SIZE, self.latent)  # random ideas
            gen_samples = G(latent_samples)

            prob_of_insp_real = D(real_samples)  # D try to reduce this prob
            prob_of_insp_fake = D(gen_samples)  # D try to increase this prob

            #     D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
            #     D_loss = - ( - torch.mul(torch.clamp(torch.mean(prob_of_insp_real)-0.1, min=0),100) - torch.mean(torch.mul(1-prob_of_insp_fake, G_paintings))   )
            D_loss = - (-torch.mean(torch.mul(torch.clamp(prob_of_insp_real - 0.1, min=0), 100)) - torch.mean(
                torch.mul(1 - prob_of_insp_fake, gen_samples)))

            #     D_loss = torch.mean(torch.mul(prob_of_fake, G_paintings)) - torch.mean(torch.log(prob_of_real))
            #     G_loss = torch.mean(torch.log(1. - prob_of_fake))
            G_loss = - torch.mean(torch.mul(1 - prob_of_insp_fake, gen_samples))

            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # reusing computational graph
            opt_D.step()

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            if step % 200 == 0:  # plotting
                print("D_loss {}, FP {}, G_loss {}".format(D_loss, torch.mean(prob_of_insp_real), G_loss))
                print("FP {}, loss {}".format(torch.mean(prob_of_insp_real),
                    torch.mean(torch.mul(1 - prob_of_insp_fake, gen_samples))
                ))

                plt.clf()
                if self.data.feature_size == 1:

                    plt.hist(real_samples.data.numpy(), density=True)
                    plt.hist(gen_samples.data.numpy(), density=True)
                    plt.xlim([self.data.mins[0], self.data.maxs[0]])

                    D.eval()
                    t = torch.from_numpy(np.linspace(self.data.mins[0], self.data.maxs[0], 101)[:, np.newaxis]).float()
                    plt.plot(t.numpy().T[0], D(t).data.numpy().T[0], 'g-')
                    plt.plot(t.numpy().T[0], (1 - D(t).data.numpy().T[0]) * t.numpy().T[0], 'r-')
                    plt.ylim([0, 2])

                plt.show()
                plt.pause(0.0001)
                D.train()

        plt.clf()
        D.eval()
        plt.plot(t.numpy().T[0], D(t).data.numpy().T[0], 'g-')
        plt.plot(t.numpy().T[0], (1 - D(t).data.numpy().T[0]) * t.numpy().T[0], 'r-')
        plt.ylim([0, 1])
        plt.xlim([self.data.mins[0], self.data.maxs[0]])
        plt.plot(np.linspace(self.data.mins[0], self.data.maxs[0], 21), list(map(lambda x: max(0, 1 + G_loss / x), np.linspace(self.data.mins[0], self.data.maxs[0], 21))), 'b-')

        plt.show()
        D.train()

        plt.ioff()
        plt.show()

    def reset_model(self, M):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        M.apply(init_weights)
