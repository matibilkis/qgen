"""
Map a value from ~U(-1, 1) to ~N(0, 1),
http://theoryandpractice.org/stats-ds-book/distributions/change-of-variables.html

Given x~p(x) ---> y = f(x),   p(y) = p(x) / (dy/dx); this is because
the probability of an interval should be preserved (basic calculus).

Now if we take x~p(x),   let y = F(x) = \int_{0}^x p(z) dz.
then   p(y) = p(x) / (dy/dx) = 1.  Which means that p(y) is the uniform distribution in [0,1], since \int_{0}^1 p(y) = \int_{0}^1 dy = y|_0^1 = 1

All this to say that if we take the inverse of the cumulative (of x) acting on y (which is uniform), then we should get p(x)!!!!

The inverse of the cumulative is ppf



there is something w/ -torch.sum(torch.log(discriminator(real_labels))) - torch.sum(torch.log(1-discriminator(fake_labels)))
... it doesn't match what i have in BCE loss!
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats



def uniform_to_normal(z):
    norm = stats.norm(0, 1)
    return norm.ppf((z+1)/2)

def generate_noise(samples, dimensions=1):
    return np.random.uniform(-1, 1, (samples, dimensions))


### Generator
class Generator(torch.nn.Module):
    def __init__(self, input_size=1):
        super(Generator, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.linear_input = torch.nn.Linear(input_size, 16)
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, 16)
        self.linear4 = torch.nn.Linear(16, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.linear3(x)
        x = self.leaky_relu(x)
        x = self.linear4(x)
        return x


### Generator
class Discriminator(torch.nn.Module):
    def __init__(self, input_size=1):
        super(Discriminator, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.linear_input = torch.nn.Linear(input_size, 64)
        self.linear1 = torch.nn.Linear(64, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.sigmoid(self.linear3(x))
        return x






from tqdm import tqdm

generator = Generator()
discriminator = Discriminator()
disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=0.002, betas=(.5, .999))
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=0.001, betas=(.5, .999))


num_batches = 600
batch_size = 512
grid_resolution = 400
loss_disc = torch.nn.BCELoss()
loss_gen = torch.nn.BCELoss()

history = {}
history["cost_g"]=[]
history["cost_d"]=[]

for step in tqdm(range(int(num_batches+ 3e4))):
    disc_optimizer.zero_grad()
    real_data = torch.tensor(uniform_to_normal(generate_noise(batch_size//2)),dtype=torch.float32)
    fake_data = generator(torch.tensor(generate_noise(batch_size//2),dtype=torch.float32))

    real_labels = torch.ones(real_data.shape)
    fake_labels = torch.zeros(fake_data.shape)

    data = torch.concatenate([real_data, fake_data], axis=0)
    labels = torch.concatenate([real_labels,fake_labels],axis=0)

    cost_d = loss_disc(discriminator(data), labels)
    cost_d.backward()

    ## generator
    gen_optimizer.zero_grad()
    noise = torch.tensor(generate_noise(batch_size,1),dtype=torch.float32)
    labels = torch.ones(batch_size,1)
    cost_g = loss_gen(discriminator(noise), labels)
    cost_g.backward()

    history["cost_g"].append(cost_g.detach())
    history["cost_d"].append(cost_d.detach())
plt.plot(torch.tensor(history["cost_g"]))



def give_histogram(samples,bins=100):
    c,b = np.histogram(samples,bins=bins,density=True)
    w = b[1]-b[0]
    x = np.linspace(np.min(b),np.max(b),bins)
    return x,c,w

x,c,w = give_histogram(fake_data.detach().numpy().squeeze())
ax=plt.subplot()
ax.bar(x,c,width=w, color="red")



list(generator.parameters())


step_count = []
D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
count = 0
for step in range(NUM_BATCHES):
    print(f'1b: {step}/{NUM_BATCHES}', end='\r')
    # Train discriminator
    D.trainable = True
    real_data = uniform_to_normal(generate_noise(BATCH_SIZE // 2, LATENT_DIM))
    fake_data = G.predict(generate_noise(BATCH_SIZE // 2, LATENT_DIM), batch_size=BATCH_SIZE // 2)
    data = np.concatenate((real_data, fake_data), axis=0)
    real_labels = np.ones((BATCH_SIZE // 2, 1))
    fake_labels = np.zeros((BATCH_SIZE // 2, 1))
    labels = np.concatenate((real_labels, fake_labels), axis=0)
    _D_loss, _D_accuracy = D.train_on_batch(data, labels)

    # Train generator
    D.trainable = False
    noise = generate_noise(BATCH_SIZE, LATENT_DIM)
    labels = np.ones((BATCH_SIZE, 1))
    _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)

    if step % PLOT_EVERY == 0:
        step_count.append(step)
        D_loss.append(_D_loss)
        D_accuracy.append(_D_accuracy)
        G_loss.append(_G_loss)
        G_accuracy.append(_G_accuracy)
        plot(G=G,
             D=D,
             step=step+1,
             step_count=step_count,
             D_accuracy=D_accuracy,
             D_loss=D_loss,
             G_accuracy=G_accuracy,
             G_loss=G_loss,
             filename=f'ims/1_1D_normal/b/1b.{count:03d}.png')
        count += 1
print()
os.system(f'ffmpeg -r 20 -i ims/1_1D_normal/b/1b.%03d.png'
              f' -crf 15 ims/1_1D_normal/b/1b.mp4')

total_steps = 30000
for step in range(NUM_BATCHES, total_steps):
    print(f'1c: {step}/{total_steps}', end='\r')
    # Train discriminator
    D.trainable = True
    real_data = uniform_to_normal(generate_noise(BATCH_SIZE // 2, LATENT_DIM))
    fake_data = G.predict(generate_noise(BATCH_SIZE // 2, LATENT_DIM), batch_size=BATCH_SIZE // 2)
    data = np.concatenate((real_data, fake_data), axis=0)
    real_labels = np.ones((BATCH_SIZE // 2, 1))
    fake_labels = np.zeros((BATCH_SIZE // 2, 1))
    labels = np.concatenate((real_labels, fake_labels), axis=0)
    _D_loss, _D_accuracy = D.train_on_batch(data, labels)

    # Train generator
    D.trainable = False
    noise = generate_noise(BATCH_SIZE, LATENT_DIM)
    labels = np.ones((BATCH_SIZE, 1))
    _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)

    if step % PLOT_EVERY == 0:
        step_count.append(step)
        D_loss.append(_D_loss)
        D_accuracy.append(_D_accuracy)
        G_loss.append(_G_loss)
        G_accuracy.append(_G_accuracy)

plot(G=G,
     D=D,
     step=step+1,
     step_count=step_count,
     D_accuracy=D_accuracy,
     D_loss=D_loss,
     G_accuracy=G_accuracy,
     G_loss=G_loss,
     filename=f'ims/1_1D_normal/c/1c.{total_steps:03d}.png')
G.save("ims/1_1D_normal/c/G.h5")
D.save("ims/1_1D_normal/c/D.h5")



grid_latent = np.linspace(-1, 1, 103)[1:-1].reshape((-1, 1))
true_mappings = uniform_to_normal(grid_latent)
GAN_mapping = G.predict(grid_latent)

plt.figure(figsize=(6,6))
plt.scatter(grid_latent.flatten(), true_mappings.flatten(),
            edgecolor='blue', facecolor='None', s=5, alpha=1,
            linewidth=1, label='Inverse CDF Mapping')
plt.scatter(grid_latent.flatten(), GAN_mapping.flatten(),
            edgecolor='red', facecolor='None', s=5, alpha=1,
            linewidth=1, label='Inverse CDF Mapping')
plt.xlim(-1, 1)
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(f'ims/1_1D_normal/c/1c.true.png')
plt.close()










def give_histogram(samples,bins=100):
    c,b = np.histogram(samples,bins=bins,density=True)
    w = b[1]-b[0]
    x = np.linspace(np.min(b),np.max(b),bins)
    return x,c,w

x,c,w = give_histogram(real_data.squeeze())
ax=plt.subplot()
ax.bar(x,c,width=w, color="red")







#




#
