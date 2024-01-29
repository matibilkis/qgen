import qiskit
import qiskit_algorithms
import torch
import matplotlib.pyplot as plt
import numpy as np


qiskit.__version__ # version 0.45.2

qiskit_algorithms.random_seed = 123456
_ = torch.manual_seed(123456)  # suppress output


num_dim = 1
num_discrete_values = 8
num_qubits = num_dim * int(np.log2(num_discrete_values))

"""
\newcommand{\RR}{\mathbb{R}}
\newcommand{\disc}{D_{\bm{\phi}}}
\newcommand{\gen}{G_{\bm{\theta}}}

Goal: train a quantum circuit to become a pdf sampler (here, normal distribution).
## (?) JUSTIFICATION? WHY WOULD IT BE BETTER TO USE A QUANTUM CIRCUIT FOR THIS?
## 1. load probability distribution into quantum state ---> then manipulate it w/ quantum routine (where you can get some advantage). EXAMPLE? (HHL, QAE..)

### ALTERNATIVES ?? Variational Autoencoders... difussion models... which are pros and cons?  Comment! Quantum counterpart ?

Brief outline:

Let X = {x^0, ...x^{s-1}} \included \mathbb{R}^{k_out} \sim p_{real}, with p_{real} unknown, our goal is to prepare a quantum state \rho \equiv \rho(p_real).

Let \gen : \RR^{k_{in}}\rightarrow\RR^{k_{out}} be the \textit{generator} (an agent maping samples from a prior pdf towards a pdf [hopefully close to the true one], and \disc:\RR^{k_{out}}\rightarrow (0,1] be the discriminator (*1), which takes samples obtained from a pdf and discerns if those were obtained from the real (values closer to 1) one or otherwise fake (values closer to 0).

Each agent has a given goal, encoded in a cost function:

L_G = - \mathbb{E}_{\bm{z}\sim p_{prior}} \log{\disc(\gen(\bm{z}))},

which maximizes the likelihood that $\gen$ generates samples labeled as real. Thus, the discriminator \textit{competes} with $\gen$, with a cost function defined as

L_D = \mathbb{E}_{x\sim p_{real}} [\log \disc(\bm{x})] + \mathbb{E}_{z \sim \p_{prior}} \log [1 - \disc(G_\theta(\bm{z}))]

(*1) There should 
"""



def normal(x,mu=0.0, sigma = 1):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(4*np.pi*sigma**2)

coords = np.linspace(-2, 2, num_discrete_values)
probs = np.array([normal(x) for x in coords])

plt.plot(coords, probs)


### Building the quantum generator

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

qc = QuantumCircuit(num_qubits)
qc.h(qc.qubits) #this applies a Haddamard to each qubit
ansatz = EfficientSU2(num_qubits, reps=2)
qc.compose(ansatz, inplace=True)

qc.decompose().draw("mpl")

qc.num_parameters
from qiskit.primitives import Sampler
shots = 10000
sampler = Sampler(options={"shots": shots, "seed": qiskit_algorithms.random_seed})

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN


def create_generator() -> TorchConnector: #this is the way Qiskit use quantum circuit (sampler QNN, etc)
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=[],
        weight_params=qc.parameters,
        sparse=False,
    )
    initial_weights = np.random.random(qc.num_parameters)
    return TorchConnector(qnn, initial_weights)


### DISCRIMINATOR


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.linear_input = torch.nn.Linear(input_size, 8)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.linear20 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.leaky_relu(x)
        x = self.linear20(x)
        x = self.sigmoid(x)
        return x

generator = create_generator()
discriminator = Discriminator(num_dim)

### LOSS
def adversarial_loss(input, target, w):
    bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    weighted_loss = w * bce_loss
    total_loss = -torch.sum(weighted_loss)
    return total_loss


from torch.optim import Adam

lr = 0.01  # learning rate
b1 = 0.7  # first momentum parameter
b2 = 0.999  # second momentum parameter

generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)


from IPython.display import clear_output

def plot_training_progress():
    # we don't plot if we don't have enough data
    if len(generator_loss_values) < 2:
        return

    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Generator Loss
    ax1.set_title("Loss")
    ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
    ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
    ax1.legend(loc="best")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.grid()

    # Relative Entropy
    ax2.set_title("Relative entropy")
    ax2.plot(entropy_values)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Relative entropy")
    ax2.grid()

    plt.show()


import time
from scipy.stats import entropy

n_epochs = 50
num_qnn_outputs = num_discrete_values**num_dim


generator_loss_values = []
discriminator_loss_values = []
entropy_values = []
start = time.time()


from tqdm import tqdm
for k in tqdm(range(400)):

    valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float)
    fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float)

    # Configure input
    real_dist = torch.tensor(probs, dtype=torch.float).reshape(-1, 1)

    # Configure samples
    samples = torch.tensor(np.expand_dims(coords,-1), dtype=torch.float)
    disc_value = discriminator(samples)

    # Generate data
    gen_dist = generator(torch.tensor([])).reshape(-1, 1)

    # Train generator
    generator_optimizer.zero_grad()
    generator_loss = adversarial_loss(disc_value, valid, gen_dist)

    # store for plotting
    generator_loss_values.append(generator_loss.detach().item())

    generator_loss.backward(retain_graph=True)
    generator_optimizer.step()

    # Train Discriminator
    discriminator_optimizer.zero_grad()

    real_loss = adversarial_loss(disc_value, valid, real_dist)
    fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
    discriminator_loss = (real_loss + fake_loss) / 2

    # Store for plotting
    discriminator_loss_values.append(discriminator_loss.detach().item())

    discriminator_loss.backward()
    discriminator_optimizer.step()

    entropy_value = entropy(gen_dist.detach().squeeze().numpy(), probs)
    entropy_values.append(entropy_value)


plt.plot(coords,np.squeeze(gen_dist.detach()))
plt.plot(coords,probs)


plt.plot(discriminator_loss_values)
plt.plot(generator_loss_values)
