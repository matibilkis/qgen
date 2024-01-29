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
\usepackage{amssymb}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\disc}{D_{\bm{\phi}}}
\newcommand{\gen}{G_{\bm{\theta}}}
\newcommand{\kin}{k_{in}}
\newcommand{\kout}{k_{out}}
\newcommand{\preal}{p_{\text{real}}}
\newcommand{\pz}{p_{z}}
\newcommand{\min}[1]{underset{[#1]}{\text{min}\;}}
\newcommand{\max}[1]{underset{[#1]}{\text{max}\;}}
\newcommand{\equ}[1]{\begin{equation}[#1]\end{equation}}

Goal: train a quantum circuit to become a pdf sampler (here, normal distribution).
## (?) JUSTIFICATION? WHY WOULD IT BE BETTER TO USE A QUANTUM CIRCUIT FOR THIS?
## 1. load probability distribution into quantum state ---> then manipulate it w/ quantum routine (where you can get some advantage). EXAMPLE? (HHL, QAE..)

### ALTERNATIVES ?? Variational Autoencoders... difussion models... which are pros and cons?  Comment! Quantum counterpart ?

Brief outline:

Let $X = {x^0, ...x^{s-1}} \subseteq \mathbb{R}^{\kout} \sim \preal$, with \preal unknown, our goal is to prepare a quantum state that encodes the pdf \preal (or some approximated version of it).

Our 'input' is - aside from the training set X, some noise z~ \pz, which can be seen as a 'prior' pdf, living in a $\kin$-dimensional space.

Let $\gen : \RR^{\kin}\rightarrow\RR^{\kout}$ be the \textit{generator} (an agent maping samples $z\sim\pz$ towards a pdf [which is hopefully close to the true one].

Let \disc:\RR^{k_{out}}\rightarrow [0,1] be the discriminator (*1), which takes samples obtained from a pdf and discerns if those were obtained from the real (values closer to 0) one or otherwise fake (values closer to 1). In other words, $\disc(\bm{y})$ retrieves the probability of $\bm{y}$ being sampled from $\preal$, according the the tester $\disc$, aimed to be optimized [1].(*2)

Informally, the problem is encoded into a cost function of the form:
\begin{align}
COST &=  TERM_1  + TERM_2 \\
TERM_1 &---> D(x) \text{classified as 1, if }x\sim\preal \\
TERM_2 &---> G(z)\\
\end{align}
gets so close to $\preal$ that D can't distinguish. Here, if you are the discriminator you would like to minimize your error probability,

but if you are the generator you would like to maximize it (to fool the discriminator!)

The discrimination-generator training can be formulated in terms of a minimax optimization problem, given by
\equ{\max{D}\min{G} \mathbb{E}_{x\sim\preal}[\log D(x)] + \mathbb{E}_{z\sim\pz}[\log (1 - D(G(z)))]}.

Note that the second term stands for the probability that the discriminator deems $G(z)$ fake: whereas the discriminator shall maximize such value, the generator aims to minimize it.

In practice, this procedure is carried out sequentially: the discriminator is trained so to perform very good for a fixed \gen, and then the generator is optimized to beat such generator but only slightly, since otherwise it can lead to instabilities [1]. Moreover, at early stages, the discriminator can easily discern whether samples $G(z)$ are obtained from $\preal$ or not, since $\gen$ will be very bad. Since the generator needs to minimize the quantity $\mathbb{E}_{z\sim\pz}[\log (1 - D(G(z)))]}$, which is \textit{saturated} at the value of $1$, oen can consider the maximization of $\mathbb{E}_{z\sim\pz}\log(D(G(z)))$ wrt to the generator; as it can easily be checked, the gradients will be higher in the warm-up scenario.

Hence, the following cost functions are taking into account:

\begin{align}
L_D &= - \big( \mathbb{E}_{x\sim\preal}[\log D(x)] + \mathbb{E}_{z\sim\pz}[\log (1 - D(G(z)))]} \big) \\
L_G &= -\mathbb{E}_{z\sim\pz}\log(D(G(z)))
\end{align}

\section{A quantum generator}
While the generator ussually consists on a NN, [2] proposes to replace it with a parametrized quantum circuit (PQC). As highlighted by the authors, the importance of having a PQC trained to sample from a target $\preal$ is that of \textit{loading} classical data into the quantum state (and afterwards perform quantum infor processing routines such as HHL or QAE)[3]


%%%% REFERENCES %%%
(*1) There should be a (0,1] instead of a {0,1} in the definition of the paper in Zoufal paper (Qgans for loadingsampling pdfs) since the log(\disc) is otherwise undefined.
(*2) Has a rigurous hypothesis testing formulation been made with these GANs? Connection w/ Matera ?

[1] https://arxiv.org/abs/1406.2661
[2] https://www.nature.com/articles/s41534-019-0223-2
[3] Let us highlight that the complexity n-bits into a quantum state is O(2^n) in terms of #gates, whereas as claimed by Zoufal the QGAN requires O(Poly(n)). This is a trap however, because of BPs.
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
