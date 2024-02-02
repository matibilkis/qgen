
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
\newcommand{\qgen}{\mathcal{G}_{\bm{\theta}}}
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\ketgen}{\ket{g_{\bm{\theta}}}}
\newcommand{\llaves}[1]{{#1}}

Goal: train a quantum circuit to become a pdf sampler (here, normal distribution).
## (?) JUSTIFICATION? WHY WOULD IT BE BETTER TO USE A QUANTUM CIRCUIT FOR THIS?
## 1. load probability distribution into quantum state ---> then manipulate it w/ quantum routine (where you can get some advantage). EXAMPLE? (HHL, QAE..)

### ALTERNATIVES ?? Variational Autoencoders... difussion models... which are pros and cons?  Comment! Quantum counterpart ?

## Classical: https://harvard-iacs.github.io/2019-CS109B/labs/lab11/GANS-sol/
Brief outline:

Let $X = {x^0, ...x^{s-1}} \subseteq \mathbb{R}^{\kout} \sim \preal$, with \preal unknown, our goal is to prepare a quantum state that encodes the pdf \preal (or some approximated version of it).

Our 'input' is - aside from the training set X, some noise z~ \pz, which can be seen as a 'prior' pdf, living in a $\kin$-dimensional space.

Let $\gen : \RR^{\kin}\rightarrow\RR^{\kout}$ be the \textit{generator} (an agent maping samples $z\sim\pz$ towards a pdf [which is hopefully close to the true one].

Let \disc:\RR^{k_{out}}\rightarrow [0,1] be the discriminator (*1), which takes samples obtained from a pdf and discerns if those were obtained from the real (values closer to 0) one or otherwise fake (values closer to 1). In other words, $\disc(\bm{y})$ retrieves the probability of $\bm{y}$ being sampled from $\preal$, according the the tester $\disc$, aimed to be optimized [1].(*2)

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
While the generator ussually consists on a NN, [2] proposes to replace it with a parametrized quantum circuit (PQC). As highlighted by the authors, the importance of having a PQC trained to sample from a target $\preal$ is that of \textit{loading} classical data into the quantum state (and afterwards perform quantum infor processing routines such as HHL or QAE) \footnote{Let us highlight that the complexity n-bits into a quantum state is O(2^n) in terms of #gates, whereas as claimed by Zoufal the QGAN requires O(Poly(n)), however the overall complexity might be higher due to BPs, so care must be taken when doing such claims, see for instance [4].}. An alternative not considered here is that of a fully-quantum GAN, i.e. quantizing the discriminator as well[5], which implies a collective measure on the generator's quantum state, not very NISQy. Other extensions of generative modelling to the quantum realm remain open to explore, such as [6,7].

The quantum counterpart of the generator is denoted by $\qgen$, and consists in a PQC with parameters $\bm{\theta}$, acting on an $n$ qubits. The input \textit{noise} $z$ will be now represented by an \textit{intial} quantum state $\ket{\psi_{in}}$ (which might potentially be of stochastic nature as well). The result is a quantum state $\ketgen$ encoding the probability distribution in the canonical basis:

\equ{\qgen \ket{\psi_{in}} = \ketgen = \sum_{k=0}^{2^n -1} \sqrt{p_k(\bm{theta})} \ket{k},}
with amplitudes $p_k(\bm{theta})$ encoding a probability distrubution (when measuring in the Z-basis, the probability of outcome $k$ is $p_k(\bm{theta})$). This setting allows for loading discrete distributions (or discretized) pdfs, with $2^n$ possible bins. Having samples $\llaves{x_j}_{j=1}^m$ obtained from $\preal$, and $\llaves{g_j}_{j=1}^m$ obtained from measuring $\qgen$ (note that there is a caveat here, since the amplitudes need to be estimated by many \textit{shots}, thus two notions of \textit{sampling} arise), the cost functions are (similarly to the classical case) given by
\begin{align}
L_G &= - \sum_k p_k(\bm{\theta}) \log(\disc(g_k)) \\
L_D &= - \big( \sum_k p_{prior}(x_k) \log \disc(x_k) + p_k(\bm{\theta}) \log(1- \disc(g_k)) \big)
\end{align}
Under this hybrid quantum-classical adversarial framework, we now move to present some numerics.

\section{Numerics}
We will consider 3-qubit systems and thus $8$ bins, and distribution using $2\times10^4$ samples.

We truncate the distribution in the interval [0,7].

The quantum circuit under consideration consists of a first layer of $R_y$ gate, and then $\ell$ alternating repetitions of $CZ$-gate plus a further layer of $R_y$ gates, as shown in Fig.~\ref{fig:ansatz}.

Ref [8], Sec. 4.1.4, presents different alternatives to prepare the initial quantum state $\ket{\psi_{in}}$ (which is the quantized version of input noise $z$), all of them using the circuit ansatz shown in Fig.~\ref{fig:ansatz}, and whose parameters are here denoted by $\bm{\alpha}$
\begin{enumerate}
\item \textit{Discrete uniform distribution}. In this case, we apply a Haddamard layer to $\ket{0}^\otimes n$, and set the parameters $\bm{\omega}$ close to zero. Note that this is equivalent to setting the first layer of rotation to identity (zero-angle rotation), and the last layer to $-\frac{\pi}{4}$\footnote{Depending on Qiskit implementation, the parameter would be $\frac{\pi}{2}$, a rotation about an angle; here we understand $R_\bm{n}(\bm{\theta}) = e^{-\frac{\ii \bm{n}\cdot \bm{\theta}}{2}}\theta$}.
\item \textit{Arbitrary distributions}, which can be obtained by minimizing the mean-squared loss between the amplitudes $p_k(\bm(\alpha)) = \langle \psi_in(\bm{\alpha}) | k\rangle$ and the corresponding target probability $q_k$ associated to such bin.
\end{enumerate}

\begin{figure}[t!]
  \includegraphics[width=.6\textwidth]{figures/ansatz_generator.png}
  \includegraphics[width=.6\textwidth]{figures/ansatz_initial.png}
  \caption{We show the circuits for the quantum generator $\qgen$ (above) and initial state $\ket{\psi_{in}}$ (below).}
    \label{fig:ansatz}
\end{figure}




%%%% REFERENCES %%%
(*1) There should be a [0,1] instead of a {0,1} in the definition of the paper in Zoufal paper [2] since the log(\disc) is otherwise undefined.
(*2) Has a rigurous hypothesis testing formulation been made with these GANs? Connection w/ Matera ?

[1] https://arxiv.org/abs/1406.2661
[2] https://www.nature.com/articles/s41534-019-0223-2
[4] https://arxiv.org/pdf/2305.02881.pdf
[5] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.040502
[6] https://arxiv.org/pdf/2302.00788.pdf
[7] https://quantum-journal.org/papers/q-2021-12-23-609/pdf/
 """

import qiskit
import qiskit_algorithms
import torch
import matplotlib.pyplot as plt
import numpy as np


from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.primitives import SamplerResult
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch.optim import Adam
from qiskit.circuit import Parameter
from tqdm import tqdm
import os

qiskit.__version__ # version 0.45.2

def seeds():
    qiskit_algorithms.random_seed = 123456
    _ = torch.manual_seed(123456)  # suppress output
    np.random.seed(qiskit_algorithms.random_seed)


############## INITIAL STATE GENERATOR ########
############## INITIAL STATE GENERATOR ########
############## INITIAL STATE GENERATOR ########
def normal(x,mu=0.0, sigma = 1):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(4*np.pi*sigma**2)

coords = np.linspace(-1, 1, 8)
probs = np.array([normal(x) for x in coords])


### Initial state of the generator (prior)
qc = QuantumCircuit(3)
ind=0
for q in qc.qubits:
    qc.ry(Parameter("{}".format(ind)), q)
    ind+=1
for q, qq in zip(qc.qubits[:-1], qc.qubits[1:]):
    qc.cz(q,qq)
for q in qc.qubits:
    qc.ry(Parameter("{}".format(ind)), q)
    ind+=1
qc.measure_all()
qc.draw("mpl")


seeds()
shots = 10000
sampler = Sampler(options={"shots": shots, "seed": qiskit_algorithms.random_seed})
qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=[], weight_params=qc.parameters,sparse=False) ##ths gives the amplitudes of the state in the computational basis
qprior = TorchConnector(qnn, initial_weights = np.random.random(qc.num_parameters))
prior_optimizer = Adam(qprior.parameters(), lr=1e-1)

#Note that the input is an empty tensor!
target_prob = torch.tensor(probs)/np.sum(probs)

def mse_probs(input,target):
    return torch.sum((input - target)**2)/input.shape[0]

cost0 = []
amps0 = []
for k in tqdm(range(200)):
    prior_optimizer.zero_grad()
    amplitudes_initial_state = qprior(torch.tensor([]))#.reshape(-1, 1)

    cost = mse_probs(amplitudes_initial_state,target_prob)
    cost.backward()
    prior_optimizer.step()

    cost0.append(cost.detach().numpy())
    amps0.append(amplitudes_initial_state.detach().numpy())

plt.figure(figsize=(10,3))
ax=plt.subplot(121)
ax.plot(cost0)
ax=plt.subplot(122)
ax.plot(amps0[-1])
ax.plot(target_prob)

trained_params_qprior = np.stack([k.detach().numpy() for k in list(qprior.parameters())[0]])
qprior_dir = "data/qprior/normal/"
os.makedirs(qprior_dir,exist_ok=True)
np.save(qprior_dir+"params",trained_params_qprior)

############## INITIAL STATE GENERATOR ########
############## INITIAL STATE GENERATOR ########
############## INITIAL STATE GENERATOR ########


##### QGAN implementation#####
##### QGAN implementation#####
##### QGAN implementation#####
##### QGAN implementation#####

def construct_qgen(L=2):
    """
    L: control number of HEA layers (related to circuit's depth)
    """
    qprior_dir = "data/qprior/normal/"

    trained_params_qprior = np.load(qprior_dir+"params.npy")

    ### \ket{\psi_in}
    qc_gen = QuantumCircuit(3)
    ind=0
    for q in qc_gen.qubits:
        qc_gen.ry(trained_params_qprior[ind], q)
        ind+=1
    for q, qq in zip(qc_gen.qubits[:-1], qc_gen.qubits[1:]):
        qc_gen.cz(q,qq)
    for q in qc_gen.qubits:
        qc_gen.ry(trained_params_qprior[ind], q)
        ind+=1

    ### GENERATOR
    ind=0
    for q in qc_gen.qubits:
        qc_gen.ry(Parameter("{}".format(ind)), q)
        ind+=1
    for p in range(L):
        for q, qq in zip(qc_gen.qubits[:-1], qc_gen.qubits[1:]):
            qc_gen.cz(q,qq)
        for q in qc_gen.qubits:
            qc_gen.ry(Parameter("{}".format(ind)), q)
            ind+=1
    qc_gen.measure_all()
    return qc_gen



### DISCRIMINATOR
class Discriminator(torch.nn.Module):
    def __init__(self, input_size=1):
        """Input_size :: k_out"""
        super(Discriminator, self).__init__()

        self.linear_input = torch.nn.Linear(input_size, 3)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.linear1 = torch.nn.Linear(3, 50)
        self.linear2 = torch.nn.Linear(50, 20)
        self.linear3 = torch.nn.Linear(20, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.leaky_relu(x)
        x = self.sigmoid(x)
        return x



def construct_nets(M= int(2e4)):
    ###Define discriminator
    discriminator = Discriminator()
    disc_optimizer = Adam(discriminator.parameters(), lr=1e-3)

    ### Define generator

    qc_gen= construct_qgen()
    bins = torch.linspace(-1,1,8)
    sampler = Sampler(options={"shots": M, "seed": qiskit_algorithms.random_seed})
    qnn_gen_sam = SamplerQNN(circuit=qc_gen, sampler=sampler, input_params=[], weight_params=qc_gen.parameters,sparse=False)
    qnn_gen = TorchConnector(qnn_gen_sam, initial_weights = np.random.random(qc_gen.num_parameters))
    gen_optimizer = Adam(qnn_gen.parameters(), lr=1e-2)

    return qnn_gen, gen_optimizer, discriminator, disc_optimizer


def cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=int(2e4)):
    uniform_prior = 1/8  #this is 1/|A| with A alphabet
    cost_discriminator_real = -torch.sum(torch.log(disc_on_real)*uniform_prior)
    cost_discriminator_fake = -torch.sum(torch.einsum('bt,bt->bt',torch.log(1.-disc_on_fake),probs_fake.detach() ))
    return cost_discriminator_fake + cost_discriminator_real


def cost_generator(disc_on_fake, probs_fake):
    return -torch.sum(torch.einsum('bt,bt->bt',torch.log(disc_on_fake.detach()),probs_fake ))

def call_qgen(qgen,M=int(2e4)):
    alphabet = np.linspace(-1,1,8)
    type_qgen = qgen(torch.tensor([]))
    samples_qgen = (torch.multinomial(type_qgen,M,replacement=True)/3.5 -1.)    #this moves the random-variable to the range [-1,1]
    probs_fakee = type_qgen[((samples_qgen+1)*3.5).int()].unsqueeze(-1)
    return probs_fakee, samples_qgen


M=int(2e4)

def normal(x,mu=0.0, sigma = 1):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(4*np.pi*sigma**2)

def give_samples_real(M=int(2e4)):
    """
    ### Check i discretized correctly
    counts,bins = np.histogram(scaled_samples_real,bins=8,density=True)
    delta_bin = (bins[1]-bins[0])
    counts*delta_bin/scaled_probs
    ####
    """
    coords = np.linspace(-1,1,8)
    probs = np.array([normal(x) for x in coords])
    scaled_probs = probs/np.sum(probs)
    samples_real = torch.multinomial(torch.tensor(scaled_probs), M, replacement=True).unsqueeze(-1)
    scaled_samples_real = (samples_real/3.5)-1.#this moves the random-variable to the range [-1,1]
    return scaled_samples_real,scaled_probs


seeds()
M=int(2e4)
qnn_gen, gen_optimizer, discriminator, disc_optimizer = construct_nets(M=M)
samples_real, probs_real = give_samples_real(M=M)
probs_fake, samples_qgen = call_qgen(qnn_gen)


## TO do... batch_size

### TRAINING LOOP ####
history = {}
costs = history["costs"] = {}
probs = history["probs"] = []
costs["disc"]= []
costs["gen"]= []
probs.append(qnn_gen(torch.tensor([])).detach().numpy())


plt.plot(probs[-1])
plt.plot(probs_real)


for k in tqdm(range(1000)):
    disc_optimizer.zero_grad()
    disc_on_real = discriminator(samples_real)
    disc_on_fake = discriminator(samples_qgen.unsqueeze(-1))
    cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=M)
    cost_discc.backward()
    disc_optimizer.step()
    costs["disc"].append(cost_discc)

plt.plot(torch.tensor(costs["disc"]))

torch.mean(disc_on_real)
torch.mean(disc_on_fake)



for epoch in tqdm(range(500)):
    samples_real, probs_real = give_samples_real(M=M)

    for k in range(10):
        disc_optimizer.zero_grad()
        disc_on_real = discriminator(samples_real)
        disc_on_fake = discriminator(samples_qgen.unsqueeze(-1))
        cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real)
        cost_discc.backward()
        disc_optimizer.step()
        costs["disc"].append(cost_discc)


    probs.append(qnn_gen(torch.tensor([])).detach().numpy())
    for k in range(1):
        gen_optimizer.zero_grad()
        type_qgen = qnn_gen(torch.tensor([]))
        probs_fake, samples_qgen = call_qgen(qnn_gen)
        disc_on_fake = discriminator(samples_qgen.unsqueeze(-1)).detach()

        cost_genn = cost_generator(disc_on_fake, probs_fake)
        cost_genn.backward()
        gen_optimizer.step()
        costs["gen"].append(cost_genn)


plt.plot(torch.tensor(costs["gen"]).detach().numpy())

plt.plot(type_qgen.detach().numpy())
