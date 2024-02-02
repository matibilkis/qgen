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


### TRAINING LOOP ####
history = {}
costs = history["costs"] = {}
probs = history["probs"] = []
costs["disc"]= []
costs["gen"]= []
probs.append(qnn_gen(torch.tensor([])).detach().numpy())



for k in tqdm(range(1000)):
    disc_optimizer.zero_grad()
    disc_on_real = discriminator(samples_real)
    disc_on_fake = discriminator(samples_qgen.unsqueeze(-1))
    cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=M)
    cost_discc.backward()
    disc_optimizer.step()
    costs["disc"].append(cost_discc)


for epoch in tqdm(range(500)):

    print("disc_on_real: ",torch.mean(disc_on_real))
    print("disc_on_fake:",torch.mean(disc_on_fake))
    print("\nDelta p/p_R: ", np.abs((probs[-1] - probs_real.squeeze())/probs_real))
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
