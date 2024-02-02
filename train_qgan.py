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
    qprior_dir = "data/qprior/uniform/"

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
    def __init__(self, input_size=int(1e3)):
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




def cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=int(2e4)):
    uniform_prior = 1/8  #this is 1/|A| with A alphabet
    cost_discriminator_real = -torch.sum(torch.log(disc_on_real)*uniform_prior)
    cost_discriminator_fake = -torch.sum(torch.einsum('bt,bt->bt',torch.log(1.-disc_on_fake),probs_fake.detach() ))
    return cost_discriminator_fake + cost_discriminator_real


def cost_generator(disc_on_fake, probs_fake):
    return -torch.sum(torch.einsum('bt,bt->bt',torch.log(disc_on_fake.detach()),probs_fake ))

def call_qgen(qgen,shots=int(2e4)):
    alphabet = np.linspace(-1,1,8)
    type_qgen = qgen(torch.tensor([]))
    samples_qgen = (torch.multinomial(type_qgen,shots,replacement=True)/3.5 -1.)    #this moves the random-variable to the range [-1,1]
    probs_fakee = type_qgen[((samples_qgen+1)*3.5).int()].unsqueeze(-1)
    return probs_fakee, samples_qgen



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


def kl(p, q):
    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))

def process_batch(a):
    return a.squeeze().unsqueeze(0)


def give_empty_history(qnn_gen,probs_real):
    ### TRAINING LOOP ####
    history = {}
    costs = history["costs"] = {}
    probs = history["probs"] = []
    metrics = history["metrics"] = {"disc_on_real":[], "disc_on_fake":[], "KL":[],"DP":[]}
    costs["disc"]= []
    costs["gen"]= []
    probs.append(qnn_gen(torch.tensor([])).detach().numpy())
    first_kl = kl(probs_real.squeeze(),probs[-1])
    metrics["KL"].append(first_kl)
    metrics["DP"].append(np.abs(probs[-1] - probs_real.squeeze())/probs_real.squeeze())
    return history,probs,metrics,costs


def construct_nets(M= int(2e4), m=int(1e3) ,lr=1e-2):
    ###Define discriminator
    discriminator = Discriminator(input_size=m)
    disc_optimizer = Adam(discriminator.parameters(), lr=1e-3)

    ### Define generator

    qc_gen= construct_qgen()
    bins = torch.linspace(-1,1,8)
    sampler = Sampler(options={"shots": M, "seed": qiskit_algorithms.random_seed})
    qnn_gen_sam = SamplerQNN(circuit=qc_gen, sampler=sampler, input_params=[], weight_params=qc_gen.parameters,sparse=False)
    qnn_gen = TorchConnector(qnn_gen_sam, initial_weights = np.random.random(qc_gen.num_parameters))
    gen_optimizer = Adam(qnn_gen.parameters(), lr=1e-4)

    return qnn_gen, gen_optimizer, discriminator, disc_optimizer


def plot(metrics, what="disc_on"):
    if what=="disc_on":
        plt.figure()
        ax=plt.subplot(111)
        ax.plot(torch.tensor(metrics["disc_on_fake"]))
        ax.plot(torch.tensor(metrics["disc_on_real"]))
    elif what=="cost_gen":
        plt.figure()
        ax=plt.subplot(111)
        ax.plot(torch.tensor(costs["gen"]))

seeds()
M=int(1e3) ### samples in each epoch
m=M ###number of splits in the epoch

qnn_gen, gen_optimizer, discriminator, disc_optimizer = construct_nets(M=M, m=M)
_,probs_real = give_samples_real(M=1)
history,probs,metrics,costs = give_empty_history(qnn_gen, probs_real)
for iteration in range(100):
    print(metrics["DP"][-1])

    ### PRE-TRAIN DISC
    for k in tqdm(range(100)):
        samples_real, probs_real = give_samples_real(M=M)

        #### DISCRMINATOR STEP
        probs_fake, samples_qgen = call_qgen(qnn_gen,shots=len(samples_real))
        samples_qgen, samples_real = process_batch(samples_qgen), process_batch(samples_real)

        disc_optimizer.zero_grad()
        disc_on_real = discriminator(samples_real)
        disc_on_fake = discriminator(samples_qgen)
        cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=M)
        cost_discc.backward()
        disc_optimizer.step()

        metrics["disc_on_real"].append(torch.mean(disc_on_real))
        metrics["disc_on_fake"].append(torch.mean(disc_on_fake))
        costs["disc"].append(cost_discc)

    for j in tqdm(range(100)):
        gen_optimizer.zero_grad()
        type_qgen = qnn_gen(torch.tensor([]))
        probs_fake, samples_qgen = call_qgen(qnn_gen,shots=M)
        samples_qgen = process_batch(samples_qgen)

        disc_on_fake = discriminator(samples_qgen).detach()
        cost_genn = cost_generator(disc_on_fake, probs_fake)
        cost_genn.backward()
        gen_optimizer.step()
        costs["gen"].append(cost_genn)
        probs.append(qnn_gen(torch.tensor([])).detach().numpy())
        metrics["KL"].append(kl(probs_real.squeeze(),probs[-1]))
        metrics["DP"].append(np.abs(probs[-1] - probs_real.squeeze())/probs_real.squeeze())

        metrics["disc_on_real"].append(torch.mean(disc_on_real))
        metrics["disc_on_fake"].append(torch.mean(disc_on_fake))

plot(metrics,what="cost_gen")





type_qgen = qnn_gen(torch.tensor([]))
probs_fake, samples_qgen = call_qgen(qnn_gen,shots=M)
samples_qgen = process_batch(samples_qgen)

disc_on_fake = discriminator(samples_qgen).detach()

for k in tqdm(range(1000)):
    samples_real, probs_real = give_samples_real(M=M)

    #### DISCRMINATOR STEP
    probs_fake, samples_qgen = call_qgen(qnn_gen,shots=len(samples_real))
    samples_qgen, samples_real = process_batch(samples_qgen), process_batch(samples_real)

    disc_optimizer.zero_grad()
    disc_on_real = discriminator(samples_real)
    disc_on_fake = discriminator(samples_qgen)
    cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=M)
    cost_discc.backward()
    disc_optimizer.step()

    metrics["disc_on_real"].append(torch.mean(disc_on_real))
    metrics["disc_on_fake"].append(torch.mean(disc_on_fake))
    costs["disc"].append(cost_discc)


    #### GENERATOR STEP
    if k%1000==0:
        gen_optimizer.zero_grad()
        type_qgen = qnn_gen(torch.tensor([]))
        probs_fake, samples_qgen = call_qgen(qnn_gen,shots=M)
        samples_qgen = process_batch(samples_qgen)

        disc_on_fake = discriminator(samples_qgen).detach()
        cost_genn = cost_generator(disc_on_fake, probs_fake)
        cost_genn.backward()
        gen_optimizer.step()
        costs["gen"].append(cost_genn)
        probs.append(qnn_gen(torch.tensor([])).detach().numpy())
        metrics["KL"].append(kl(probs_real.squeeze(),probs[-1]))
        metrics["DP"].append(np.abs(probs[-1] - probs_real.squeeze())/probs_real.squeeze())

        print(metrics["DP"][-1])

plt.plot(probs[-1])
plt.plot(probs_real)
