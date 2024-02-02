
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
#probs = np.array([normal(x) for x in coords])
probs = np.array([1/len(coords) for x in coords])


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
qprior_dir = "data/qprior/uniform/"
os.makedirs(qprior_dir,exist_ok=True)
np.save(qprior_dir+"params",trained_params_qprior)
