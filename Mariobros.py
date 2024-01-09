#No deja de trabarse el visual maldita sea:(
import torch
import torch.nn as nn
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from PIL import Image
import numpy as np
import pandas as pd

class mnit_neural_network(nn.Module):
    def __init__(self, n_input, n_output, W) -> None:
        super(mnit_neural_network, self).__init__()
        self.weights = torch.tensor(W).reshape(1280, 10)
        self.input_layer = nn.Linear(n_input, 10)
        self.weights_inputlayer = self.weights[:, :200]
        self.weights_outputlayer = self.weights[:, 200:]
        self.output_layer = nn.Linear(10, n_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoide = nn.Sigmoid()
    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.sigmoide(self.output_layer(out)) 
        #out = 11 * out 
        return out 
def funcion_U(V, gen_actual, CR):
    U = gen_actual.copy()
    for i in range(len(V)):
        valor_random = np.random.uniform(0, 1)
        if valor_random <= CR:
            U[i] = V[i]
    return U

def getindex(index_actual, size):
    arreglo_index = []
    while len(arreglo_index) != 3:
        valor = np.random.randint(0, size-1)
        if valor != index_actual and valor not in arreglo_index:
            arreglo_index.append(valor)
    return arreglo_index

def evaluar_cerebro(W, n_input, n_output):
    cerebro = mnit_neural_network(n_input, n_output, W)
    recompensa = 0
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    for step in range(5000):
        obs_gris = convertir_a_gris(obs)
        obstensor = torch.from_numpy(obs_gris)  # Convertir a tensor
        obstensor = obstensor.float()
        accion = cerebro.forward(obstensor).argmax().item()
        print(accion)
        #accion = min(max(0, accion), 11)
        #print(accion)
        obs, reward, terminated, truncated, info = env.step(accion)
        recompensa += reward
        done = terminated or truncated

        if done:
            env.reset()

    return recompensa

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env,SIMPLE_MOVEMENT)
n_input = 50
n_output = len(SIMPLE_MOVEMENT)
CR = 0.8
F = 0.9
NP = 10
Num_of_gen = 10
print(COMPLEX_MOVEMENT)
# Función para convertir la observación a escala de grises
def convertir_a_gris(obs):
    img = Image.fromarray(obs, 'RGB').convert('L')
    img_redimensionada = img.resize((50, 50))
    return np.array(img_redimensionada)

def funcion_V_Mario(indexarr, df: pd.DataFrame, f):
    r0 = np.array(df.iloc[[indexarr[0]]])
    r1 = np.array(df.iloc[[indexarr[1]]])
    r2 = np.array(df.iloc[[indexarr[2]]])
    return r0 + f * (r1 - r2)

env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.reset()

tamaño_arreglo = ((n_input*200) + (400*n_output))
columna = np.arange(tamaño_arreglo)
columna_texto = str(columna)

Poblacion = pd.DataFrame(np.random.uniform(-2, 2, size=(Num_of_gen, len(columna))), columns=columna)
Best_gen = Poblacion.copy()
gen = 0

while gen < 2:
    recompensa = []
    for i in range(0, Num_of_gen):
        index = getindex(i, Num_of_gen)
        genactual = Best_gen.iloc[[i]]
        V = funcion_V_Mario(index, Best_gen, F)
        U = funcion_U(V[0], genactual, CR)

        # Convertir U a un tensor de PyTorch
        U_tensor = torch.tensor(U.values, dtype=torch.float32)

        reward_mario = evaluar_cerebro(U_tensor.numpy(), n_input, n_output)  # Convertir el tensor de vuelta a un array de NumPy para que sirva XD
        recompensa.append(reward_mario)
        Best_gen.iloc[[i]]
    gen += 1
    Best_gen['reward'] = recompensa
