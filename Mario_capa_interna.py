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

class MariobrosNN:
    def __init__(self, entrada, salida,W):
        self.capaentrada = entrada
        self.capasalida = salida
        self.pesos=W
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoide=nn.Sigmoid()

    def forward(self, input):
        # Entrada
        Pesos_entrada=self.pesos[:15625]
        bias1 = np.zeros((1, 25))
        output_capaentrada = np.dot(input, Pesos_entrada.reshape(625,25)) + bias1
        output_capaentrada_act = self.relu(torch.tensor(output_capaentrada))
        #Capa_oculta_1
        Pesos_capa_oculta=self.pesos[15625:15713]

        # Salida
        Pesos_salida=self.pesos[15713:]
        bias2 = np.zeros((1, 7))
        output_capasalida = np.dot(output_capaentrada_act, Pesos_salida.reshape(25,7)) + bias2
        output_capasalida_act = self.softmax(torch.tensor(output_capasalida))
        return output_capasalida_act

def FVMARIO(indexarr,pob,f):
    r0=np.array(pob[indexarr[0]])
    r1=np.array(pob[indexarr[1]])
    r2=np.array(pob[indexarr[2]])
    return r0 + f * (r1 - r2)
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

def convertir_a_gris(obs):
    img = Image.fromarray(obs, 'RGB').convert('L')
    img_redimensionada = img.resize((25, 25))
    return np.array(img_redimensionada)

def evaluar_cerebro(W, n_input, n_output):
    cerebro = MariobrosNN(n_input, n_output,W)
    recompensa = 0
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for step in range(5000):
        obs_gris = convertir_a_gris(obs)
        obs_gristensor = torch.tensor(np.array(obs_gris).flatten().tolist())
        accion = cerebro.forward(obs_gristensor).argmax().item()
        print(accion)
        obs, reward, terminated, truncated, info = env.step(accion)
        recompensa += reward
        done = terminated or truncated

        if done:
            env.reset()

    return recompensa

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env,SIMPLE_MOVEMENT)
n_input = 25 * 25
n_output = 7
CR = 0.8
F = 0.9
NP = 10
Num_of_gen = 10
print(SIMPLE_MOVEMENT)
tamaño_arreglo = 15800

# Aqui creamos la poblacion
Poblacion = []
for i in range(0, 10):
    arreglo = np.random.uniform(-2, 2, tamaño_arreglo)
    Poblacion.append(arreglo)

gen = 0

index = [2, 7, 5]
v = (FVMARIO(index, Poblacion, F))
print(v)
u = (funcion_U(v, Poblacion[1], CR))
mayorrecompensa=-1000
def mejor(mayor,arreglo):
    for i in range(len(arreglo)):
        if arreglo[i]>mayor:
            bestindex=i
            mayor=arreglo[i]
    return bestindex,mayor
            
# Example usage
Best_gen = Poblacion.copy()
evaluar_cerebro(u, n_input, n_output)
while gen<3:
    Best_gen=Best_gen.copy()
    recompensa=[]
    for i in range(0,Num_of_gen):
        index=getindex(i,Num_of_gen)
        v = (FVMARIO(index, Best_gen, F))
        u = (funcion_U(v, Best_gen[i], CR))
        r=evaluar_cerebro(u, n_input, n_output)
        recompensa.append(r)
        print("Recompensa = {}".format(r))
        print("Cerebro {} terminado".format(i))
        Best_gen[i]=u
    print("Fin de generacion {}".format(gen))
    mejorcerebro,mayorrecompensa=mejor(mayorrecompensa,recompensa)
    pesosideales=Best_gen[mejorcerebro]
    gen+=1

'''

while gen < 10:
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
'''