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
    def __init__(self, entrada, salida,W:np.array):
        self.capaentrada = entrada
        self.capasalida = salida
        self.pesos=W
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoide=nn.Sigmoid()

    def forward(self, input):
        # Entrada
        Pesos_entrada=self.pesos[:250000]
        bias1 = np.zeros((1, 100))
        output_capaentrada = np.dot(input, Pesos_entrada.reshape(2500, 100)) + bias1
        output_capaentrada_act = self.relu(torch.tensor(output_capaentrada))
        #Capa_oculta
        Pesos_capaoculta_1=self.pesos[250000:258000]
        bias2=np.zeros((1,80))
        output_capaoculta=np.dot(output_capaentrada_act,Pesos_capaoculta_1.reshape(100,80))+bias2
        activacion_oculta=self.softmax(torch.tensor(output_capaoculta))
        # Salida
        Pesos_salida=self.pesos[258000:]
        bias3 = np.zeros((1, 12))
        output_capasalida = np.dot(activacion_oculta, Pesos_salida.reshape(80,12)) + bias3
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
    img_redimensionada = img.resize((50, 50))
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
        #print(accion)
        obs, reward, terminated, truncated, info = env.step(accion)
        recompensa += reward
        done = terminated or truncated

        if done:
            recompensa=0
            env.reset()

    return recompensa
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
n_input = 50 * 50
n_output = 12
CR = 0.8
F = 0.9
NP = 10
Num_of_gen = 10
print(COMPLEX_MOVEMENT)
tamaño_arreglo = 258960#El numero de pesos que tendra la red neuronal

with open('pesosideales_Mario_Pasa_los_tubos.txt', 'r') as file:
    pesos_leidos = [float(line.strip()) for line in file]

pesos_leidos=(np.array(pesos_leidos))
loaded_Best_gen = np.loadtxt('Best_gen.txt')
print(evaluar_cerebro(pesos_leidos,n_input,n_output))