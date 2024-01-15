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

#Estrctura de la red neuoronal
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

#Funciones
def FVMARIO(indexarr,pob,f):
    r0=(pob[indexarr[0]])
    r1=(pob[indexarr[1]])
    r2=(pob[indexarr[2]])
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

def evaluar_cerebro(W, n_input, n_output):#Esta es la funcion donde jugara Mario
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
            env.reset()

    return recompensa
def mejor(mayor,arreglo,mejorcerebro,pesosmejores,pesos):
    bestindex=mejorcerebro
    pesosideales=pesosmejores
    for i in range(len(arreglo)):
        if arreglo[i]>mayor:
            bestindex=i
            mayor=arreglo[i]
            pesosideales=pesos[bestindex]

    return bestindex,mayor,pesosideales

def mejorescerebros(recompensaactual,recompensa2,bestgen,cerebros2):
    for i in range(0,10):
        if recompensaactual[i]<recompensa2[i]:
            bestgen[i]=cerebros2[i]
            recompensaactual[i]=recompensa2[i]
    return bestgen,recompensaactual

#Creacion de la red neuronal
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

# Aqui creamos la poblacion
Poblacion = []
for i in range(0, 10):
    arreglo = np.random.uniform(-2, 2, tamaño_arreglo)
    Poblacion.append(arreglo)
#Inica entrenamiento de la red neuoronal
mayorrecompensa=-1000
Best_gen = Poblacion.copy()
mejorcerebro=0
pesosideales=Best_gen[0]
mejoresrecompensas=[0,0,0,0,0,0,0,0,0,0]
gen=0
while gen<=10:
    Best_gen=Best_gen.copy()
    recompensa=[]
    pesos_actuales=[]
    for i in range(0,Num_of_gen):
        index=getindex(i,Num_of_gen)
        v = (FVMARIO(index, Best_gen, F))
        u = (funcion_U(v, Best_gen[i], CR))
        r=evaluar_cerebro(u, n_input, n_output)
        recompensa.append(r)
        pesos_actuales.append(u)
    print("Fin de generacion {}".format(gen))
    print("Todas las recompensas = {}".format(recompensa))
    Best_gen,mejoresrecompensas=mejorescerebros(mejoresrecompensas,recompensa,Best_gen,pesos_actuales)
    print("Estas son las mejores recompensas= {}".format(mejoresrecompensas))
    mejorcerebro,mayorrecompensa,pesosideales=mejor(mayorrecompensa,recompensa,mejorcerebro,pesosideales,Best_gen)
    print("Mayor recompensa = {}".format(mayorrecompensa))
    print(pesosideales)
    gen+=1

#Proceso de serializacion de los datos
with open('pesosideales.txt', 'w') as file:
    for peso in pesosideales:
        file.write(str(peso) + '\n')

with open('mayorrecompensa.txt', 'w') as file:
    file.write(str(mayorrecompensa) + '\n')