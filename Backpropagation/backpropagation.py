# import all the necessary libraries
from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import math

# data-preprocesing on UCI-iris dataset
def load_d(d_path, n_train):
    
    d = []
    l = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(d_path, 'r') as file:
       
        csv_file_reader = reader(file, delimiter=',')
        for r in csv_file_reader:
            
            r[0:4] = list(map(float, r[0:4]))
            
            r[4] = l[r[4]]
            
            d.append(r)

    
    d = np.array(d)
    mms = MinMaxScaler()
    for i in range(d.shape[1] - 1):
        d[:, i] = mms.fit_transform(d[:, i].reshape(-1, 1)).flatten()

    
    d = d.tolist()
    for r in d:
        r[4] = int(r[4])
    
    random.shuffle(d)

   
    t_d = d[0:n_train]
    val_d = d[n_train:]

    return t_d, val_d

# calculate activation 
def fun(w, inputs):
    
    b = w[-1]
    z = 0
    for i in range(len(w)-1):
        z += w[i] * inputs[i]
    z += b
    return z

# transfer function
def sig(z):
   
    return 1.0 / (1.0 + math.exp(-z))

# derivative for the transfer function
def sig_der(o):
    return o * (1.0 - o)

# neccesary calculation for the forward propagation
def forward_prop(Network, inputs):
   
    for layer in Network: 
        new_inputs = []
        for n in layer: 
            z = fun(n['w'], inputs)
            n['o'] = sig(z)
            new_inputs.append(n['o'])
        inputs = new_inputs
    return inputs

# Caalculating the backward propagate error
def back_prop_error(Network, a_l):
   
    for i in reversed(range(len(Network))):  
        layer = Network[i]
        es = list()
        if i != len(Network)-1:  
            for j in range(len(layer)): 
                e = 0.0
                for n in Network[i + 1]:
                    e += (n['w'][j] * n['delta'])
                es.append(e)
        else: 
            for j in range(len(layer)): 
                n = layer[j]
                es.append(a_l[j] - n['o'])
        
        for j in range(len(layer)):
            n = layer[j]
            n['delta'] = es[j] * sig_der(n['o'])

# update parameters for backpragation 
def update_para(Network, r, l_r):
    
    for i in range(len(Network)):
        inputs = r[:-1]
        if i != 0:  
            inputs = [n['o'] for n in Network[i - 1]]
        for n in Network[i]:
            
            for j in range(len(inputs)):
                n['w'][j] += l_r * n['delta'] * inputs[j]
           
            n['w'][-1] += l_r * n['delta']

# iniatialization of the Network
def initialize_Network(n_inputs, n_hidden, n_outputs):
   
    Network = list()
    
    hidden_l = [{'w': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    Network.append(hidden_l)
   
    output_l = [{'w': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    Network.append(output_l)
    return Network

# training of the Network
def train(t_d, l_r, epochs, n_hidden, val_d):
    
  
    n_inputs = len(t_d[0]) - 1
    
    n_outputs = len(set([r[-1] for r in t_d]))
   
    Network = initialize_Network(n_inputs, n_hidden, n_outputs)

    acc = []
    for epoch in range(epochs): 
        for r in t_d:
         
            _ = forward_prop(Network, r)
         
            a_l = [0 for i in range(n_outputs)]
            a_l[r[-1]] = 1
           
            back_prop_error(Network, a_l)
           
            update_para(Network, r, l_r)
        
        acc.append(val(Network, val_d))
    print("Testing acc:", max(acc))

    return Network

# check validation
def val(Network, val_d):
    
    p_l = []
    for r in val_d:
        prediction = predict(Network, r)
        p_l.append(prediction)
    
    a_l = [r[-1] for r in val_d]
  
    acc = acc_cal(a_l, p_l)
   
    return acc

# calculate accuaracy
def acc_cal(a_l, p_l):
    
    true_count = 0
    for i in range(len(a_l)):
        if a_l[i] == p_l[i]:
            true_count += 1
    return true_count / float(len(a_l)) * 100.0

# predict new data
def predict(Network, r):
  
    outputs = forward_prop(Network, r)
    return outputs.index(max(outputs))

# give values to variable and calling the functions
if __name__ == "__main__":
    # load data from csv file
    file_path = '/content/iris.csv'

    # set all parameters for training
    l_r = 0.1 
    epochs = 80
    n_hidden = 3  
    n_train = 105 # 70% of the total 150 iris data
    
    t_d, val_d = load_d(file_path, n_train)
    # Calling function
    Network = train(t_d, l_r, epochs, n_hidden, val_d)
    