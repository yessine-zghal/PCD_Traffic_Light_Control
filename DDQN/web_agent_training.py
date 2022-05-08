from flask import Flask, request, jsonify
import numpy as np

from memory import Memory
from model import TrainModel
from tensorflow.keras.utils import plot_model
from waitress import serve

#Remove verbose except errors
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

#Random agent hyperparameters
num_layers = 4
width_layers = 480
batch_size = 100
learning_rate = 0.001
training_epochs = 800
num_states = 320
num_actions = 4

memory_size_min = 600
memory_size_max = 50000

model = TrainModel(
    num_layers, 
    width_layers, 
    batch_size, 
    learning_rate, 
    input_dim=num_states, 
    output_dim=num_actions
)
mem = Memory(
   memory_size_max, 
   memory_size_min
)


@app.route('/initialize_agent', methods=['POST'])
def initialize_agent():
    model._num_layers =  request.get_json()['num_layers']
    model._width = request.get_json()['width_layers']
    model._batch_size = request.get_json()['batch_size']
    model._learning_rate = request.get_json()['learning_rate']
    model._input_dim = request.get_json()['num_states']
    model._output_dim = request.get_json()['num_actions']
    
    mem._size_max = request.get_json()['memory_size_max']
    mem._size_min = request.get_json()['memory_size_min']
    
    return "ok"

@app.route('/add_sample', methods=['POST'])
def add_sample():
    old_state = np.array(request.get_json()['old_state'])
    old_action = request.get_json()['old_action']
    reward = request.get_json()['reward']
    current_state = np.array(request.get_json()['current_state'])
    mem.add_sample((old_state, old_action, reward, current_state))
    return "ok"

@app.route('/predict', methods=['POST'])
def predict():
    state = np.array(request.get_json()['state'])
    prediction = model.predict_one(state)
    return jsonify(prediction=prediction.tolist())


@app.route('/replay', methods=['POST'])
def replay():
    
    num_states = request.get_json()['num_states']

    num_actions = request.get_json()['num_actions']
    gamma = request.get_json()['gamma']
    
    batch = mem.get_samples(model.batch_size)

    if len(batch) > 0:  # if the memory is full enough
        states = np.array([val[0] for val in batch])  # extract states from the batch
        print("states",states)
        print("states_len",states.shape)
        next_states = np.array([val[3] for val in batch])  # extract next states from the batch

        # prediction
        q_s_a = model.predict_batch(states)  # predict Q(state), for every sample
        print("q_s_a",q_s_a)
        print("lende qsa", len(q_s_a))
        q_s_a_d = model.predict_batch(next_states)  # predict Q(next_state), for every sample
        print("q_s_a_d",q_s_a_d)
        print("lennnnn ",len(q_s_a_d))
        # setup training arrays
        x = np.zeros((len(batch), num_states))
        y = np.zeros((len(batch), num_actions))

        for i, b in enumerate(batch):
            print("la valeur de i ",i)
            state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample

            print("le state est ",state)
            print("le len de state est ",len(state))
            print("le action est ", action)

            current_q = q_s_a[i]  # get the Q(state) predicted before

            print("le currrent_q est ", current_q)
            print("le  est ", len(current_q))
            current_q[action] = reward + gamma * np.amax(q_s_a_d[i])  # update Q(state, action)

            print("le currrent_q_before est ", current_q)
            print("le  est before  ", len(current_q))
            x[i] = state
            y[i] = current_q  # Q(state) that includes the updated action value
        model.train_batch(x, y)  # train the NN

    return jsonify(loss=model._training_loss)

@app.route('/save_model', methods=['POST'])
def save_model():
    path = request.get_json()['path']
    model.save_model(path)
    #plot_model(model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)
    return "ok"

if __name__ == '__main__':
    # Start Web App
    #Locally : app.run(threaded=False)
    #Remote :
    serve(app, host='0.0.0.0', port=5000)




     