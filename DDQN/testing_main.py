from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


from random import randrange
import statistics
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['art_queue'],
        None
    )
    #None or "NS" or "EW"

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['n_cars_generated'],
        config['static_traffic_lights'] #STL or NOT
    )
    
    reward=0 #reward
    episode = 0 #episode number
    ql=[] #queue length vector for 5 episodes
    awt=[] #average waiting time per vehicle vector for 5 episodes
    total_testing_simulation_time=0

    #seed = [i + config['episode_seed'] for i in [1, 2, 3, 4, 5]] could have used it

    cuts = [0]


    #seed = [randrange(5, 100000)  for i in range(0,5)] #seeds for reproducibility
    seed = [45715, 92490, 80265, 3957, 40983]
    while episode < 5:
        print('\n----- Test episode n°', episode)
        simulation_time = Simulation.run(seed[episode])
        print('Simulation time:', simulation_time, 's')
        
        total_testing_simulation_time += simulation_time
        reward+=Simulation._sum_neg_reward        
        ql.append(Simulation._sum_queue_length)
        awt.append(Simulation._sum_queue_length/sum(Simulation._waits))
        episode += 1
    
        cuts.append(len(Simulation.reward_episode))


    print('\n----- Testing finished -----')
    print('Total testing simulation time:', total_testing_simulation_time, 's')
    avg_reward = reward/5
    twt = sum(ql)/5
    awt = statistics.median(awt)


    print("----- Testing info saved at:", plot_path)
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini')) #Save to recall the test settings
    
    # print("Saved into informations.txt")
    # f = open(os.path.join(plot_path, "informations"), "a")
    # f.write("\n----- Total simulation time : " + str(total_testing_simulation_time))
    # f.write("\n----- nrw : " + str(avg_reward))
    # f.write("\n----- twt : " + str(twt))
    # f.write("\n----- awt : " + str(awt))
    # f.write("\n----- seeds : " + str(seed))
    # f.write("\n\n")
    # f.close()

    #Print informations for average episodes
    print('seeds', seed)
    print('nrw', avg_reward)
    print('twt', twt)
    print('awt', awt)
    #print('Action step cuts', cuts)

    #The 5 cruve profiles are side by side
    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward',title="Reward during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', title="Queue length during testing", xlabel='Step', ylabel='Queue length (vehicles)')


    Visualization.save_data_and_plot(data=Simulation.reward_episode[:cuts[1]], filename='reward_1',title="Reward during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode[:config['max_steps']], filename='queue_1', title="Queue length during testing", xlabel='Step', ylabel='Queue length (vehicles)')
    #Visualization.save_data_x_y_and_plot(data_x=Simulation.xs, data_y=Simulation.reward_episode, filename='reward_steps',title="Reward during testing", xlabel='Step', ylabel='Reward')

    #Average DOES NOT WORK BECAUSE NOT SAME AMOUNT OF POINTS
    #Visualization.save_data_and_plot(data=[(Simulation.reward_episode[i]+Simulation.reward_episode[i+cuts[0]]+Simulation.reward_episode[i+cuts[1]]  + Simulation.reward_episode[i+cuts[2]] + Simulation.reward_episode[i+cuts[3]])/5 for i in range(cuts[0])], filename='reward_avg',title="Reward average during testing", xlabel='Action step', ylabel='Reward average')
    
    Visualization.save_data_and_plot(data=[(Simulation.queue_length_episode[i]+Simulation.queue_length_episode[i+config['max_steps']]+Simulation.queue_length_episode[i+(2*config['max_steps'])]  + Simulation.queue_length_episode[i+(3*config['max_steps'])] + Simulation.queue_length_episode[i+(4*config['max_steps'])])/5 for i in range(5400)], filename='queue_avg', title="Queue length average during testing", xlabel='Step', ylabel='Queue length average (vehicles)')


    #LINEAR INTERPOLATION
    # mean_x_axis = [i for i in range(max(Simulation.xs))]
    # ys_interp = [np.interp(mean_x_axis, Simulation.xs[cuts[i]:cuts[i+1]], Simulation.reward_episode[cuts[i]:cuts[i+1]]) for i in range(len(cuts)-1)]
    # mean_y_axis = np.mean(ys_interp, axis=0)
    # Visualization.save_data_x_y_and_plot(data_x=mean_x_axis, data_y=mean_y_axis, filename='reward_avg_linear_interpolation_steps',title="Reward average linear interpolation", xlabel='Step', ylabel='Reward average')

    # max_mean_x_axis = [abs(cuts[i+1] - cuts[i]) for i in range(len(cuts)-1)]
    # mean_x_axis = [i for i in range(max(max_mean_x_axis))]
    # ys_interp = [np.interp(mean_x_axis, Simulation.xs[cuts[i]:cuts[i+1]], Simulation.reward_episode[cuts[i]:cuts[i+1]]) for i in range(len(cuts)-1)]
    # mean_y_axis = np.mean(ys_interp, axis=0)
    # Visualization.save_data_x_y_and_plot(data_x=mean_x_axis, data_y=mean_y_axis, filename='reward_avg_linear_interpolation_action_steps',title="Reward average linear interpolation", xlabel='Action Step', ylabel='Reward average')
