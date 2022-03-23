from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path

import statistics


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])
    
    Model_A1 = TestModel(
        input_dim=config['num_states'],
        model_path=model_path,
        num=1
    )
    
    Model_A2 = TestModel(
        input_dim=config['num_states'],
        model_path=model_path,
        num=2
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['art_queue'],
        None
    )
    #None for Low or High and "EW" or "NS"

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model_A1,
        Model_A2,
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

    cuts = [0]
    
    seed = [45715, 92490, 80265, 3957, 40983]
    while episode < 5:
        print('\n----- Test episode n°', episode)
        simulation_time = Simulation.run(seed[episode])  # run the simulation
        print('Simulation time:', simulation_time, 's')
        
        total_testing_simulation_time += simulation_time
        reward+=Simulation._sum_neg_reward_one + Simulation._sum_neg_reward_two       
        ql.append(Simulation._sum_queue_length)
        awt.append(Simulation._sum_queue_length/sum(Simulation._waits))
        episode += 1
        
        cuts.append(len(Simulation.reward_episode))

        
    # print('\n----- Test episode')
    # simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    # print('Simulation time:', simulation_time, 's')

    print('\n----- Testing finished -----')
    print('Total testing simulation time:', total_testing_simulation_time, 's')
    avg_reward = reward/5
    twt = sum(ql)/5
    awt = statistics.median(awt)

    print("----- Testing info saved at:", plot_path)
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    print("Saved into informations.txt")
    f = open(os.path.join(plot_path, "informations"), "a")
    f.write("\n----- Total simulation time : " + str(total_testing_simulation_time))
    f.write("\n----- nrw : " + str(avg_reward))
    f.write("\n----- twt : " + str(twt))
    f.write("\n----- awt : " + str(awt))
    f.write("\n----- seeds : " + str(seed))
    f.write("\n\n")
    f.close()

    #Print informations for average episodes
    print('seeds', seed)
    print('nrw', avg_reward)
    print('twt', twt)
    print('awt', awt)

    #Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    #Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')

    #Global + Locals
    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward',title="Reward during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.reward_episode_1, filename='reward_TL',title="Reward of TL during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.reward_episode_2, filename='reward_TL2',title="Reward of 2_TL during testing", xlabel='Action step', ylabel='Reward')

    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', title="Queue length during testing", xlabel='Step', ylabel='Queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode_1, filename='queue_TL', title="Queue length around TL during testing", xlabel='Step', ylabel='Queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode_2, filename='queue_TL2', title="Queue length around 2_TL during testing", xlabel='Step', ylabel='Queue length (vehicles)')

    Visualization.save_data_and_plot(data=Simulation.reward_episode[:cuts[1]], filename='reward_1',title="Reward during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.reward_episode_1[:cuts[1]], filename='reward_1_TL',title="Reward of TL during testing", xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.reward_episode_2[:cuts[1]], filename='reward_1_TL2',title="Reward of 2_TL during testing", xlabel='Action step', ylabel='Reward')

    Visualization.save_data_and_plot(data=Simulation.queue_length_episode[:config['max_steps']], filename='queue_1', title="Queue length during testing", xlabel='Step', ylabel='Queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode_1[:config['max_steps']], filename='queue_1_TL', title="Queue length around TL during testing", xlabel='Step', ylabel='Queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode_2[:config['max_steps']], filename='queue_1_TL2', title="Queue length around 2_TL during testing", xlabel='Step', ylabel='Queue length (vehicles)')

    Visualization.save_data_and_plot(data=[(Simulation.queue_length_episode[i]+Simulation.queue_length_episode[i+config['max_steps']]+Simulation.queue_length_episode[i+(2*config['max_steps'])]  + Simulation.queue_length_episode[i+(3*config['max_steps'])] + Simulation.queue_length_episode[i+(4*config['max_steps'])])/5 for i in range(5400)], filename='queue_avg', title="Queue length average during testing", xlabel='Step', ylabel='Queue length average (vehicles)')
    Visualization.save_data_and_plot(data=[(Simulation.queue_length_episode_1[i]+Simulation.queue_length_episode_1[i+config['max_steps']]+Simulation.queue_length_episode_1[i+(2*config['max_steps'])]  + Simulation.queue_length_episode_1[i+(3*config['max_steps'])] + Simulation.queue_length_episode_1[i+(4*config['max_steps'])])/5 for i in range(5400)], filename='queue_avg_TL', title="Queue length average around TL during testing", xlabel='Step', ylabel='Queue length average (vehicles)')
    Visualization.save_data_and_plot(data=[(Simulation.queue_length_episode_2[i]+Simulation.queue_length_episode_2[i+config['max_steps']]+Simulation.queue_length_episode_2[i+(2*config['max_steps'])]  + Simulation.queue_length_episode_2[i+(3*config['max_steps'])] + Simulation.queue_length_episode_2[i+(4*config['max_steps'])])/5 for i in range(5400)], filename='queue_avg_TL2', title="Queue length average around 2_TL during testing", xlabel='Step', ylabel='Queue length average (vehicles)')

 

    
