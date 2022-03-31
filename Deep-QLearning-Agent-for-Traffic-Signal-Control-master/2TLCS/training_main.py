from __future__ import absolute_import
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import datetime
from shutil import copyfile
#import tensorflow as tf
from model import TrainModel
from train import TrainModel2
from mem import mem
from Buffer import Buffer
from training_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
import tensorflow as tf
#import multiprocessing as mp
# import requests
import timeit
#from multiprocessing import Queue


def avg_occupancy_and_flow(list_occupancy, list_flow):
    '''
    Method to calculate the average occupancy and the average flow of the traffic
    '''
    avg_occ = [sum(i) / len(list_occupancy) for i in zip(*list_occupancy)]
    o_max = max(avg_occ)  # maximum occupancy
    max_index = avg_occ.index(o_max)
    avg_occ = avg_occ[:max_index + 1]
    avg_flow = [sum(i) / len(list_flow) for i in zip(*list_flow)][:max_index + 1]
    return avg_occ, avg_flow


def avg_density_and_flow(list_density, list_flow):
    '''
    Method to calculate the average density and the average flow of the traffic
    '''
    avg_den = [sum(i) / len(list_density) for i in zip(*list_density)]
    d_max = max(avg_den)  # maximum density
    max_index = avg_den.index(d_max)
    avg_density = avg_den[:max_index + 1]
    avg_flow = [sum(i) / len(list_flow) for i in zip(*list_flow)][:max_index + 1]
    return avg_density, avg_flow


def gpu_available():
    '''
    Tells you if tensorflow detects your GPU device
    '''
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


"""def launch_process(simulation, episode, epsilon, mode, return_dict):
     '''
     Launches the simulation process and returns the simulation data when it is finished
     '''
     simulation.run(episode, epsilon)
     return_dict[mode] = simulation.stop()
"""

"""def save_to_visualize(measurements):
     '''
     Method to save and visualize data (in order to avoid repetitions)
     '''
     for m in measurements:
         data, filename, title, xlabel, ylabel = m[0], m[1], m[2], m[3], m[4]
         scenarios=['High', 'Low']
         Visualization.save_data_and_plot_multiple_curves(list_of_data=[[data[i] for i in range(len(data)) if i%4==0], [data[i] for i in range(len(data)) if i%4==1], [data[i] for i in range(len(data)) if i%4==2], [data[i] for i in range(len(data)) if i%4==3]], filename=filename, title=title, xlabel=xlabel, ylabel=ylabel, scenarios=scenarios)
"""

if __name__ == "__main__":

    # is your GPU available ?
    # gpu_available()
    # the number of processors in your computer
    # print("Number of processors: ", mp.cpu_count())

    # import configuration information
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # High traffic generator
    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated_high'],
        # config['art_queue']
    )
    # Low traffic generator
    """TrafficGen_2 = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated_high'],
        #config['art_queue']
    )
    """
    Model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions'],
    )

    Model2 = TrainModel2(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions'],
    )

    mem = Buffer(
        config['memory_size_max'],
        config['memory_size_min']
    )

    Buffer= Buffer(
        config['memory_size_max'],
        config['memory_size_min']
    )


    # Low traffic generator
    # TrafficGen_2 = TrafficGenerator(
    #     config['max_steps'], 
    #     config['n_cars_generated_low'],
    #     config['art_queue']
    # )

    # EW traffic generator
    # TrafficGen_3 = TrafficGenerator(
    #     config['max_steps'], 
    #     config['n_cars_generated_ew'],
    #     config['art_queue'],
    #     'EW'
    # )

    # NS traffic generator
    # TrafficGen_4 = TrafficGenerator(
    #     config['max_steps'], 
    #     config['n_cars_generated_ns'],
    #     config['art_queue'],
    #     'NS'
    # )

    # Same visualization
    Visualization = Visualization(
        path,
        dpi=96
    )

    # Simulation with High traffic
    Sim = Simulation(
        Model,
        mem,
        Model2,
        Buffer,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )

    # Simulation with Low traffic
    """"Simulation_2 = Simulation(
        Model1,
        Memory2,
        Model2,
        Memory2,
        TrafficGen_2,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )"""

    # Simulation with EW traffic
    # Simulation_3 = Simulation(
    #     TrafficGen_3,<
    #     sumo_cmd,
    #     config['gamma'],
    #     config['max_steps'],
    #     config['green_duration'],
    #     config['yellow_duration'],
    #     config['num_cells'],
    #     config['num_states'],
    #     config['num_actions'],
    #     config['training_epochs']
    # )

    # Simulation with NS traffic
    # Simulation_4 = Simulation(
    #     TrafficGen_4,
    #     sumo_cmd,
    #     config['gamma'],
    #     config['max_steps'],
    #     config['green_duration'],
    #     config['yellow_duration'],
    #     config['num_cells'],
    #     config['num_states'],
    #     config['num_actions'],
    #     config['training_epochs']
    # )

    # inititalization of agent
    # print("Initialization of the agent")
    # requests.post('http://127.0.0.1:5000/initialize_agents', json={'num_layers': config['num_layers'], 
    #     'width_layers': config['width_layers'], 
    #     'batch_size': config['batch_size'], 
    #     'learning_rate': config['learning_rate'], 
    #     'num_states': config['num_states'], 
    #     'num_actions': config['num_actions'],
    #     'memory_size_max': config['memory_size_max'], 
    #     'memory_size_min': config['memory_size_min']})

    # Statistics
    REWARD_STORE = []  # Global negative reward
    REWARD_STORE_A1 = []  # Local negative reward first agent
    REWARD_STORE_A2 = []  # Local negative reward second agent
    CUMULATIVE_WAIT_STORE = []  # Global cumulative wait store
    CUMULATIVE_WAIT_STORE_A1 = []  # Local cumulative wait store first agent
    CUMULATIVE_WAIT_STORE_A2 = []  # Local cumulative wait store second agent
    AVG_QUEUE_LENGTH_STORE = []  # Global average queue length store
    AVG_QUEUE_LENGTH_STORE_A1 = []  # Local average queue length store first agent
    AVG_QUEUE_LENGTH_STORE_A2 = []  # Local average queue length store second agent
    AVG_WAIT_TIME_PER_VEHICLE = []  # Global average time per vehicle
    AVG_WAIT_TIME_PER_VEHICLE_A1 = []  # Local average time per vehicle first agent
    AVG_WAIT_TIME_PER_VEHICLE_A2 = []  # Local average time per vehicle second agent
    MIN_LOSS_A1 = []  # Minimum loss of the first agent per episode
    AVG_LOSS_A1 = []  # Avergae loss of the first agent per episode
    MIN_LOSS_A2 = []  # Minimum loss of the second agent per episode
    AVG_LOSS_A2 = []  # Average loss of the second agent per episode
    DENSITY = []  # Density information
    FLOW = []  # Flow information
    OCCUPANCY = []  # Occupancy information

    # Main loop
    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        start_sim_time = timeit.default_timer()



        # manager = mp.Manager()
        # return_dict = manager.dict()

        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        return_dict = []
        epsilon = 1.0 - (episode /config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        print("la valauer de ep", epsilon)

        simulation_time, training_time = Sim.run(episode, epsilon)  # run the simulation

        # Processes generation
        print("Launch processes")
        # pool = mp.Pool(processes=mp.cpu_count())
        # sims=[Sim]
        # mode=['HIGH']
        """for i in range(len(sims)):
             print('test')
             pool.apply_async(launch_process, (Sim, episode, epsilon, ["HIGH"], return_dict),).get()
             print('end test')
        print('yup')
        pool.close()
        pool.join()"""
        # simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
        round(simulation_time + training_time, 1), 's')
        episode += 1
        # total = round(timeit.default_timer() - start_sim_time, 1)
        # print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')

        # Replay

        print("Training...")
        start_time = timeit.default_timer()
        print("First agent replays")
        model_loss_agent_one=[]
        num_agent = 1
        for _ in range(config['training_epochs']):
            Simulation._replay()
            tr_loss = Model.training_loss
            model_loss_agent_one.append(tr_loss)
            
        print("Second agent replays")
        model_loss_agent_two=[]
        num_agent = 2
        for _ in range(config['training_epochs']):
            Simulation._replay2()
            tr_loss = Model2.training_loss

            model_loss_agent_two.append(tr_loss)
            
        training_time = round(timeit.default_timer() - start_time, 1)
        print('Training time: ', training_time)
        
        print('\nTotal time for this simulation: ', simulation_time+training_time)
        
        print("Saving loss results...")
        if(len(model_loss_agent_one) > 0):
             AVG_LOSS_A1.append(sum(model_loss_agent_one)/config['training_epochs'])
             MIN_LOSS_A1.append(min(model_loss_agent_one))
             
        if(len(model_loss_agent_two) > 0):
             AVG_LOSS_A2.append(sum(model_loss_agent_two)/config['training_epochs'])
             MIN_LOSS_A2.append(min(model_loss_agent_two))

        return_dict = Sim.stop()

        REWARD_STORE.append(return_dict[0])
        REWARD_STORE_A1.append(return_dict[1])
        REWARD_STORE_A2.append(return_dict[2])
        CUMULATIVE_WAIT_STORE.append(return_dict[3])
        CUMULATIVE_WAIT_STORE_A1.append(return_dict[4])
        CUMULATIVE_WAIT_STORE_A2.append(return_dict[5])
        AVG_QUEUE_LENGTH_STORE.append(return_dict[6])
        AVG_QUEUE_LENGTH_STORE_A1.append(return_dict[7])
        AVG_QUEUE_LENGTH_STORE_A2.append(return_dict[8])
        AVG_WAIT_TIME_PER_VEHICLE.append(return_dict[9])
        AVG_WAIT_TIME_PER_VEHICLE_A1.append(return_dict[10])
        AVG_WAIT_TIME_PER_VEHICLE_A2.append(return_dict[11])
        DENSITY.append(return_dict[12])
        FLOW.append(return_dict[13])
        OCCUPANCY.append(return_dict[14])

        # episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    print("Saved into time.txt")
    f = open(os.path.join(path, "time.txt"), "a")
    f.write("\n----- Start time: " + str(timestamp_start))
    f.write("----- End time: " + str(datetime.datetime.now()))
    f.close()

    # Save models when the simulation ends
    # requests.post('http://127.0.0.1:5000/save_models', json={'path': path})
    Model.save_model(path)
    Model2.save_model(path)


    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    measurements = []
    print("\nPlotting the aggregate global measures...")
    measurements.append([REWARD_STORE, 'negative_reward', "Cumulative negative reward per episode", 'Episodes',
                         'Cumulative negative reward'])
    # Visualization.save_data_and_plot(data=REWARD_STORE, filename='negative_reward', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')
    measurements.append(
        [CUMULATIVE_WAIT_STORE, 'cum_delay', "Cumulative delay per episode", 'Episodes', 'Cumulative delay [s]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==0], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==1], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==2], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==3]],  filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]',  scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE, filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    measurements.append([AVG_QUEUE_LENGTH_STORE, 'queue', "Average queue length per episode", 'Episodes',
                         'Average queue length [vehicles]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==0], [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==1], [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==2], [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==3]], filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE, filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    measurements.append(
        [AVG_WAIT_TIME_PER_VEHICLE, 'wait_per_vehicle', "Average waiting time per vehicle per episode", 'Episodes',
         'Average waiting time per vehicle [s]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==0], [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==1],  [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==2],  [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==3]], filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE, filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')

    print("\nPlotting the aggregate local measures for agent 1 (left intersection)...")
    measurements.append(
        [REWARD_STORE_A1, 'negative_reward_agent_one', "Cumulative negative reward per episode", 'Episodes',
         'Cumulative negative reward'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[REWARD_STORE_A1[i] for i in range(len(REWARD_STORE_A1)) if i%4==0], [REWARD_STORE_A1[i] for i in range(len(REWARD_STORE_A1)) if i%4==1], [REWARD_STORE_A1[i] for i in range(len(REWARD_STORE_A1)) if i%4==2], [REWARD_STORE_A1[i] for i in range(len(REWARD_STORE_A1)) if i%4==3]], filename='negative_reward_agent_one', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=REWARD_STORE_A1, filename='negative_reward_agent_one', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')
    measurements.append([CUMULATIVE_WAIT_STORE_A1, 'cum_delay_agent_one', "Cumulative delay per episode", 'Episodes',
                         'Cumulative delay [s]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[CUMULATIVE_WAIT_STORE_A1[i] for i in range(len(CUMULATIVE_WAIT_STORE_A1)) if i%4==0], [CUMULATIVE_WAIT_STORE_A1[i] for i in range(len(CUMULATIVE_WAIT_STORE_A1)) if i%4==1], [CUMULATIVE_WAIT_STORE_A1[i] for i in range(len(CUMULATIVE_WAIT_STORE_A1)) if i%4==2], [CUMULATIVE_WAIT_STORE_A1[i] for i in range(len(CUMULATIVE_WAIT_STORE_A1)) if i%4==3]], filename='cum_delay_agent_one', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]',  scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE_A1, filename='cum_delay_agent_one', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    measurements.append([AVG_QUEUE_LENGTH_STORE_A1, 'queue_agent_one', "Average queue length per episode", 'Episodes',
                         'Average queue length [vehicles]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_QUEUE_LENGTH_STORE_A1[i] for i in range(len(AVG_QUEUE_LENGTH_STORE_A1)) if i%4==0], [AVG_QUEUE_LENGTH_STORE_A1[i] for i in range(len(AVG_QUEUE_LENGTH_STORE_A1)) if i%4==1], [AVG_QUEUE_LENGTH_STORE_A1[i] for i in range(len(AVG_QUEUE_LENGTH_STORE_A1)) if i%4==2], [AVG_QUEUE_LENGTH_STORE_A1[i] for i in range(len(AVG_QUEUE_LENGTH_STORE_A1)) if i%4==3]], filename='queue_agent_one',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE_A1, filename='queue_agent_one',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    measurements.append(
        [AVG_WAIT_TIME_PER_VEHICLE_A1, 'wait_per_vehicle_agent_one', "Average waiting time per vehicle per episode",
         'Episodes', 'Average waiting time per vehicle [s]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_WAIT_TIME_PER_VEHICLE_A1[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE_A1)) if i%4==0], [AVG_WAIT_TIME_PER_VEHICLE_A1[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE_A1)) if i%4==1],  [AVG_WAIT_TIME_PER_VEHICLE_A1[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE_A1)) if i%4==2],  [AVG_WAIT_TIME_PER_VEHICLE_A1[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE_A1)) if i%4==3]], filename='wait_per_vehicle_agent_one', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE_A1, filename='wait_per_vehicle_agent_one', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')

    print("\nPlotting the aggregate local measures for agent 2 (right intersection)...")
    measurements.append(
        [REWARD_STORE_A2, 'negative_reward_agent_two', "Cumulative negative reward per episode", 'Episodes',
         'Cumulative negative reward'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[REWARD_STORE_A2[i] for i in range(len(REWARD_STORE_A2)) if i%4==0], [REWARD_STORE_A2[i] for i in range(len(REWARD_STORE_A2)) if i%4==1], [REWARD_STORE_A2[i] for i in range(len(REWARD_STORE_A2)) if i%4==2], [REWARD_STORE_A2[i] for i in range(len(REWARD_STORE_A2)) if i%4==3]], filename='negative_reward_agent_two', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=REWARD_STORE_A2, filename='negative_reward_agent_two', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')

    measurements.append([CUMULATIVE_WAIT_STORE_A2, 'cum_delay_agent_two', "Cumulative delay per episode", 'Episodes',
                         'Cumulative delay [s]'])
    # Visualization.save_data_and_plot_multiple_curves(list_of_data=[[CUMULATIVE_WAIT_STORE_A2[i] for i in range(len(CUMULATIVE_WAIT_STORE_A2)) if i%4==0], [CUMULATIVE_WAIT_STORE_A2[i] for i in range(len(CUMULATIVE_WAIT_STORE_A2)) if i%4==1], [CUMULATIVE_WAIT_STORE_A2[i] for i in range(len(CUMULATIVE_WAIT_STORE_A2)) if i%4==2], [CUMULATIVE_WAIT_STORE_A2[i] for i in range(len(CUMULATIVE_WAIT_STORE_A2)) if i%4==3]], filename='cum_delay_agent_two', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]',  scenarios=['High', 'Low', 'EW', 'NS'])
    # Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE_A2, filename='cum_delay_agent_two', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    measurements.append([AVG_QUEUE_LENGTH_STORE_A2, 'queue_agent_two', "Average queue length per episode", 'Episodes',
                         'Average queue length [vehicles]'])
    # Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE_A2, filename='queue_agent_two',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    measurements.append(
        [AVG_WAIT_TIME_PER_VEHICLE_A2, 'wait_per_vehicle_agent_two', "Average waiting time per vehicle per episode",
         'Episodes', 'Average waiting time per vehicle [s]'])
    # Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE_A2, filename='wait_per_vehicle_agent_two', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')

    # save_to_visualize(measurements)
    print("\nCalculating Average loss of models...")
    # measurements.append([MIN_LOSS_A1, 'min_loss_agent_one', "Minimum MAE loss of the first model per episode", 'Episodes', 'Minimum MAE'])
    # measurements.append([AVG_LOSS_A1, 'avg_loss_agent_one', "Average MAE loss of the first model per episode", 'Episodes', 'Average MAE'])
    Visualization.save_data_and_plot(data=MIN_LOSS_A1, filename='min_loss_agent_one',
                                     title="Minimum MSE loss of the first model per episode", xlabel='Episodes',
                                     ylabel='Minimum MSE')
    Visualization.save_data_and_plot(data=AVG_LOSS_A1, filename='avg_loss_agent_one',
                                     title="Average MSE loss of the first model per episode", xlabel='Episodes',
                                     ylabel='Average MSE')

    # measurements.append([MIN_LOSS_A2, 'min_loss_agent_two', "Minimum MAE loss of the first model per episode", 'Episodes', 'Minimum MAE'])
    # measurements.append([AVG_LOSS_A2, 'avg_loss_agent_two', "Average MAE loss of the first model per episode", 'Episodes', 'Average MAE'])
    Visualization.save_data_and_plot(data=MIN_LOSS_A2, filename='min_loss_agent_two',
                                     title="Minimum MSE loss of the second model per episode", xlabel='Episodes',
                                     ylabel='Minimum MSE')
    Visualization.save_data_and_plot(data=AVG_LOSS_A2, filename='avg_loss_agent_two',
                                     title="Average MSE loss of the second model per episode", xlabel='Episodes',
                                     ylabel='Average MSE')

    # print("\nPlotting the fundamental diagrams of traffic flow depending on the scenario...")
    # s1 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==0]  , [FLOW[i] for i in range(len(FLOW)) if i%4==0])
    # s2 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==1]  , [FLOW[i] for i in range(len(FLOW)) if i%4==1])
    # s3 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==2]  , [FLOW[i] for i in range(len(FLOW)) if i%4==2])
    # s4 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==3]  , [FLOW[i] for i in range(len(FLOW)) if i%4==3])

    # Visualization.save_data_and_plot_fundamental_diagram(data=s1, filename='fundamental_diagram_High', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='High')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s2, filename='fundamental_diagram_Low', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='Low')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s3, filename='fundamental_diagram_EW', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='EW')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s4, filename='fundamental_diagram_NS', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='NS')

    # Visualization.save_data_and_plot_multiple_fundamental_diagram(data=[s1, s2, s3, s4], filename='fundamental_diagram', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])

    # print("\nPlotting the occupancy diagrams of traffic flow depending on the scenario...")

    # o1 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==0]  , [FLOW[i] for i in range(len(FLOW)) if i%4==0])
    # o2 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==1]  , [FLOW[i] for i in range(len(FLOW)) if i%4==1])
    # o3 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==2]  , [FLOW[i] for i in range(len(FLOW)) if i%4==2])
    # o4 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==3]  , [FLOW[i] for i in range(len(FLOW)) if i%4==3])

    # Visualization.save_data_and_plot_fundamental_diagram(data=o1, filename='occ_fundamental_diagram_High', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='High')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o2, filename='occ_fundamental_diagram_Low', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='Low')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o3, filename='occ_fundamental_diagram_EW', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='EW')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o4, filename='occ_fundamental_diagram_NS', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='NS')

    # Visualization.save_data_and_plot_multiple_fundamental_diagram(data=[o1, o2, o3, o4], filename='occ_fundamental_diagram', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])
