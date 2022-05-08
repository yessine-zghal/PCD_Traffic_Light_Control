import traci
import numpy as np
import random
import timeit
import requests


# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation():
    def __init__(self, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_cells, num_states, num_actions, training_epochs):
        
        self._TrafficGen = TrafficGen # the traffic generator class
        self._gamma = gamma #agent performances between 0 and 1
        self._step = 0 #step counter
        self._sumo_cmd = sumo_cmd #cmd command to run sumo at simulation time
        self._max_steps = max_steps #maximum number of steps in a simulation
        self._green_duration = green_duration #fixed green duration, in seconds, for every green phase
        self._yellow_duration = yellow_duration #fixed yellow duration, in seconds, for every yellow phase
        self._num_cells = num_cells #number of cells in the state repr
        self._num_states = num_states #number of states in the state space repr
        self._num_actions = num_actions #number of actions
        self._training_epochs = training_epochs #number of training epochs for the model
        
        #metrics store
        self._reward_store = [] #store the rewards for every simulation
        self._cumulative_wait_store = [] #store the cumulative waiting times for every simulation
        self._avg_queue_length_store = [] #store the average queue length for every simulation
        self._avg_loss = [] #store the average loss for every training epoch
        self._avg_wait_time_per_vehicle = [] #store the average waiting time for every vehicle for every simulationTeX
        self._min_loss = [] #store the minimum loss of the model (MSE, MAE or Huber)
        self._list_density = [] #list density
        self._list_flow = [] #list flow
        self._avg_density = [] #average density
        self._avg_flow = [] #average flow
        self._list_occupancy = [] #list_occupancy
        


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._model_training_loss = []
        self._already_in = []
        self._density = []
        self._flow = []
        self._occupancy = []
        self._cumulative_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        # run the simulation
        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state_with_advanced_perception()
            
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            #reward = (0.9 * old_total_wait) - current_total_wait
            
            self._cumulative_waiting_time+= current_total_wait #cumulative waiting time

            # saving the data into the memory
            if self._step != 0:
                #self._Memory.add_sample((old_state, old_action, reward, current_state))
                #print(type(old_state), type(old_action), type(reward), type(current_state))
                #print(type(old_state.tolist()))
                requests.post('http://127.0.0.1:5000/add_sample', json={'old_state':  old_state.tolist(),
                                                                    'old_action': int(old_action),
                                                                    'reward': reward,
                                                                    'current_state': current_state.tolist()})

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        print("Saving episodes stats...")
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # print("Training...")
        # start_time = timeit.default_timer()
        # for _ in range(self._training_epochs):
        #     #self._replay()
        #     tr_loss = requests.post('http://127.0.0.1:5000/replay', json={'num_states': self._num_states,
        #                                                       'num_actions': self._num_actions,
        #                                                       'gamma': self._gamma}).json()['loss']
        #     #print(tr_loss)
        #     self._model_training_loss.append(tr_loss)
        # training_time = round(timeit.default_timer() - start_time, 1)
        
        # if(len(self._model_training_loss) > 0):
        #     print("Saving loss results...")
        #     #print(self._model_training_loss)
        #     self._avg_loss.append(sum(self._model_training_loss)/self._training_epochs)
        #     self._min_loss.append(min(self._model_training_loss))

        return simulation_time#, training_time


    def _simulate(self, steps_todo):    
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds

            #at each time step
            self._flow.append(self._get_flow())
            self._density.append(self._get_density())
            self._occupancy.append(self._get_occupancy())


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            pred = np.array(requests.post('http://127.0.0.1:5000/predict', json={'state': state.tolist()}).json()['prediction'])
            return np.argmax(pred)
            #return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_N + halt_S + halt_E + halt_W
    
    def _get_density(self):
        """
        Retrieve the density (vehicles per km) of every edges/lanes
        """
        divider = traci.lane.getLength("N2TL_0") / 1000 #  derives the id of the first lane from the edge id (all lanes of an edge have the same length) and 1000 m -> km
        den_N = traci.edge.getLastStepVehicleNumber("N2TL") / divider
        den_S = traci.edge.getLastStepVehicleNumber("S2TL") / divider
        den_E = traci.edge.getLastStepVehicleNumber("E2TL") / divider
        den_W = traci.edge.getLastStepVehicleNumber("W2TL") / divider
        return den_N + den_S + den_E + den_W
    
    def _get_flow(self):
        """
        Retrieve the flow (vehicles per hour) of every edges/lanes
        """
        counter_entered = 0
        #Returns the list of ids of vehicles that were on the named edge in the last simulation step.
        ids_N = traci.edge.getLastStepVehicleIDs("N2TL")
        ids_S = traci.edge.getLastStepVehicleIDs("S2TL")
        ids_W = traci.edge.getLastStepVehicleIDs("W2TL")
        ids_E = traci.edge.getLastStepVehicleIDs("E2TL")
        car_list = ids_N + ids_S + ids_W + ids_E
        for car_id in car_list:
            if car_id not in self._already_in:
                counter_entered+=1
                self._already_in.append(car_id)
        return (counter_entered/5400)*3600
                
    def _get_occupancy(self):
        """
        Retrieve the occupancy of every edges
        It is the ratio of the sum of the lengths of the vehicles to the length of the road section in which those vehicles are present in %.
        """
        occ_N = traci.edge.getLastStepOccupancy("N2TL")
        occ_S = traci.edge.getLastStepOccupancy("S2TL")
        occ_W = traci.edge.getLastStepOccupancy("W2TL")
        occ_E = traci.edge.getLastStepOccupancy("E2TL")
        return (occ_N + occ_S + occ_W + occ_E)/4
        

    
    def _get_state_with_advanced_perception(self):
        """
        Retrieve the state of the intersection from sumo, in the form of four arrays concatenated representing respectively :
        - the number of cars per each cell
        - the average speed of cars in each cell
        - the cumulated waiting time of vehicles per each cell
        - the number of queued cars per each cell
        """
        #Initialize the four arrays that will form our state representation
        nb_cars = np.zeros(self._num_cells)
        avg_speed = np.zeros(self._num_cells)
        cumulated_waiting_time = np.zeros(self._num_cells)
        nb_queued_cars = np.zeros(self._num_cells)
        
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_speed = traci.vehicle.getSpeed(car_id)
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                nb_cars[car_position] += 1  
                avg_speed[car_position] += car_speed
                # A speed of less than 0.1 m/s is considered a halt.
                if (car_speed  < 0.1):
                    nb_queued_cars[car_position] += 1
                cumulated_waiting_time[car_position] += wait_time
                
                     
        #avg_speed is an accumulative speed for the moment, we need to divide by the number of cars to obtain the average speed
        for i in range(len(avg_speed)):
            if (nb_cars[i] > 1):
                avg_speed[i] /= nb_cars[i] # avg_speed[i] = avg_speed[i] / nb_cars[i] 
                
            
        #State is now a vector of 80 * 4
        state = np.concatenate((nb_cars, avg_speed, cumulated_waiting_time, nb_queued_cars))

        #print(state.shape)
        return state



    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN
            self._model_training_loss.append(self._Model.training_loss) #get the MAE loss 
            
            
    def _calculate_avg_loss(self):
        """
        Calculate the average loss of the model depending on scenario
        """
        self._avg_loss = [sum(elts)/self._training_epochs for elts in zip(*self._total_loss)]


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        #self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._cumulative_wait_store.append(self._cumulative_waiting_time) #cumulative wait time in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_wait_time_per_vehicle.append(self._cumulative_waiting_time/self._sum_queue_length) #how much time a vehicle wait in an episode
        self._list_density.append(self._density)
        self._list_flow.append(self._flow)
        self._list_occupancy.append(self._occupancy)
        
        
    @property
    def avg_density_and_flow(self):
        avg_den = [sum(i)/len(self._list_density) for i in zip(*self._list_density)]
        d_max = max(avg_den) #maximum density
        self._max_index = avg_den.index(d_max)
        self._avg_density = avg_den[:self._max_index+1]
        self._avg_flow = [sum(i)/len(self._list_flow) for i in zip(*self._list_flow)][:self._max_index+1]
        return self._avg_density, self._avg_flow
        
    @property
    def get_avg_density_and_flow(self):
        return self._avg_density, self._avg_flow
    
    @property
    def get_avg_occupancy_and_flow(self):
        avg_occ = [sum(i)/len(self._list_occupancy) for i in zip(*self._list_occupancy)]
        o_max = max(avg_occ) #maximum occupancy
        max_index = avg_occ.index(o_max)
        avg_occ = avg_occ[:max_index+1]
        avg_flow = [sum(i)/len(self._list_flow) for i in zip(*self._list_flow)][:max_index+1]
        return avg_occ, avg_flow
    
    @property
    def avg_wait_time_per_vehicle(self):
        return self._avg_wait_time_per_vehicle
    
    @property
    def avg_loss(self):
        return self._avg_loss
    
    @property
    def min_loss(self):
        return self._min_loss

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
    
    @property
    def density(self):
        return self._density
    
    @property
    def flow(self):
        return self._flow
    
    @property
    def occupancy(self):
        return self._occupancy
    
    
    
    #End simulation
    def stop(self):
        return self.reward_store[0], self.cumulative_wait_store[0], self.avg_queue_length_store[0], self.avg_wait_time_per_vehicle[0],self.density, self.flow, self.occupancy #self.min_loss[0], self.avg_loss[0], 
 
    
