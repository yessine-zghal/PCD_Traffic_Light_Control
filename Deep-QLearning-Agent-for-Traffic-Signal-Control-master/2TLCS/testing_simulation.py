import traci
import numpy as np
import timeit
import re
import os


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
    def __init__(self, Model_A1, Model_A2, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_cells, num_states, num_actions, n_cars, stl):
        
        self._Model_A1 = Model_A1
        self._Model_A2 = Model_A2
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_cells = num_cells
        self._num_states = num_states
        self._num_actions = num_actions
        self._n_cars = n_cars
        self._reward_episode = []
        self._reward_TL1_episode = []
        self._reward_TL2_episode = []
        self._queue_length_episode = []
        self._queue_length_TL1_episode = []
        self._queue_length_TL2_episode = []
        self._stl = stl #True or False, if we use the static traffic lights or not


        self._action_steps = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait_one = 0
        old_total_wait_two = 0
        old_action_one = -1
        old_action_two = -1
        
        #Metrics
        self._sum_neg_reward = 0
        self._sum_neg_reward_one = 0
        self._sum_neg_reward_two = 0

        self._sum_queue_length = 0
        self._sum_queue_length_a1 = 0
        self._sum_queue_length_a2 = 0

        self._waits = [0 for i in range(self._n_cars)]
        self._pass = [0 for i in range(self._n_cars)]
        action_rotation=[0,1,2,3]
        ar=0

        while self._step < self._max_steps:

            # get current state of each of the intersections
            current_state_one, current_state_two = self._get_states_with_advanced_perception()

            if (self._num_states == 321):
                #Adding the knowledge of the other agent previous action
                current_state_one = np.append(current_state_one, old_action_two)
                current_state_two = np.append(current_state_two, old_action_one)         
            
            
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            ## Reward per agents
            # current_total_wait_one = self._collect_waiting_times_first_intersection()
            # reward_one = old_total_wait_one - current_total_wait_one
            # current_total_wait_two = self._collect_waiting_times_second_intersection()
            # reward_two = old_total_wait_two - current_total_wait_two

            ## New reward per agents
            current_total_wait_one = 0.2 * self._collect_waiting_times_first_intersection() + self._get_queue_length_intersection_one() 
            reward_one = old_total_wait_one - current_total_wait_one
            current_total_wait_two = 0.2 * self._collect_waiting_times_second_intersection() + self._get_queue_length_intersection_two()
            reward_two = old_total_wait_two - current_total_wait_two

            ## Mutual reward
            reward_one += 0.5 * reward_two
            reward_two += 0.5 * reward_one
            
            ## STL : predefined actions, NOT STL: model-based action
            if(self._stl):
                action_one = action_rotation[ar]
                action_two = action_rotation[ar]
                #print(action_one)
                if action_one % 2 == 0:
                    self._green_duration = 30
                else:
                    self._green_duration = 15
                if (ar==3): 
                    ar=0
                else:
                    ar+=1
            else:
                # choose the light phase to activate, based on the current state of the first intersection
                action_one = self._choose_action(current_state_one, self._Model_A1)
                # choose the light phase to activate, based on the current state of the second intersection
                action_two = self._choose_action(current_state_two, self._Model_A2)

            # if the chosen phase is different from the last phase, activate the yellow phase
            #Simultaneity of the 2 traffic lights : manages different cases 
            #Two in yellow phases
            if self._step != 0 and old_action_one != action_one and old_action_two != action_two:
                self._set_yellow_phase(old_action_one)
                self._set_yellow_phase_two(old_action_two)
                self._simulate(self._yellow_duration) 
            elif self._step != 0 and old_action_one != action_one:
                self._set_yellow_phase(old_action_one)
                self._simulate(self._yellow_duration)
            elif self._step != 0 and old_action_two != action_two:
                self._set_yellow_phase_two(old_action_two)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action_one)
            self._set_green_phase_two(action_two)
            self._simulate(self._green_duration)
            
            old_action_one = action_one
            old_action_two = action_two
            old_total_wait_one = current_total_wait_one
            old_total_wait_two = current_total_wait_two

            # saving only the meaningful rewards to better see if the agents are behaving correctly
            if reward_one < 0:
                self._sum_neg_reward_one += reward_one
                
            if reward_two < 0:
                self._sum_neg_reward_two += reward_two

            #Reward episode
            self._reward_episode.append(reward_one + reward_two)
            self._reward_TL1_episode.append(reward_one)
            self._reward_TL2_episode.append(reward_two)

        #print("Total reward:", self._sum_neg_reward_one + self._sum_neg_reward_two)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

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
            queue_length = self._get_queue_length_intersection_one() + self._get_queue_length_intersection_two()
            self._sum_queue_length += queue_length 
            self._queue_length_episode.append(queue_length)

            queue_length_1 = self._get_queue_length_intersection_one()
            self._sum_queue_length_a1 += queue_length_1
            self._queue_length_TL1_episode.append(queue_length_1)

            queue_length_2 = self._get_queue_length_intersection_two()
            self._sum_queue_length_a2 += queue_length_2
            self._queue_length_TL2_episode.append(queue_length_2)

            
            self._get_waiting_vehicles()
            #print(sum(self._waits))
            self._get_pass_vehicles()
            #print(sum(self._pass))
            
       
    #Test vehicles
    def _get_pass_vehicles(self):
        """
        Test function to check if the number of vehicles that should pass the intersection is correct
        """
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            listi = re.findall(r'\d+', car_id)
            i = int(listi[len(listi)-1])
            #print("P", i)
            if(self._pass[i] == 0):
                self._pass[i] += 1
            
            
    def _get_waiting_vehicles(self):
        """
        Store if the vehicle of id *car_id* has waited (1) or not (0) in waits[].
        """
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            listi = re.findall(r'\d+', car_id)
            i = int(listi[len(listi)-1])
            if traci.vehicle.getWaitingTime(car_id) > 0 and self._waits[i] == 0:
                self._waits[i] += 1
        

    def _collect_waiting_times_first_intersection(self):
        """
        Retrieve the waiting time of every car in the incoming roads of the first intersection (left one)
        """
        incoming_roads = ["N2TL", "W2TL", "S2TL", "2_TL2W"]
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
    
    def _collect_waiting_times_second_intersection(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["2_N2TL", "2_W2TL", "2_S2TL", "TL2E"]
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


    def _choose_action(self, state, model):
        """
        Pick the best action known based on the current state of the env
        Method for testing not with Flask
        """
        return np.argmax(model.predict_one(state))



    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo for the first intersection 'TL'
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_yellow_phase_two(self, old_action):
        """
        Activate the correct yellow light combination in sumo for the second intersection 'TL_2'
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("2_TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo for the first intersection 'TL'
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
            
            
    def _set_green_phase_two(self, action_number):
        """
        Activate the correct green light combination in sumo for the second intersection 'TL'
        """
        if action_number == 0:
            traci.trafficlight.setPhase("2_TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("2_TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("2_TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("2_TL", PHASE_EWL_GREEN)



    def _get_queue_length_intersection_one(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane of the first intersection
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("2_TL2W")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_N + halt_S + halt_E + halt_W
    
    def _get_queue_length_intersection_two(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane of the second intersection
        """
        halt_N = traci.edge.getLastStepHaltingNumber("2_N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("2_S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("2_E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("TL2E")
        return halt_N + halt_S + halt_E + halt_W
    
    def _get_density(self):
        """
        Retrieve the density (vehicles per km) of every edges/lanes
        """
        divider = traci.lane.getLength("N2TL_0") / 1000 #  derives the id of the first lane from the edge id (all lanes of an edge have the same length) and 1000 m -> km
        den_N = traci.edge.getLastStepVehicleNumber("N2TL") / divider
        den_S = traci.edge.getLastStepVehicleNumber("S2TL") / divider
        den_E = traci.edge.getLastStepVehicleNumber("2_TL2W") / divider
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
        ids_E = traci.edge.getLastStepVehicleIDs("2_TL2W")
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
        occ_E = traci.edge.getLastStepOccupancy("2_TL2W")
        return (occ_N + occ_S + occ_W + occ_E)/4
        

    
    def _get_states_with_advanced_perception(self):
        """
        Retrieve the state of the intersection from sumo, in the form of four arrays concatenated representing respectively :
        - the number of cars per each cell
        - the average speed of cars in each cell
        - the cumulated waiting time of vehicles per each cell
        - the number of queued cars per each cell
        """
        #Initialize the four arrays that will form our state representation
        nb_cars = np.zeros(self._num_cells*2)
        avg_speed = np.zeros(self._num_cells*2)
        cumulated_waiting_time = np.zeros(self._num_cells*2)
        nb_queued_cars = np.zeros(self._num_cells*2)
        
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
            elif lane_id == "2_TL2W_0" or lane_id == "2_TL2W_1" or lane_id == "2_TL2W_2":
                lane_group = 4
            elif lane_id == "2_TL2W_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            #2_xTL_x are the lanes of the second intersection
            elif lane_id == "TL2E_0" or lane_id == "TL2E_1" or lane_id == "TL2E_2":
                lane_group = 8
            elif lane_id == "TL2E_3":
                lane_group = 9
            elif lane_id == "2_N2TL_0" or lane_id == "2_N2TL_1" or lane_id == "2_N2TL_2":
                lane_group = 10
            elif lane_id == "2_N2TL_3":
                lane_group = 11
            elif lane_id == "2_E2TL_0" or lane_id == "2_E2TL_1" or lane_id == "2_E2TL_2":
                lane_group = 12
            elif lane_id == "2_E2TL_3":
                lane_group = 13
            elif lane_id == "2_S2TL_0" or lane_id == "2_S2TL_1" or lane_id == "2_S2TL_2":
                lane_group = 14
            elif lane_id == "2_S2TL_3":
                lane_group = 15
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 15:
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
        #First half for first intersection and second for second intersection  
        state_one = np.concatenate((nb_cars[:self._num_cells], avg_speed[:self._num_cells], cumulated_waiting_time[:self._num_cells], nb_queued_cars[:self._num_cells]))
        state_two = np.concatenate((nb_cars[self._num_cells:], avg_speed[self._num_cells:], cumulated_waiting_time[self._num_cells:], nb_queued_cars[self._num_cells:]))
        
        return state_one, state_two



    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        
        self._reward_store.append(self._sum_neg_reward_one + self._sum_neg_reward_two)  # how much negative reward in this episode for both agents
        self._reward_store_a1.append(self._sum_neg_reward_one)
        self._reward_store_a2.append(self._sum_neg_reward_two)
        
  
    @property
    def reward_episode(self):
        return self._reward_episode

    @property
    def reward_episode_1(self):
        return self._reward_TL1_episode

    @property
    def reward_episode_2(self):
        return self._reward_TL2_episode


    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def queue_length_episode_1(self):
        return self._queue_length_TL1_episode
    
    @property
    def queue_length_episode_2(self):
        return self._queue_length_TL2_episode
   