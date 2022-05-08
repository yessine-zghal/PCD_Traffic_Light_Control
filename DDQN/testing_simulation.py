import traci
import numpy as np
import timeit
import re

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_cells, num_states, num_actions, n_cars, stl):
        self._Model = Model
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
        self._queue_length_episode = []
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
        old_total_wait = 0
        old_action = -1 # dummy init
        
        #Metrics
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._waits = [0 for i in range(self._n_cars)]
        
        #for static traffic lights, we need to set the initial phases manually
        action_rotation=[0,1,2,3]
        ar=0
        while self._step < self._max_steps:
            self._action_steps.append(self._step)

            # get current state of the intersection
            current_state = self._get_state_with_advanced_perception()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            #STL : predefined actions, NOT STL: model-based action
            if(self._stl):
                action = action_rotation[ar]
                if action % 2 == 0:
                    self._green_duration = 30
                else:
                    self._green_duration = 15
                if (ar==3): 
                    ar=0
                else:
                    ar+=1
            else:
                # choose the light phase to activate, based on the current state of the intersection
                action = self._choose_action(current_state)
            
            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait
            
            if reward < 0:
                self._sum_neg_reward += reward
            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))  
        #self._save_episode_stats()
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._sum_queue_length += queue_length
            self._queue_length_episode.append(queue_length)
            self._get_waiting_vehicles()
            
            
    def _get_waiting_vehicles(self):
        """
        Store if the vehicle of id *car_id* has waited (1) or not (0) in waits[].
        """
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            i = int(re.findall(r'\d+', car_id)[0])
            if traci.vehicle.getWaitingTime(car_id) > 0 and self._waits[i] == 0:
                self._waits[i] += 1
        #print(sum(self._waits))
                


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


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


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
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    
    def _get_state_with_advanced_perception(self):
        """
        Retrieve the state of the intersection from sumo, in the form of four arrays representing respectively :
        - the number of cars per each cell
        - the average speed of cars in each cell
        - the cumulated waiting time per each cell
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


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode

    @property
    def xs(self):
        return self._action_steps

