import mesa
import numpy as np
import mesa
import random
import matplotlib.pyplot as plt
import modules

NO_NEIGHBORS_THETA = 0.5

class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, unique_id, model, agent_type, budget):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
           budget: Budget for the agent
        """
        super().__init__(unique_id, model)
        self.type = agent_type
        self.budget = budget
        self.utility = 0.5
        self.segregation = None
        self.move_counter = 0

    def calc_theta(self):
        # Calculate theta using the model's get_theta method
        self.segregation = modules.get_theta(self.model, self.pos, self.type)

    def step(self):
        """
        Step for agent to move
        In a step an agent will:
            1. Find available properties to move to
            2. Calculate their utility for each property
            3. If the property with the highest utility has a higher utility than the current property, move there
            4. Update the utility of the agent in their new location
        """
        # update utility
        self.utility = self.model.utility_func(self.model, self, self.pos)
        
        self.calc_theta()

        # find the available properties to move to
        available_cells = self.model.find_available_cells(self)
                
        if len(available_cells) < 0:
            return
        
        # list all utilities of available properties
        move_util = []
        for cell in available_cells:
            # store as (cell, utility) tuple
            move_util.append((cell, self.model.utility_func(self.model, self, cell)))
        
        # sort by utility
        move_util.sort(key=lambda x: x[1], reverse=True)
        
        # move if utility is higher than current
        if move_util[0][1] > self.utility:
            self.model.grid.move_agent(self, move_util[0][0])
            # update utility
            self.utility = move_util[0][1]
            self.move_counter += 1


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(
        self,
        property_value_func,
        income_func,
        desirability_func,
        utility_func,
        price_func,
        ##########
        compute_similar_neighbours,
        ##########
        height=20,
        width=20,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        alpha=0.5,
        income_scale=1.5, # the scale by which the income is higher than the property value
        property_value_weight=0.1,
        mu_theta = 0.4,
        sigma_theta = 0.6,
        agent_entropy = None,
        desirability_entropy = None, 
        seed=None
    ):
        """
        Create a new Schelling model.

        Args:
            width, height: Size of the space.
            density: Initial chance for a cell to be populated
            minority_pc: Chance for an agent to be in minority class
            radius: Search radius for checking similarity
            seed: Seed for reproducibility
            property_value: Value for the property
        """

        super().__init__(seed=seed)
        self.utility_func = utility_func
        self.price_func = price_func
        self.desirability_func = desirability_func
        self.prop_value_weight = property_value_weight
        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.radius = radius
        self.alpha = alpha
        self.mu_theta = mu_theta
        self.sigma_theta = sigma_theta
        self.agent_entropy = agent_entropy
        self.desirability_entropy = desirability_entropy
        
        #############
        self.compute_similar_neighbours = compute_similar_neighbours
        self.neighbor_similarity_counter = {}
        #############

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        # Property Value Layer
        self.property_value_layer = property_value_func(name="property_values", width=width, height=height)
        self.grid.add_property_layer(self.property_value_layer)

        # Desirability Layer
        self.desirability_layer = mesa.space.PropertyLayer("desirability", width, height, 0.5)
        # for _, pos in self.grid.coord_iter():
        #     self.desirability_layer[pos] = 1
        self.grid.add_property_layer(self.desirability_layer)
        
        # Interested Agents Counter Layer
        self.interested_agents_layer = mesa.space.PropertyLayer("interested_agents", width, height, 0)
        # for _, pos in self.grid.coord_iter():
        #     self.interested_agents_layer[pos] = 0
        self.grid.add_property_layer(self.interested_agents_layer)
        
        # Utility Layer
        self.utility_layer = mesa.space.PropertyLayer("utility", width, height, 0.5) 
        self.grid.add_property_layer(self.utility_layer)

        #Data Collectors
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Utility": "utility", "Segregation":"segregation", "Moves":"move_counter"}, model_reporters={"Desirability": self.desirability_layer.data.tolist}  # Collect the utility of each agent
        )

        # Set up agents
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < self.minority_pc else 0
                budget = income_func(scale=income_scale)
                agent = SchellingAgent(self.next_id(), self, agent_type, budget)
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)

        self.datacollector.collect(self)

    def find_available_cells(self, agent):
        available_cells = []
        for _, pos in self.grid.coord_iter():
            if self.grid.is_cell_empty(pos):
                available_cells.append(pos)        
        return available_cells

    def step(self):
        """
        Run one step of the model.
        """
        # Set the count of agents who like to move somewhere to 0 for all cells
        self.interested_agents_layer.set_cells(0)

        ########
        self.neighbor_similarity_counter.clear()
        ########

        for agent in self.schedule.agents:
            # Iterate over cells and compare utility to current location, add to interested_agents_layer if better
            for _, loc  in self.grid.coord_iter():
                utility = self.utility_func(self, agent, loc)
                
                if utility > agent.utility:
                    self.interested_agents_layer.modify_cell(loc, lambda v: v + 1)

        ###### ADDED #############
            # Compute number of agents with the same number of similar neighbours 
            similar_neighbors = self.compute_similar_neighbours(self, agent)
            if similar_neighbors not in self.neighbor_similarity_counter:
                self.neighbor_similarity_counter[similar_neighbors] = 0
            self.neighbor_similarity_counter[similar_neighbors] += 1

        # Compute total number of agents included
        total_agents = len(self.schedule.agents) #sum(self.neighbor_similarity_counter.values())

        # Compute agent entropy and store it 
        current_agent_entopy = 0
        for _, p in self.neighbor_similarity_counter.items():
            if p > 0:  # To avoid domain error for log(0)
                probability = p / total_agents
                value = probability * np.log10(probability)
                current_agent_entopy += value
        self.agent_entropy = -current_agent_entopy
        #############################
        
        # Set desirability layer to the proportion of interested agents
        #num_agents = len(self.schedule.agents)
        self.desirability_layer.set_cells(
            self.desirability_func(self, prop_value_weight=self.prop_value_weight)
        )
        
        ##### Compute entropy for desirability ###########
        desirability_current_entropy = modules.compute_entropy(self)
        self.desirability_entropy = desirability_current_entropy
        ############################

        self.schedule.step()
        self.datacollector.collect(self)
        

# import modules
# # Create and run the model
# model = Schelling(
    # property_value_func=modules.property_value_quadrants,
    # utility_func=modules.utility_func,
    # price_func=modules.price_func,
    # height=20,
    # width=20,
    # radius=1,
    # density=0.8,
    # minority_pc=0.2,
    # alpha=0.5,
    # seed=42
# )

# # Run the model for a certain number of steps
# for i in range(5):
    # print(i)
    # model.step()

# # Retrieve the collected data
# agent_data = model.datacollector.get_agent_vars_dataframe()
# model_data = model.datacollector.get_model_vars_dataframe()

# print(agent_data)

### ADDED ####
import modules
# # Create and run the model
model = Schelling(
     property_value_func=modules.property_value_quadrants,
     income_func=modules.income_func,
     desirability_func=modules.desirability_func,
     utility_func=modules.utility_func,
     price_func=modules.price_func,
     compute_similar_neighbours=modules.compute_similar_neighbours,
     height=20,
     width=20,
     radius=1,
     density=0.8,
     minority_pc=0.2,
     alpha=0.5,
     seed=42
 )

# # Run the model for a certain number of steps
for i in range(5):
     print(i)
     #print(model.entropy)
     model.step()