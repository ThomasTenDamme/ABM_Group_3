import mesa
import numpy as np
import mesa
import random

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

    def step(self):
        """
        Step for agent to move
        """


        available_cells = self.model.find_available_cells(self.budget)
        if len(available_cells) < 0:
            return
        
        # destination = random.choice(available_cells)
        # self.model.grid.move_agent(self, destination)
        
        # TODO - iterate over available cells to find the highest utility, move if higher than current
        
        # TODO - update utility value for current location (only if moved)


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(
        self,
        property_value_func,
        utility_func,
        price_func,
        height=20,
        width=20,
        homophily=0.5,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        seed=None
    ):
        """
        Create a new Schelling model.

        Args:
            width, height: Size of the space.
            density: Initial chance for a cell to be populated
            minority_pc: Chance for an agent to be in minority class
            homophily: Minimum number of agents of the same class needed to be happy
            radius: Search radius for checking similarity
            seed: Seed for reproducibility
            property_value: Value for the property
        """

        super().__init__(seed=seed)
        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.radius = radius
        
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        # Property Value Layer
        self.property_value_layer = property_value_func(name="property_values", width=width, height=height, torus=True)
        self.grid.add_property_layer(self.property_value_layer)

        # Desirability Layer
        self.desirability_layer = mesa.space.PropertyLayer("desirability", width, height, torus=True)
        for _, pos in self.grid.coord_iter():
            self.desirability_layer[pos] = 1
        self.grid.add_property_layer(self.desirability_layer)
        
        # Interested Agents Counter Layer
        self.interested_agents_layer = mesa.space.PropertyLayer("interested_agents", width, height, torus=True)
        for _, pos in self.grid.coord_iter():
            self.interested_agents_layer[pos] = 0
        self.grid.add_property_layer(self.interested_agents_layer)
        
        # self.happy = 0
        self.datacollector = mesa.DataCollector(
            # model_reporters={"happy": "happy"},  # Model-level count of happy agents
        )

        # Set up agents
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < self.minority_pc else 0
                budget = 1000
                agent = SchellingAgent(self.next_id(), self, agent_type, budget)
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)

        self.datacollector.collect(self)

    def get_theta(self, loc: tuple, type):
        similar = 0
        num_neighbours = 0
        
        for neighbor in self.model.grid.iter_neighbors(
            loc, moore=True, radius=self.model.radius
        ):
            
            num_neighbours += 1
            if neighbor.type == type:
                similar += 1
        
        if num_neighbours == 0:
            return NO_NEIGHBORS_THETA
                
        proportion_similar = similar / num_neighbours
        
        return proportion_similar

    def find_available_cells(self, budget, utility_agent):
        available_cells = []
        for _, pos in self.grid.coord_iter():
            if self.grid.is_cell_empty(pos) and self.property_values[pos] < budget and self.utility_house[pos] > utility_agent:
                available_cells.append(pos)        
        return available_cells

    def step(self):
        """
        Run one step of the model.
        """
        # TODO - reset every cell's counter in the interested_agents_layer
        
        for agent in self.schedule.agents:
            # TODO - Calc current agent's utility at current location
            utility = 0
            
            # TODO - iterate over cells and compare utility to current location, add to interested_agents_layer if better
            
        for cell in self.grid.coord_iter():
            # TODO - update desirability layer by normalizing interested_agents_layer value
            pass
        
        
        self.schedule.step()
        self.datacollector.collect(self)
