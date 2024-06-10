import mesa
import numpy as np
import mesa
import random

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

    def step(self):
        similar = 0
        for neighbor in self.model.grid.iter_neighbors(
            self.pos, moore=True, radius=self.model.radius
        ):
            if isinstance(neighbor, SchellingAgent) and neighbor.type == self.type:
                similar += 1


        available_cells = self.model.find_available_cells(self.budget)
        if len(available_cells) > 0:
            destination = random.choice(available_cells)
            self.model.grid.move_agent(self, destination)



class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(
        self,
        height=20,
        width=20,
        homophily=3,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        seed=None,
        property_value=100
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

        self.property_values = np.full((width, height), property_value)
        self.desirability = np.full((width, height), 1)

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        self.happy = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={"happy": "happy"},  # Model-level count of happy agents
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
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        self.datacollector.collect(self)

        if self.happy == self.schedule.get_agent_count():
            self.running = False
