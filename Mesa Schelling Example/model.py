import mesa


class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, unique_id, model, agent_type):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           x, y: Agent initial location.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(unique_id, model)
        self.type = agent_type

    def step(self):
        similar = 0
        for neighbor in self.model.grid.iter_neighbors(
            self.pos, moore=True, radius=self.model.radius
        ):
            if neighbor.type == self.type:
                similar += 1

        total_neighbours = (2 * self.model.radius + 1) ** 2 - 1
        proportion_similar = similar / total_neighbours

        # If unhappy, move:
        if proportion_similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
            
            # TODO: change to move to good cell
            # for agent, loc in self.model.grid.coord_iter():
                
            
        else:
            self.model.happy += 1

def get_segregation(model):
    """
    Find the % of agents that only have neighbors of their same type.
    """
    segregated_agents = 0
    for agent in model.schedule.agents:
        segregated = True
        for neighbor in model.grid.iter_neighbors(agent.pos, True):
            if neighbor.type != agent.type:
                segregated = False
                break
        if segregated:
            segregated_agents += 1
    return segregated_agents / model.schedule.get_agent_count()

class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(
        self,
        height=20,
        width=20,
        homophily=0.3,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        seed=None,
    ):
        """
        Create a new Schelling model.

        Args:
            width, height: Size of the space.
            density: Initial Chance for a cell to populated
            minority_pc: Chances for an agent to be in minority class
            homophily: Minimum number of agents of same class needed to be happy
            radius: Search radius for checking similarity
            seed: Seed for Reproducibility
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

        self.happy = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={"happy": "happy", "Segregated_Agents" : get_segregation},  # Model-level count of happy agents
        
            agent_reporters={"x": lambda a: a.pos[0], "y": lambda a: a.pos[1], "type": "type"}
        
        )

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < self.minority_pc else 0
                agent = SchellingAgent(self.next_id(), self, agent_type)
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)

        self.datacollector.collect(self)
    
    def step(self):
        """
        Run one step of the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()

        self.datacollector.collect(self)

        if self.happy == self.schedule.get_agent_count():
            self.running = False
            
