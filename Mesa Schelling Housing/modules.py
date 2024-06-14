import mesa
import mesa.agent
import numpy as np

NO_NEIGHBORS_THETA = 0.5

def property_value_func(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    for i in range(height):
        for j in range(width):
            value = abs(np.random.gumbel())
            layer.set_cell((i, j), value)

    return layer

def utility_func(model: mesa.Model, agent: mesa.Agent, property_loc: tuple) -> float:
    # agent_loc = agent.pos
    
    theta = get_theta(model, property_loc, agent.type)

    desirability = model.desirability_layer.data[property_loc]

    alpha = model.alpha

    budget = agent.budget
    
    price = model.price_func(model, property_loc)
    
    return theta**alpha*desirability**(1-alpha)*((budget-price)/budget)

def price_func(model: mesa.Model, property_loc: tuple) -> float:
    
    desirability = model.desirability_layer.data[property_loc]
    property_value = model.property_value_layer.data[property_loc]
    
    return (0.5 + desirability) * property_value

def get_theta(model: mesa.Model, loc: tuple, type):
    similar = 0
    num_neighbours = 0
    
    for neighbor in model.grid.iter_neighbors(
        loc, moore=True, radius=model.radius
    ):
        
        num_neighbours += 1
        if neighbor.type == type:
            similar += 1
    
    if num_neighbours == 0:
        return NO_NEIGHBORS_THETA
            
    proportion_similar = similar / num_neighbours

    # theta = np.exp(-((proportion_similar - self.mu_theta) ** 2) / (2 * self.sigma_theta ** 2))

    # return theta #proportion_similar
    
    return proportion_similar