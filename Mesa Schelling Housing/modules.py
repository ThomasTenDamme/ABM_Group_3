import mesa
import mesa.agent
import numpy as np


NO_NEIGHBORS_THETA = 0.5

def property_value_func_random(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    # parameters
    beta = 400
    mu = 1200
    
    for i in range(height):
        for j in range(width):
            rent = np.random.gumbel(loc=mu, scale=beta)
            layer.set_cell((i, j), abs(rent))

    return layer

def property_value_quadrants(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    for i in range(height):
        for j in range(width):
            if i < height/2:
                if j < width/2:
                    rent = 500
                else:
                    rent = 1000
            else:
                if j < width/2:
                    rent = 1000
                else:
                    rent = 2000
            layer.set_cell((i, j), abs(rent))

    return layer

def income_func(scale=1.5):
    # Use the same gumbel distribution as the property value, but scale with scale value
    return np.random.gumbel(loc=400, scale=1200) * scale

# def utility_func(model: mesa.Model, agent: mesa.Agent, property_loc: tuple) -> float:
#     # agent_loc = agent.pos
    
#     theta = model.get_theta(property_loc, agent.type)

#     desirability = model.desirability_layer.data[property_loc]

#     alpha = model.alpha

#     budget = agent.budget
    
#     price = model.price_func(model, property_loc)
    
#     return theta**alpha*desirability**(1-alpha)*((budget-price)/budget)

def desirability_func(model: mesa.Model, prop_value_weight: float = 0.1) -> float:
    
    
    most_expensive_prop = np.max(model.property_value_layer.data)
    num_agents = len(model.schedule.agents)
    
    return prop_value_weight * model.property_value_layer.data / most_expensive_prop + (1-prop_value_weight) * model.interested_agents_layer.data / num_agents

def utility_func(model: mesa.Model, agent: mesa.Agent, property_loc: tuple) -> float:
    # agent_loc = agent.pos
    
    theta = get_theta(model, property_loc, agent.type)

    desirability = model.desirability_layer.data[property_loc]

    alpha = model.alpha

    budget = agent.budget
    
    price = model.price_func(model, property_loc)
    
    if budget < price:
        return 0
    
    return theta**alpha*desirability**(1-alpha)


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