import mesa
import mesa.agent
import numpy as np

def property_value_func(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    for i in range(height):
        for j in range(width):
            layer[i][j] = abs(np.random.gumbel())

    return layer

def utility_func(model: mesa.Model, agent: mesa.Agent, agent_loc: tuple, property_loc: tuple) -> float:
    
    theta = model.get_theta(agent_loc, model.grid.get_property(property_loc))

    desirability = model.desirability_layer(property_loc)

    alpha = model.alpha

    budget = agent.budget
    
    price = model.price_func(property_loc)
    
    return theta**alpha*desirability**(1-alpha)*((budget-price)/budget)

def price_func(model: mesa.Model, property_loc: tuple) -> float:
    
    desirability = model.desirability_layer[property_loc]
    property_value = model.property_value_layer[property_loc]
    
    return (0.5 + desirability) * property_value