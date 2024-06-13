import mesa
import mesa.agent
import numpy as np

def property_value_func(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    for i in range(height):
        for j in range(width):
            layer.set_cell((i, j), abs(np.random.gumbel()))

    return layer

def utility_func(model: mesa.Model, agent: mesa.Agent, property_loc: tuple) -> float:
    # agent_loc = agent.pos
    
    theta = model.get_theta(property_loc, agent.type)

    desirability = model.desirability_layer.data[property_loc]

    alpha = model.alpha

    budget = agent.budget
    
    price = model.price_func(model, property_loc)
    
    return theta**alpha*desirability**(1-alpha)*((budget-price)/budget)

def price_func(model: mesa.Model, property_loc: tuple) -> float:
    
    desirability = model.desirability_layer.data[property_loc]
    property_value = model.property_value_layer.data[property_loc]
    
    return (0.5 + desirability) * property_value