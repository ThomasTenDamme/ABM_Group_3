import mesa
import mesa.agent


def property_value_func(name, width, height, torus=True) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, torus)
    
    # TODO
    
    return layer

def utility_func(model: mesa.Model, agent_loc: tuple, property_loc: tuple) -> float:
    
    # TODO
    
    return 0.0

def price_func(model: mesa.Model, property_loc: tuple) -> float:
    
    # TODO
    
    return 0.0