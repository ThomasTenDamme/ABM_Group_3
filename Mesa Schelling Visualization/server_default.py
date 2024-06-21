import mesa
from model import Schelling
from modules import update_interested_agents_concurrently, property_value_func_random, utility_func, price_func, income_func, property_value_from_gdf, property_value_quadrants, desirability_func, compute_similar_neighbours, property_value_equal, calculate_gi_star, price_func_cap


def get_average_utility(model):
    """
    Display a text count of the average utility.
    """
    print(f"Agents: {len(model.schedule.agents)}")
    # return f"Average Utility: {sum([agent.utility for agent in model.schedule.agents])/len(model.schedule.agents)}"

def color_gradient(value, min_val, max_val):
    """
    return hex code for color gradient blue to red
    """
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    # scale value to [0, 1]
    value = (value - min_val) / (max_val - min_val)
    # interpolate between magenta and green
    r = int(255 * value)
    g = 0
    b = int(255 * (1 - value))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def schelling_draw(agent, value_io_desire=False, draw_agents=True):
    """
    Portrayal Method for canvas
    value_io_desire = boolean, if True the value of the property is visualized instead of the desirability
    """
    # portray property value
    if agent is None:
        return
    
    # on layer 0, portray property value which has agent type -1
    if agent.type == -1:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

        prop_val = agent.model.property_value_layer.data[agent.pos]

        if prop_val < 0:
            color = "#FFFFFF"
        elif value_io_desire:
            color = color_gradient(prop_val, 0, 10000)
        else: 
            desirability = agent.model.desirability_layer.data[agent.pos]
            color = color_gradient(desirability, 0, 1)
        portrayal["Color"] = [color, color]

        return portrayal

    # on layer 1, portray agents
    if draw_agents:
        portrayal = {"Shape": "circle", "r": 0.6, "Filled": "true", "Layer": 1}

        if agent.type == 0:
            portrayal["Color"] = ["#FFFFFF", "#FFFFFF"]
            portrayal["stroke_color"] = "#000000"
        else:
            portrayal["Color"] = ["#000000", "#000000"]
            portrayal["stroke_color"] = "#FFFFFF"
        return portrayal

def draw_main(agent):
    return schelling_draw(agent, value_io_desire=False, draw_agents=True)

def draw_other(agent):
    return schelling_draw(agent, value_io_desire=True, draw_agents=True)

def whitespace(_):
    return ""

width = 20
height = 20

# grid of agents and desirability
canvas_main = mesa.visualization.CanvasGrid(
    portrayal_method=draw_main,
    grid_width=width,
    grid_height=height,
    canvas_width=500,
    canvas_height=500
)

utility_chart = mesa.visualization.ChartModule([
    {"Label": "Average Utility", "Color": "Black"},
    {"Label": "Minority Average Utility", "Color": "Blue"},
    {"Label": "Majority Average Utility", "Color": "Red"}])

model_params = {
    "property_value_func": property_value_quadrants,
    "income_func": income_func,
    "utility_func": utility_func,
    "price_func": price_func,
    "update_interested_agents_func" : update_interested_agents_concurrently,
    "desirability_func": desirability_func,
    ####
    "compute_similar_neighbours": compute_similar_neighbours,
    "calculate_gi_star": calculate_gi_star,
    "price_func_cap": price_func_cap,
    "policy_singapore" : False,
    ####
    "height": height,
    "width": width,
    "density": mesa.visualization.Slider(
        name="Agent density", value=0.8, min_value=0.1, max_value=1.0, step=0.05
    ),
    "minority_pc": mesa.visualization.Slider(
        name="Fraction minority", value=0.2, min_value=0.00, max_value=1.0, step=0.05
    ),
    #"homophily": mesa.visualization.Slider(
     #   name="Homophily", value=0.5, min_value=0, max_value=1, step=0.05
    #),
    # "radius": mesa.visualization.Slider(
    #     name="Search Radius", value=1, min_value=1, max_value=5, step=1
    # ),
    "alpha": mesa.visualization.Slider(
        name="Alpha", value=0.5, min_value=0, max_value=1, step=0.05
    ),
    "income_scale": mesa.visualization.Slider(
        name="Income Scale (over rent)", value=1.5, min_value=1, max_value=3, step=0.1
    ),
    "property_value_weight": mesa.visualization.Slider(
        name="Property Value Weight", value=0.1, min_value=0, max_value=1, step=0.05
    ),
    "mu_theta": mesa.visualization.Slider(
        name="Mu Theta", value=0.7, min_value=0, max_value=1, step=0.05
    ),
    "sigma_theta": mesa.visualization.Slider(
        name="Sigma Theta", value=0.3, min_value=0, max_value=1, step=0.05
    ),
        
    
    # TODO - add all sliders
}

server = mesa.visualization.ModularServer(
    model_cls=Schelling,
    visualization_elements=[
        canvas_main,
        whitespace, 
        get_average_utility, 
        utility_chart
    ],
    name="Schelling Segregation Model with Housing Market",
    model_params=model_params,
)

