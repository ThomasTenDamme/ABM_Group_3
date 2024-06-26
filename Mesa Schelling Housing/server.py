import mesa
from model import Schelling
from modules import update_interested_agents_concurrently, property_value_func_random, utility_func, price_func, income_func, property_value_quadrants, desirability_func, compute_similar_neighbours, property_value_equal, calculate_gi_star


def get_average_utility(model):
    """
    Display a text count of the average utility.
    """
    return f"Average Utility: {sum([agent.utility for agent in model.schedule.agents])/len(model.schedule.agents)}"


def schelling_draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}

    if agent.type == 0:
        portrayal["Color"] = ["#FF0000", "#FF9999"]
        portrayal["stroke_color"] = "#00FF00"
    else:
        portrayal["Color"] = ["#0000FF", "#9999FF"]
        portrayal["stroke_color"] = "#000000"
    return portrayal

width = 20
height = 20

canvas_element = mesa.visualization.CanvasGrid(
    portrayal_method=schelling_draw,
    grid_width=width,
    grid_height=height,
    canvas_width=500,
    canvas_height=500,
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
    visualization_elements=[canvas_element, get_average_utility, utility_chart],
    name="Schelling Segregation Model with Housing Market",
    model_params=model_params,
)

