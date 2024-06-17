import mesa
from model import Schelling
from modules import property_value_func_random, utility_func, price_func, income_func, property_value_quadrants, desirability_func, compute_similar_neighbours


def get_happy_agents(model):
    """
    Display a text count of how many happy agents there are.
    """
    return f"Happy agents: {1}"


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
happy_chart = mesa.visualization.ChartModule([{"Label": "happy", "Color": "Black"}])

model_params = {
    "property_value_func": property_value_quadrants,
    "income_func": income_func,
    "utility_func": utility_func,
    "price_func": price_func,
    "desirability_func": desirability_func,
    ####
    "compute_similar_neighbours": compute_similar_neighbours,
    ####
    "height": height,
    "width": width,
    "density": mesa.visualization.Slider(
        name="Agent density", value=0.8, min_value=0.1, max_value=1.0, step=0.05
    ),
    "minority_pc": mesa.visualization.Slider(
        name="Fraction minority", value=0.2, min_value=0.00, max_value=1.0, step=0.05
    ),
    "homophily": mesa.visualization.Slider(
        name="Homophily", value=0.5, min_value=0, max_value=1, step=0.05
    ),
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
        
    
    # TODO - add all sliders
}

server = mesa.visualization.ModularServer(
    model_cls=Schelling,
    visualization_elements=[canvas_element, get_happy_agents, happy_chart],
    name="Schelling Segregation Model",
    model_params=model_params,
)

