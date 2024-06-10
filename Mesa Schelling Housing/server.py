import mesa
from model import Schelling
from modules import property_value_func, utility_func, price_func


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


canvas_element = mesa.visualization.CanvasGrid(
    portrayal_method=schelling_draw,
    grid_width=10,
    grid_height=10,
    canvas_width=500,
    canvas_height=500,
)
happy_chart = mesa.visualization.ChartModule([{"Label": "happy", "Color": "Black"}])

model_params = {
    "property_value_func": property_value_func,
    "utility_func": utility_func,
    "price_func": price_func,
    "height": 10,
    "width": 10,
    "density": mesa.visualization.Slider(
        name="Agent density", value=0.8, min_value=0.1, max_value=1.0, step=0.1
    ),
    "minority_pc": mesa.visualization.Slider(
        name="Fraction minority", value=0.2, min_value=0.00, max_value=1.0, step=0.05
    ),
    "homophily": mesa.visualization.Slider(
        name="Homophily", value=0.5, min_value=0, max_value=1, step=0.1
    ),
    "radius": mesa.visualization.Slider(
        name="Search Radius", value=1, min_value=1, max_value=5, step=1
    ),
    
    # TODO - add all sliders
}

server = mesa.visualization.ModularServer(
    model_cls=Schelling,
    visualization_elements=[canvas_element, get_happy_agents, happy_chart],
    name="Schelling Segregation Model",
    model_params=model_params,
)