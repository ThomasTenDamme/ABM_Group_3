import mesa
import mesa.agent
import mesa.model
import numpy as np
############
from collections import Counter
from scipy.stats import entropy
############
import concurrent.futures
import geopandas as gpd
import mesa.space
from shapely.geometry import Point

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

def property_value_equal(name, width, height) -> mesa.space.PropertyLayer:
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    for i in range(height):
        for j in range(width):
            rent = 1

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


def property_value_from_gdf(name, width, height, gdf) -> mesa.space.PropertyLayer:
    # Create the PropertyLayer
    layer = mesa.space.PropertyLayer(name, width, height, 0)
    
    # Iterate over the cells in the PropertyLayer
    for i in range(height):
        for j in range(width):
            # Create a point for the current cell
            point = Point(j, i)
            
            # Find the corresponding property in the GeoDataFrame
            match = gdf[gdf.geometry.contains(point)]
            
            if not match.empty:
                # Get the rent value from the GeoDataFrame
                rent = match.iloc[0]['rent']
            else:
                # Default rent if no match is found
                rent = 0

            # Set the cell value in the PropertyLayer
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

def utility_func(model: mesa.Model, agent: mesa.Agent, property_loc: tuple, budgetless = False) -> float:
    theta, _ = get_theta(model, property_loc, agent.type)

    desirability = model.desirability_layer.data[property_loc]

    alpha = model.alpha

    budget = agent.budget
    
    price = model.price_func(model, property_loc)
    
    if budget < price and not budgetless:
        return 0

    return theta**alpha*desirability**(1-alpha)


def price_func(model: mesa.Model, property_loc: tuple) -> float:
    
    desirability = model.desirability_layer.data[property_loc]
    property_value = model.property_value_layer.data[property_loc]
    
    return (0.5 + desirability) * property_value

def price_func_cap(model: mesa.Model, property_loc: tuple, param:float) -> float:
    desirability = model.desirability_layer.data[property_loc]
    property_value = model.property_value_layer.data[property_loc]

    initial_price = property_value

    price= (0.5 + desirability) * property_value

    price_cap = param * initial_price 
    
    if price > price_cap:
        #print("price_cap applied")
        return price_cap
    
    return price

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
        return NO_NEIGHBORS_THETA, NO_NEIGHBORS_THETA
       
    proportion_similar = similar / num_neighbours

    theta = np.exp(-((proportion_similar - model.mu_theta) ** 2) / (2 * model.sigma_theta ** 2))
    
    return theta, proportion_similar
    #return proportion_similar


def compute_similar_neighbours(model:mesa.Model,agent:mesa.Agent):
    similar = 0
    num_neighbours = 0

    type = agent.type
    pos = agent.pos
    for neighbor in model.grid.iter_neighbors(
        pos, moore=True, radius=model.radius
    ):
        
        num_neighbours += 1
        if neighbor.type == type:
            similar += 1
    
    if num_neighbours == 0:
        return 0 #NO NEIGHBORS

    return similar

def compute_entropy(model: mesa.model):
    # Flatten the layer's cell values into a single list
    desirability = model.desirability_layer.data 
    cell_values = [cell_value for row in desirability for cell_value in row]
    
    # Compute the frequency of each unique value
    value_counts = Counter(cell_values)
    
    # Compute the probabilities for each unique value
    total_cells = len(cell_values)
    probabilities = [count / total_cells for count in value_counts.values()]
    
    # Calculate the entropy using scipy's entropy function
    layer_entropy = entropy(probabilities)
    
    return layer_entropy

def calculate_gi_star(grid, values, x, y, d):
    sum_wx = 0
    sum_w = 0
    sum_wx2 = 0
    n = len(values)

    for (i, j), value in values.items():
        dist = np.sqrt((x - i)**2 + (y - j)**2)
        if dist <= d:
            w = 1  # Binary weight, 1 if within distance threshold
            sum_wx += w * value
            sum_w += w
            sum_wx2 += w * value**2

    mean_x = np.mean(list(values.values()))
    s = np.std(list(values.values()))

    numerator = sum_wx - mean_x * sum_w
    denominator = s * np.sqrt((n * sum_w - sum_w**2) / (n - 1))

    return numerator / denominator if denominator != 0 else 0

def agent_to_interested_grid(inputs):
    model = inputs[0]
    agent = inputs[1]
    grid = model.grid
    
    interested = np.zeros((grid.width, grid.height))

    for x in range(grid.width):
        for y in range(grid.height):
            interested[x, y] = 1 if model.utility_func(model, agent, (x, y), budgetless=False) > agent.utility else 0
    return interested

def update_interested_agents_concurrently(model):
    agents = model.schedule.agents
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        inputs = [(model, agent) for agent in agents]
        interested_agents = executor.map(agent_to_interested_grid, inputs)
    
        interested_agents = np.sum(list(interested_agents), axis=0)
        model.interested_agents_layer.data = interested_agents
    