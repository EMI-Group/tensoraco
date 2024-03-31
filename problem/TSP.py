import numpy as np

class TSP():
    def read_tsp_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        coordinates = []
        node_coord_section = False

        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                node_coord_section = True
            elif node_coord_section and line.strip() != "EOF":
                parts = line.split()
                coordinates.append((float(parts[1]), float(parts[2])))

        return np.array(coordinates)

    def calculate_distance_matrix(coordinates):
        num_nodes = len(coordinates)
        distance_matrix = np.ones((num_nodes, num_nodes))
        distance_matrix *= np.inf

        for i in range(num_nodes):
            if i % 1000 == 0:
                print("Processed {} out of {} nodes".format(i, num_nodes))
            for j in range(i + 1, num_nodes):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    # Specify the path to your tsp file
    tsp_file_path = "pcb1173.tsp"

    # Read coordinates from the tsp file
    node_coordinates = read_tsp_file(tsp_file_path)

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(node_coordinates)

