from random import randint

def connect_squares_into_islands(graph) -> list: 
        """
        Connects neighboring squares of graph into a blob. returns a list of blobs

        Parameters: 
        -----------
        graph : list of coordinates [y, x]
            Graph previously generated in generate_depth_graph
        """
        blobs = []
        for node in graph: 
            graph.remove(node)
            blob = [node]
            neighbors = get_neighbors(node, graph)
            while len(neighbors) > 0: 
                neighbor = neighbors.pop()
                blob.append(neighbor)
                try:
                    if graph.count(neighbor) > 0:
                        graph.remove(neighbor)
                except ValueError as e: 
                    print(e)
                    print(neighbor)
                    break
                neighbors = neighbors + get_neighbors(neighbor, graph)
            blobs.append(blob) 
        return blobs
    
def get_neighbors(coord, graph) -> list: 
        """
        Returns neighboring grid tiles present in graph

        ... 

        Parameters: 
        coord: tuple(y, x)
            Grid coordinate of tile
        graph: list[tuple(y,x)]
            List of all elligible tiles
        """
        x = coord[1]
        y = coord[0]
        neighbors = []
        for yn in range(-1,2): 
            for xn in range(-1,2):
                if not yn and not xn or x+xn<0 or x+yn<0: continue
                if (y + yn, x + xn) in graph: 
                    neighbors.append((y + yn, x + xn)) 
        return neighbors

graph = [(randint(0,10), randint(0,10)) for i in range(100)]
blobs = connect_squares_into_islands(graph)
sorted_blobs = [sorted(blob, key=lambda tup: tup[0]) for blob in blobs]

blobs_2d = []
for blob in sorted_blobs:
    blobs_rows = []
    blob_row = []
    curr_row = blob[0][0]
    for coord in blob:
        if coord[0] == curr_row: 
            blob_row.append(coord)
        else:
            blobs_rows.append(blob_row)
            curr_row = coord[0]
            blob_row = [coord]
    blobs_2d.append(blobs_rows)

for i, blob in enumerate(blobs_2d): 
    blobs_2d[i] = [sorted(row, key=lambda coord: coord[1]) for row in blob]

def prune_blobs(blob):
    for row in blob:

for blob in blob: 
    pruned_blobs = prune_blobs(blob)
# We want to: 
# Sort blobs by column and row 
# check if len is less than min height or width 
# Then look for columns / rows with a width greater than the others by some threshold
# delete gridsquares that go past or before it
# should get rectangularish blobs
# For vars, we need min x, min y, max x, max y, then we can know +/- the threshold where to remove squares 
#       with bounds checking 
# also need max squares in image

