
path = "/home/bagro/CSC2530GroupProject/BARF/output/plant/nerf_baseline/quant.txt" 

# read the rows of the space separated csv file and avage across all rows
with open(path, 'r') as f:
    lines = f.readlines()
    # remove the first row
    lines = lines[1:]
    # split the rows into columns
    lines = [line.split() for line in lines]
    # convert the columns to floats
    lines = [[float(x) for x in line] for line in lines]
    # average across all rows
    average = [sum(x)/len(x) for x in zip(*lines)]
    print(average)