import pathlib

my_folder = './yolov5/runs/detect/try/labels'
first_col =[]
for path in pathlib.Path(my_folder).glob("*.txt"):
    #print(str(path))
    with open(path) as f:
        lines = f.readlines()
        if len(lines)>=1:
            for line in lines:
                first_col.append(line.split(' ')[0])
num_madeliefje = first_col.count('0')
num_paardenbloem = first_col.count('1')
print('number of madeliefje is %d'%num_madeliefje)
print('number of paardenbloem is %d'%num_paardenbloem)
None
