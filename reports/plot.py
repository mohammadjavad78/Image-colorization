dict={}
with open("report.csv",'r') as f:
    lines=f.readlines()
for i in range(75):
    dict[i+1]={}
for line in lines[1:]:
    name=line.split('model_name,')[1].split(',')[0]
    i=line.split('original,')[1].split(',')[0]
    if(name=="train"):
        dict[int(i)][name]=float(line.split(',')[-3])
    if(name=="test"):
        dict[int(i)][name]=float(line.split(',')[-1].split('\n')[0])
    if(name=="val"):
        dict[int(i)][name]=float(line.split(',')[-2])

# print(dict)
import matplotlib.pyplot as plt
trains=[dict[i+1]['train'] for i in range(75)]
plt.plot(trains,label="Train")
# plt.show()


trains=[dict[i+1]['test'] for i in range(75)]
plt.plot(trains,label="test")
# plt.show()


trains=[dict[i+1]['val'] for i in range(75)]
plt.plot(trains,label="val")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()