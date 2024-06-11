"""
lines = open("nn_plot_2.txt", "r").readlines()
trimmed = open("nn_trimmed.txt", "w")

for line in lines:
    line = line.strip()
    if (line != "Training complete." and line[0:8] == "Training") or line[0:5] == "Epoch":
        trimmed.write(line + "\n")

file_names = []
lines = open("nn_trimmed.txt", "r").readlines()
temp = open("nn_plot.txt", "r")
for line in lines:
    line = line.strip()
    pieces = line.split()
    #print(pieces)
    #print(pieces[3])
    if line[0:8] == "Training":
        file_names.append(line)
        temp.close()
        temp = open(line, "w")
    elif pieces[2] == "Train":
        #print("here")
        #print(pieces[-1])
        temp.write(pieces[-1] + "\n")

print(file_names)
"""

import matplotlib.pyplot as plt

file_names = ['Training ctrl + uptake model', 'Training ctrl + eliciting model', 'Training ctrl + talktime model', 'Training ctrl + grade model', 'Training tutor + uptake model', 'Training tutor + eliciting model', 'Training tutor + talktime model', 'Training tutor + grade model', 'Training tutor + talktime model', 'Training tutor + grade model', 'Training tutor_social + uptake model', 'Training tutor_social + eliciting model', 'Training tutor_social + talktime model', 'Training tutor_social + grade model', 'Training tutor_goal + uptake model', 'Training tutor_goal + eliciting model', 'Training tutor_goal + talktime model', 'Training tutor_goal + grade model']



nested = [['Training ctrl + uptake model', 'Training tutor + uptake model', 'Training tutor_social + uptake model', 'Training tutor_goal + uptake model'], ['Training ctrl + eliciting model', 'Training tutor + eliciting model', 'Training tutor_social + eliciting model', 'Training tutor_goal + eliciting model'], ['Training ctrl + grade model', 'Training tutor + grade model', 'Training tutor_social + grade model', 'Training tutor_goal + grade model'], ['Training ctrl + talktime model', 'Training tutor + talktime model', 'Training tutor_social + talktime model', 'Training tutor_goal + talktime model']]
x_axis = [i for i in range(10, 251, 10)]
#print(numbers)

for n in nested:
    all_y = []
    for file in n:
        file = open(file, "r")
        y_axis = []
        for line in file.readlines():
            y_axis.append(float(line.strip()))
            if len(y_axis) == len(x_axis):
                break
        all_y.append(y_axis)
    plt.plot(x_axis, all_y[0], label="Control")
    plt.plot(x_axis, all_y[1], label="Tutor Feedback")
    plt.plot(x_axis, all_y[2], label="Tutor + social")
    plt.plot(x_axis, all_y[3], label="Tutor + goal")
    plt.legend()
    plt.xlabel("Num Epochs")
    plt.ylabel("Loss")
    plt.title(n[0].split()[3].capitalize() + " Model")
    plt.show()

"""
for f in file_names:
    file = open(f, "r")
    y_axis = []
    for line in file.readlines():
        y_axis.append(float(line.strip()))

    plt.plot(x_axis, y_axis)
    plt.xlabel("X-axis data")
    plt.ylabel("Y-axis data")
    plt.title('multiple plots')
    plt.show()
"""
    