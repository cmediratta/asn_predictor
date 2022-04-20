from src.asnDataset import get_train_test
import random

def perturb_data(filename):

  train, test = get_train_test(filename, 0, True)

  num_by_category = [[],[],[],[],[]]

  train_size = 0
  num_new = 0

  with open("data/" + filename[:(len(filename)-4)]+"Perturbed.csv", "w") as file:

    for i in range(len(train)):
      num_by_category[int(train[i][1])].append(train[i])
      train_size+=1
      write_to_file(train[i], file)

    largest = len(max(num_by_category, key=len))
    for i in range(5):
      for j in range(len(num_by_category[i]),int(largest)):
        point_to_perturb = num_by_category[i][int(random.random()*len(num_by_category[i]))]
        for k in range(5):
          point_to_perturb[0][k]*=(1.05-random.random()*0.1)
        train_size+=1
        write_to_file(point_to_perturb, file)

    for i in range(len(test)):
      num_by_category[int(test[i][1])].append(test[i])
      write_to_file(test[i], file)

  return train_size



def write_to_file(torch_array, file):
  writestring = "perturbed," + str(torch_array[0][0]) + "," + str(torch_array[0][1]) + "," + str(torch_array[0][2]) + ","\
                + str(torch_array[0][3]) + "," + str(torch_array[0][4]) + "," + str(int(torch_array[1])) + "\n"
  file.write(writestring)



