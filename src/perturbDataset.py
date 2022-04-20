from src.asnDataset import get_train_test
import random

"""
This function is built to create a perturbed dataset to create more robustness
in our model. It takes existing datapoints and perturbs them slightly so that
the model counts loss evenly among catagories.

Inputs:
filename - the file to perturb

Outputs:
A file in the /data folder under the name (filename-.csv)Perturb.csv which
    is the perturbed dataset in addition to the initial train then the test data.
train_size - the size of the training data in the new (filename-.csv)Perturb.csv
"""
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


"""
Helper function to write data point to csv file.

Inputs:
torch_array - format np.array([float,float,float,float,float],float)
file - file to write to

Output:
torch_array gets written in csv format to file
"""
def write_to_file(torch_array, file):
  writestring = "perturbed," + str(torch_array[0][0]) + "," + str(torch_array[0][1]) + "," + str(torch_array[0][2]) + ","\
                + str(torch_array[0][3]) + "," + str(torch_array[0][4]) + "," + str(int(torch_array[1])) + "\n"
  file.write(writestring)



