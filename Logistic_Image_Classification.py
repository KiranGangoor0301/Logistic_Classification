#!/usr/bin/env python
# coding: utf-8

# In[184]:


ROOT_DIR = r"C:\Users\Kirankumar Gangoor\OneDrive\Documents\CogniFirst\Input"


# In[185]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install opencv_python_headless')
get_ipython().system('pip install torch')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')
get_ipython().system('pip install torchvision')


# ---

# In[186]:


import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[187]:


import os

Training_folder = ROOT_DIR + "\Data\Training_data"
os.listdir(Training_folder)


# In[188]:


from torchvision.datasets import ImageFolder

training_dataset = ImageFolder(root=Training_folder)


# In[189]:


for path in os.listdir(Training_folder):
    for i in range(3):
        temp_path = os.path.join(Training_folder, path)
        file = random.choice(os.listdir(temp_path))
        image_path = os.path.join(temp_path, file)
        img = mpimg.imread(image_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(img)


# In[190]:


IMG_WIDTH = 200
IMG_HEIGHT = 200


Train_folder = ROOT_DIR + '\Data\Training_data'  
Test_folder = ROOT_DIR + '\Data\Testing_Data'   


# In[191]:


def create_dataset(Train_folder):
    img_data_array = []  
    class_name = []      
    
    classes = {'driving_license': [1, 0, 0], 'others': [0, 1, 0], 'social_security': [0, 0, 1]}
    
    for PATH in os.listdir(Train_folder):
        for file in os.listdir(os.path.join(Train_folder, PATH)):
            image_path = os.path.join(Train_folder, PATH, file)
            
           
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            
            image = np.array(image).astype('float64')
            
        
            if len(image.shape) == 3:
                img_data_array.append(np.array(image).reshape([3, IMG_HEIGHT, IMG_WIDTH]))
                class_name.append(classes[PATH]) 
    
    return img_data_array, class_name


# In[192]:


Train_img_data, train_class_name = create_dataset(Train_folder)

Test_img_data, test_class_name = create_dataset(Test_folder)

len(Train_img_data)


# In[193]:


train_class_name[0]


# ---

# ### Implementing a CNN in PyTorch

# In[194]:


import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.utils.data as Data
from torch import Tensor
from torch.autograd import Variable


# In[195]:


print(torch.__version__)


# In[196]:


torch_dataset_train = Data.TensorDataset(
    torch.Tensor(np.array(Train_img_data)),  
    torch.Tensor(np.array(train_class_name))  
)

torch_dataset_test = Data.TensorDataset(
    torch.Tensor(np.array(Test_img_data)),  
    torch.Tensor(np.array(test_class_name))  
)


# In[197]:


trainloader = Data.DataLoader(
    torch_dataset_train, 
    batch_size=8,         
    shuffle=True          
)

testloader = Data.DataLoader(
    torch_dataset_test,   
    batch_size=8,         
    shuffle=True          
)


# In[198]:


dataiter = iter(trainloader)

images = next(dataiter)
images[0].shape


# In[199]:


import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionNet, self).__init__()

       
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
        return x


input_size = 3 * 200 * 200  
output_size = 3

model = LogisticRegressionNet(input_size, output_size)

optimizer = optim.SGD(model.parameters(), lr=0.0001)  
criterion = nn.CrossEntropyLoss()  

if torch.cuda.is_available():
    model = model.to("cuda")
    criterion = criterion.to("cuda")

=print(model)


# In[ ]:


get_ipython().system('export CUDA_LAUNCH_BLOCKING=1')


# In[ ]:


for i in range(10):
    running_loss = 0
    model.train() 
    for images, labels in trainloader:
        if torch.cuda.is_available():
            images = images.to("cuda")
            labels = labels.to("cuda")


        optimizer.zero_grad()

        output = model(images)

        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i + 1, running_loss / len(trainloader)))


# In[ ]:


filepath = r"C:\Users\Kirankumar Gangoor\OneDrive\Documents\CogniFirst\output\model.pt"
torch.save(model.state_dict(), filepath)


# In[ ]:


y_pred_list = []  
y_true_list = []  

with torch.no_grad():
    for x_batch, y_batch in testloader:
        x_batch, y_batch = x_batch.to(), y_batch.to()
        
        y_test_pred = model(x_batch)
        print(y_test_pred) 
        
        _, y_pred_tag = torch.max(y_test_pred, dim=1)
        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())


# In[ ]:


y_true_list_max = [m.argmax() for m in y_true_list]


# In[ ]:


correct_count, all_count = 0, 0

for i in range(len(y_pred_list)):
    if y_pred_list[i] == y_true_list_max[i]:
        correct_count += 1
    all_count += 1

accuracy = correct_count / all_count
print("\nModel Accuracy =", accuracy)


# ---

# In[ ]:


import torch
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class LogisticRegressionNet(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionNet, self).__init__()
        self.fc = torch.nn.Linear(200 * 200 * 3, 3)  

    def forward(self, x):
        x = x.view(-1, 200 * 200 * 3)  # Flatten the image
        x = self.fc(x)
        return x

def load_image(image_path, img_height=200, img_width=200):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_width, img_height))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    image = transform(image)
    
    
    image = image.unsqueeze(0)
    
    return image

def predict_image(model, image_path, class_names=['driving_license', 'others', 'social_security']):
    image = load_image(image_path)
    
    if torch.cuda.is_available():
        image = image.to("cuda")
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
    
    _, predicted_class = torch.max(output, 1)
    
    predicted_label = class_names[predicted_class.item()]
    
    return predicted_label

model = LogisticRegressionNet()  
model.load_state_dict(torch.load(r"C:\Users\Kirankumar Gangoor\OneDrive\Documents\CogniFirst\output\model.pt"))
model.eval()

image_path = r"C:\Users\Kirankumar Gangoor\OneDrive\Documents\CogniFirst\Input\Data\Testing_Data\others\105.jpg"
predicted_class = predict_image(model, image_path)
print(f'The predicted class is: {predicted_class}')

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

model_file = "cnn_modelnew.h5"
scripted_model = torch.jit.script(model)
scripted_model.save(model_file)

print(f"Model saved as {model_file}")


# In[ ]:




