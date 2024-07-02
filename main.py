import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset


class Net(nn.Module):
    def __init__(self, isDeconvolution=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv2T = nn.ConvTranspose2d(16, 6, 5, stride=1, padding=0, output_padding=0)
        self.conv1T = nn.ConvTranspose2d(6, 3, 5, stride=1, padding=0, output_padding=0) 
        self.isDeconvolution = isDeconvolution

    def forward(self, x, zero_all_except=-1, layer=1):
        x, indices1 = self.pool(F.relu(self.conv1(x)))
        if zero_all_except != -1 and layer == 1:
          for i in range(x.size(1)):
            if i != zero_all_except:
              x[0, i, :, :] = 0
        x, indices2 = self.pool(F.relu(self.conv2(x)))
        if zero_all_except != -1 and layer == 2:
          for i in range(x.size(1)):
            if i != zero_all_except:
              x[0, i, :, :] = 0
        if not self.isDeconvolution:
          x = torch.flatten(x, 1)  # flatten all dimensions except batch
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x
        else:
          x = self.conv2T(F.relu(self.unpool(x, indices2)))
          x = self.conv1T(F.relu(self.unpool(x, indices1)))
              
          return x

    def set_isDeconvolution(self, isDeconvolution):
        self.isDeconvolution = isDeconvolution

class CustomLoss(nn.Module):
    def __init__(self, lambda_rec=1.0):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, x_recon, x_orig): #customm loss function
        ce_term = self.ce_loss(y_pred, y_true)
        # rec_term = 0
        # for example in range(x_recon.size(0)): #iterate all examples in batch
        #   mse_all_channel = 0
        #   for channel in range(x_recon.size(1)):  # Iterate over channels
        #     flattened_recon = x_recon[example, channel, :, :].flatten(start_dim=1)
        #     flattened_orig = x_orig[example, channel, :, :].flatten(start_dim=1)
        #     mse_all_channel += F.mse_loss(flattened_recon, flattened_orig, reduction='sum')
        #   rec_term += mse_all_channel / x_recon.size(1)
        # rec_term /= x_recon.size(0)
        rec_term = F.mse_loss(x_recon, x_orig, reduction='sum') / (x_recon.size(0) * x_recon.size(1)) # Use mean squared error for reconstruction loss
        total_loss = ce_term + self.lambda_rec * rec_term
        return total_loss



class Trainer:
    def __init__(self, dataShrinkFactor=1, reconstructionRegularized = False):
        # Device
        self._set_device()
        self.dataShrinkFactor = dataShrinkFactor
        # Dataset and Dataloaders
        self._prepare_dataloaders()
      
        # Model
        self.net = Net().to(self.device)

        # Loss and Optimizer
        self.reconstructionRegularized = reconstructionRegularized
        if not reconstructionRegularized:
          self.criterion = nn.CrossEntropyLoss()
        else:
          self.criterion = CustomLoss(lambda_rec=0.1)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def _set_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def _prepare_dataloaders(self):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        
        self.batch_size = 4

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        self.trainset =  Subset(self.trainset, torch.randperm(len(self.trainset))[:int(len(self.trainset) / self.dataShrinkFactor)])
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=self.transform)
        self.testset =  Subset(self.testset, torch.randperm(len(self.testset))[:int(len(self.testset) / self.dataShrinkFactor)])
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def display_images(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = next(dataiter)

        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)))

    def train_network(self):
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.net.set_isDeconvolution(False)

                # forward + backward + optimize
                outputs = self.net(inputs)
                if not self.reconstructionRegularized:
                  loss = self.criterion(outputs, labels)
                else:
                  self.net.set_isDeconvolution(True)
                  reconstructions = self.net(inputs)
                  loss = self.criterion(outputs, labels, reconstructions, inputs)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % (2000/self.dataShrinkFactor) == int(1999/self.dataShrinkFactor): 
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (int(2000/self.dataShrinkFactor)):.3f}')
                    running_loss = 0.0

        print('Finished Training')

        # Save model
        PATH = './cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)


    def test_network(self):
        dataiter = iter(self.testloader)
        images, labels = next(dataiter)
        self.net.set_isDeconvolution(False)

        # print images
        self.imshow(torchvision.utils.make_grid(images))

        self.net = Net().to(self.device)
        PATH = './cifar_net.pth'
        self.net.load_state_dict(torch.load(PATH))

        images = images.to(self.device)
        outputs = self.net(images)

        _, predicted = torch.max(outputs, 1)
        if(self.reconstructionRegularized):
          print('Corresponding reconstruction images:')
          self.net.set_isDeconvolution(True)
          reconstructed = self.net(images)
          self.imshow(torchvision.utils.make_grid(reconstructed))
        print('GroundTruth: ', ' '.join(f'{self.classes[labels[j]]:5s}' for j in range(4)))
        print('Predicted: ', ' '.join(f'{self.classes[predicted[j]]:5s}'
                                      for j in range(4)))

    def evaluate_network(self):
        correct = 0
        total = 0
        self.net.set_isDeconvolution(False)
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                # calculate outputs by running images through the network
                
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    def evaluate_per_class(self):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        self.net.set_isDeconvolution(False)

        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def reconstruct_every_feature(self):
      with torch.no_grad():
        images, labels = next(iter(self.trainloader))
        train_image = images[0].unsqueeze(0).to(self.device) # batch of one image
        images, labels = next(iter(self.testloader))
        test_image = images[0].unsqueeze(0).to(self.device) # batch of one image

        self.net.set_isDeconvolution(True)

        print(f'Train image: ')
        self.imshow(torchvision.utils.make_grid(train_image))

        all_outputs = []  
        for channel in range(6):
          output = self.net(train_image,zero_all_except=channel,layer=1)
          all_outputs.append(output)

        concatenated_outputs = torch.cat(all_outputs, dim=0)  
        print(f'Train image layer 1 deconvolutions: ')
        self.imshow(torchvision.utils.make_grid(concatenated_outputs))
        all_outputs = []  
        for channel in range(3):
          output = self.net(train_image,zero_all_except=channel,layer=2)
          all_outputs.append(output)

        concatenated_outputs = torch.cat(all_outputs, dim=0)  
        print(f'Train image layer 2 deconvolutions: ')
        self.imshow(torchvision.utils.make_grid(concatenated_outputs))

        print(f'Test image: ')
        self.imshow(torchvision.utils.make_grid(test_image))

        all_outputs = []  
        for channel in range(6):
          output = self.net(test_image,zero_all_except=channel,layer=1)
          all_outputs.append(output)

        concatenated_outputs = torch.cat(all_outputs, dim=0)  
        print(f'Test image layer 1 deconvolutions: ')
        self.imshow(torchvision.utils.make_grid(concatenated_outputs))
        all_outputs = []  
        for channel in range(3):
          output = self.net(test_image,zero_all_except=channel,layer=2)
          all_outputs.append(output)

        concatenated_outputs = torch.cat(all_outputs, dim=0)  
        print(f'Test image layer 2 deconvolutions: ')
        self.imshow(torchvision.utils.make_grid(concatenated_outputs)) 


def task1():
    shrinkFactor = 100
    trainer = Trainer(shrinkFactor)
    trainer.display_images()
    trainer.train_network()
    trainer.test_network()
    trainer.evaluate_network()
    trainer.evaluate_per_class()

def task2():
    shrinkFactor = 100
    trainer = Trainer(shrinkFactor,reconstructionRegularized=True)
    trainer.train_network()
    trainer.test_network()
    trainer.evaluate_network()
    trainer.evaluate_per_class()

def task3():
    shrinkFactor = 10
    trainer = Trainer(shrinkFactor,reconstructionRegularized=True)
    trainer.train_network()
    # trainer.test_network()
    # trainer.evaluate_network()
    # trainer.evaluate_per_class()
    trainer.reconstruct_every_feature()



if __name__ == '__main__':
    task1()
    task2()
    task3()
