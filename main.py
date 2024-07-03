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
    def __init__(self):
        super().__init__()

        # Custom attributes
        self.enable_deconvolutional_model = False
        self.zero_all_except = -1
        self.layer = -1

        # Original layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Deconvolution layers
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv2T = nn.ConvTranspose2d(16, 6, 5, stride=1, padding=0, output_padding=0)
        self.conv1T = nn.ConvTranspose2d(6, 3, 5, stride=1, padding=0, output_padding=0) 

    def forward(self, x):
        z, indices1 = self.pool(F.relu(self.conv1(x)))

        if self.zero_all_except != -1 and self.layer == 1:
            for i in range(z.size(1)):
                if i != self.zero_all_except:
                    z[0, i, :, :] = 0

        z, indices2 = self.pool(F.relu(self.conv2(z)))

        if self.zero_all_except != -1 and self.layer == 2:
            for i in range(z.size(1)):
                if i != self.zero_all_except:
                    z[0, i, :, :] = 0

        # Original behavior
        if not self.enable_deconvolutional_model:
            y = torch.flatten(z, 1)  # flatten all dimensions except batch
            y = F.relu(self.fc1(y))
            y = F.relu(self.fc2(y))
            y = self.fc3(y)
            return y
        # Modified behavior
        else:
            y = torch.flatten(z, 1)  # flatten all dimensions except batch
            y = F.relu(self.fc1(y))
            y = F.relu(self.fc2(y))
            y = self.fc3(y)

            x = self.conv2T(F.relu(self.unpool(z, indices2)))
            x = self.conv1T(F.relu(self.unpool(x, indices1)))
            return y, x
              

class CustomLoss(nn.Module):
    def __init__(self, lambda_rec=1.0):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, y_pred, y_true, x_recon, x_orig):
        batch_size = x_orig.size(0)
        channel_size = x_orig.size(1)

        # custom loss function
        ce_term = self.ce_loss(y_pred, y_true)
        rec_term = self.mse_loss(x_recon, x_orig) / (batch_size * channel_size)
        total_loss = ce_term + self.lambda_rec * rec_term
        return total_loss


class Trainer:
    def __init__(self, shrink_factor=1, reconstruction_regularized=False):
        self.shrink_factor = shrink_factor
        self.reconstruction_regularized = reconstruction_regularized

        # Device
        self._set_device()

        # Dataset and Dataloaders
        self._prepare_dataloaders()
      
        # Model
        self.net = Net().to(self.device)

        # Loss and Optimizer
        if not self.reconstruction_regularized:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.net.enable_deconvolutional_model = True
            self.criterion = CustomLoss(lambda_rec=0.1)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def _set_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def _save_model(self):
        PATH = './cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)

    def _load_model(self):
        # Save original state of deconvolutional model
        net_original_state = self.net.enable_deconvolutional_model

        # Load model
        self.net = Net().to(self.device)
        PATH = './cifar_net.pth'
        self.net.load_state_dict(torch.load(PATH))

        # Restore deconvolutional model state
        self.net.enable_deconvolutional_model = net_original_state

    def _prepare_dataloaders(self):
        # Required to enable download of CIFAR-10 dataset
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.batch_size = 4

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.transform)
        if self.shrink_factor != 1:
            self.trainset = Subset(self.trainset,
                                   torch.randperm(len(self.trainset))[:int(len(self.trainset) / self.shrink_factor)])
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        if self.shrink_factor != 1:
            self.testset = Subset(self.testset,
                                  torch.randperm(len(self.testset))[:int(len(self.testset) / self.shrink_factor)])
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

                # forward + backward + optimize
                outputs = self.net(inputs)

                # Original behavior
                if not self.reconstruction_regularized:
                    loss = self.criterion(outputs, labels)
                # Modified behavior
                else:
                    outputs_y, outputs_x = outputs
                    loss = self.criterion(outputs_y, labels, outputs_x, inputs)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                # Original behavior
                if self.shrink_factor == 1:
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0
                # Modified behavior
                else:
                    if i % (2000/self.shrink_factor) == int(1999/self.shrink_factor):
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (int(2000/self.shrink_factor)):.3f}')
                        running_loss = 0.0

        print('Finished Training')

        # Save model
        self._save_model()

    def test_network(self):
        # Disable deconvolutional model (if enabled)
        net_original_state = self.net.enable_deconvolutional_model
        self.net.enable_deconvolutional_model = False

        dataiter = iter(self.testloader)
        images, labels = next(dataiter)

        # print images
        self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{self.classes[labels[j]]:5s}' for j in range(4)))

        self._load_model()

        images = images.to(self.device)
        outputs = self.net(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{self.classes[predicted[j]]:5s}'
                                      for j in range(4)))

        # Restore deconvolutional model state
        self.net.enable_deconvolutional_model = net_original_state

    def evaluate_network(self):
        self._load_model()

        # Disable deconvolutional model (if enabled)
        net_original_state = self.net.enable_deconvolutional_model
        self.net.enable_deconvolutional_model = False

        correct = 0
        total = 0
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

        if self.shrink_factor == 1:
            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        else:
            print(f'Accuracy of the network on the test images: {100 * correct // total} %')

        # Restore deconvolutional model state
        self.net.enable_deconvolutional_model = net_original_state

    def evaluate_per_class(self):
        self._load_model()

        # Disable deconvolutional model (if enabled)
        net_original_state = self.net.enable_deconvolutional_model
        self.net.enable_deconvolutional_model = False

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

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

        # Restore deconvolutional model state
        self.net.enable_deconvolutional_model = net_original_state

    def display_reconstructed_images(self,):
        if not self.reconstruction_regularized:
            raise Exception('This method is only available when the network is reconstruction regularized')

        self._load_model()

        dataiter = iter(self.testloader)
        images, labels = next(dataiter)

        images = images.to(self.device)
        _, outputs = self.net(images)

        # Display reconstruction images
        num_of_examples = 3
        for i in range(num_of_examples):
            result = torch.cat((images[i: i+1], outputs[i: i+1]), dim=0)
            plt.title(label=f'Image {i + 1} and its reconstruction')
            self.imshow(torchvision.utils.make_grid(result.cpu()))

    def reconstruct_every_feature(self):
        if not self.reconstruction_regularized:
            raise Exception('This method is only available when the network is reconstruction regularized')

        self._load_model()

        with torch.no_grad():
            images, labels = next(iter(self.trainloader))
            train_image = images[0].unsqueeze(0).to(self.device)  # batch of one image
            images, labels = next(iter(self.testloader))
            test_image = images[0].unsqueeze(0).to(self.device)  # batch of one image

            # Plot 1
            plt.title(label='Train image')
            self.imshow(torchvision.utils.make_grid(train_image.cpu()))

            # Plot 2
            all_outputs = list()
            for channel in range(6):
                self.net.zero_all_except = channel
                self.net.layer = 1
                _, output = self.net(train_image)
                all_outputs.append(output)
            concatenated_outputs = torch.cat(all_outputs, dim=0)
            plt.title(label='Train image layer 1 deconvolutions')
            self.imshow(torchvision.utils.make_grid(concatenated_outputs.cpu()))

            # Plot 3
            all_outputs = list()
            for channel in range(3):
                self.net.zero_all_except = channel
                self.net.layer = 2
                _, output = self.net(train_image)
                all_outputs.append(output)
            concatenated_outputs = torch.cat(all_outputs, dim=0)
            plt.title(label='Train image layer 2 deconvolutions')
            self.imshow(torchvision.utils.make_grid(concatenated_outputs.cpu()))

            # Plot 4
            plt.title(label='Test image')
            self.imshow(torchvision.utils.make_grid(test_image.cpu()))

            # Plot 5
            all_outputs = list()
            for channel in range(6):
                self.net.zero_all_except = channel
                self.net.layer = 1
                _, output = self.net(test_image)
                all_outputs.append(output)
            concatenated_outputs = torch.cat(all_outputs, dim=0)
            plt.title(label='Test image layer 1 deconvolutions')
            self.imshow(torchvision.utils.make_grid(concatenated_outputs.cpu()))

            # Plot 6
            all_outputs = list()
            for channel in range(3):
                self.net.zero_all_except = channel
                self.net.layer = 2
                _, output = self.net(test_image)
                all_outputs.append(output)
            concatenated_outputs = torch.cat(all_outputs, dim=0)
            plt.title(label='Test image layer 2 deconvolutions')
            self.imshow(torchvision.utils.make_grid(concatenated_outputs.cpu()))

            # Restore zero_all_except and layer attributes
            self.net.zero_all_except = -1
            self.net.layer = -1


def task1():
    shrink_factor = 1
    reconstruction_regularized = False

    trainer = Trainer(shrink_factor=shrink_factor, reconstruction_regularized=reconstruction_regularized)
    trainer.display_images()
    trainer.train_network()
    trainer.test_network()
    trainer.evaluate_network()
    trainer.evaluate_per_class()


def task2():
    shrink_factor = 1
    reconstruction_regularized = True

    trainer = Trainer(shrink_factor=shrink_factor, reconstruction_regularized=reconstruction_regularized)
    trainer.train_network()
    trainer.evaluate_network()
    trainer.evaluate_per_class()
    trainer.display_reconstructed_images()


def task3():
    shrink_factor = 1
    reconstruction_regularized = True

    trainer = Trainer(shrink_factor=shrink_factor, reconstruction_regularized=reconstruction_regularized)
    trainer.train_network()
    trainer.reconstruct_every_feature()


if __name__ == '__main__':
    task1()
    task2()
    task3()
