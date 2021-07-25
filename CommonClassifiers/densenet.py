
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim

from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy


from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix


format_text = lambda string: string[4:].replace("-", " ")


def get_data_loaders(data_dir, batch_size):

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    # Change the image so that we can use it in our model
    all_images = datasets.ImageFolder(data_dir, transform)

    # train_images_len = int(len(all_images) * 0.75) # 75% of images will trainable
    # valid_images_len = int((len(all_images) - train_images_len) / 2.0)
    # test_images_len = int(len(all_images) - train_images_len - valid_images_len)

    train_images_len = int(len(all_images) * 0.95) # 75% of images will trainable
    valid_images_len = int((len(all_images) - train_images_len) / 2.0)
    test_images_len = int(len(all_images) - train_images_len - valid_images_len)

    # train_images_len = 0# int(len(all_images) * 0.9) # 75% of images will trainable
    # valid_images_len = 0#int((len(all_images) - train_images_len) / 2.0)
    # test_images_len = int(len(all_images)) #- train_images_len - valid_images_len)

    train_data, val_data, test_data = random_split(all_images, [train_images_len, valid_images_len, test_images_len])

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return (train_loader, val_loader, test_loader), all_images.classes


data_path = r'D:\Datasets\data_by_lps-220721\train'
#data_path = r'/mnt/d/Datasets/Caltech/256_ObjectCategories'

(train_loader, val_loader, test_loader), classes = get_data_loaders(data_path, 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet161(pretrained=True)

print(model)

# Freezing params if we are not interested in training the feature extraction layers
for param in model.parameters():
    param.requires_grad = False


# Now we want to change the number of output classes to our number of classes
# We will take the last "classifier" layer (in other models may named differently
# And assign a linear layer to it

n_inputs = model.classifier.in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier = last_layer

print("Uploading model to GPU/CPU")
model = model.to(device)

# Define our optimizer and loss function
criterion = nn.CrossEntropyLoss()  # This is the loss function
optimizer = optim.Adam(model.classifier.parameters()) # We need to feed the parameters into the optimizer

# Model training
training_history = {'accuracy': [], 'loss': []}
validation_history = {'accuracy': [], 'loss': []}


trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model, device=device, metrics={'accuracy': Accuracy(),
                                                                       'loss': Loss(criterion),
                                                                       'cm': ConfusionMatrix(len(classes))})
#
#
# Create event handler to show our training progress
@trainer.on(Events.ITERATION_COMPLETED)
def log_a_dot(engine):
    print(".", end="")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print()
    print(f"Training results - Epoch: {trainer.state.epoch} Avg accuracy: {accuracy} Loss: {loss}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    torch.save(model, 'model_cp_' + str(trainer.state.epoch).zfill(2) + '.pth')
    print()
    print(f"Validation results - Epoch: {trainer.state.epoch} Avg accuracy: {accuracy} Loss: {loss}")


print("Running Training")
trainer.run(train_loader, max_epochs=4)


# # plotting results
# fig, axs = plt.subplot(2, 2)
# fig.set_figheight(6)
# fig.set_figwidth(14)
# axs[0, 0].plot(training_history["accuracy"])
# axs[0, 0].set_title("Training Accuracy")
# axs[0, 1].plot(validation_history["accuracy"])
# axs[0, 1].set_title("Validation Accuracy")
# axs[1, 0].plot(training_history["loss"])
# axs[1, 0].set_title("Training Loss")
# axs[1, 1].plot(validation_history["loss"])
# axs[1, 1].set_title("Validation Loss")
# fig.show()
#model = torch.load('model_cr.pth')

test_loss = 0.0
class_correct = numpy.zeros((len(classes)))
class_total = numpy.zeros((len(classes)))
model.eval()  # Setting model to evaluation mode

for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = numpy.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else numpy.squeeze(correct_tensor.cpu().numpy())

    #print("target length ", len(target))
    if len(target) == 16:
        for i in range(16):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1


test_loss /= len(test_loader.dataset)

print("Test Loss: {:.6f}\n".format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of {}: {} ({}/{})".format(classes[i],
                                                       100*class_correct[i] / class_total[i],
                                                       numpy.sum(class_correct[i]),
                                                       numpy.sum(class_total[i])))

    else:
        print("Test Accuracy of {}: N/A (since there are no examples)".format(classes[i]))

print("Test Accuracy Overall: {} ({}/{})".format(100*numpy.sum(class_correct) / numpy.sum(class_total[i]),
                                                       numpy.sum(class_correct[i]),
                                                       numpy.sum(class_total[i])))


# torch.save(model, 'model_cr.pth')
#
# x = torch.randn(1, 3, 224, 224, requires_grad=True)
# torch_out = model(x)
# torch.onnx.export(model,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "densenet_cr.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],   # the model's input names
#                   output_names=['output'],  # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
#                                 'output': {0: 'batch_size'}})
