import cv2
import torch
import torch.onnx
import numpy

# model = torch.load("model.pth")
# x = torch.randn(1, 3, 224, 224, requires_grad=True)
# torch_out = model(x)
#
# # Export the model
# torch.onnx.export(model,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                 'output' : {0 : 'batch_size'}})

model = cv2.dnn.readNet("densenet_cr.onnx")

image = cv2.imread(r'D:\Datasets\data_by_lps-220721\test\006.turn_signal__damaged\FXV923_00010.png')

blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))

model.setInput(blob)
outputs = model.forward()
final_outputs = outputs[0]
final_outputs = final_outputs.reshape(6, 1)
label_id = numpy.argmax(final_outputs)

probs = numpy.exp(final_outputs) / numpy.sum(numpy.exp(final_outputs))
print(probs)
final_prob = numpy.max(probs) * 100.
class_names = ['lights_front__clean', 'lights_front__damaged', 'mirror__clean', 'mirror__damaged', 'turn_signal__clean', 'turn_signal__damaged']
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"
print(out_text)