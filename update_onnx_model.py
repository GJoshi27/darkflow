# This code is written for tensorflow v1.x
# ------------------------------------------------
# ONNX Model Editor and Graph Extractor
# License under The MIT License
# Took reference from https://github.com/saurabh-shandilya/onnx-utils
# Written by Gaurav Joshi
# -

import numpy as np
import onnx
from onnx import helper, checker
from onnx import TensorProto
from onnx.tools import update_model_dims
import re
import argparse

def createGraphMemberMap(graph_member_list):
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map

def onnx_update(input_model, output_model, verify):
    """ edits and modifies an onnx model 
    Arguments: 
        input_model: path of input onnx model
        output_model: path of output onnx model    
        verify: set to true if input and output models need to be verified.
    """
    # LOAD MODEL AND PREP MAPS
    model = onnx.load(input_model)
    #variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'image': ['seq', 'batch', 3, -1]})
    #onnx.save(variable_length_model,output_model)
    #exit(0)
	
    graph = model.graph
    if(verify):
        print("input model Errors: ", onnx.checker.check_model(model))
    
    node_map = createGraphMemberMap(graph.node)
    input_map = createGraphMemberMap(graph.input)
    value_info = createGraphMemberMap(graph.value_info)
    output_map = createGraphMemberMap(graph.output)
    initializer_map = createGraphMemberMap(graph.initializer)

    val_info = model.graph.input[0];
    dimsproto = val_info.type.tensor_type.shape

    
    graph.output.remove(output_map['BiasAdd_93:0'])
    graph.output.remove(output_map['BiasAdd_101:0'])
    graph.output.remove(output_map['BiasAdd_109:0'])
    max_detections = 1000
    node = onnx.helper.make_node('CustomYoloPP',['BiasAdd_93:0','BiasAdd_101:0','BiasAdd_109:0'],['output_class_indices','output_boxes','output_confidences'],
                                 numClasses=80,nmsThr=0.6,cnfThr=0.5,image_width=416,image_height=416,max_detections=1000, 
                                 anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])
    node.name = "YoloPP"
    #for nhwc
    output_class_indices = helper.make_tensor_value_info('output_class_indices', TensorProto.FLOAT, [1,max_detections,1,1])
    output_boxes         = helper.make_tensor_value_info('output_boxes', TensorProto.FLOAT, [1,max_detections,4,1,])
    output_confidences   = helper.make_tensor_value_info('output_confidences', TensorProto.FLOAT, [1,max_detections,1,1])

    graph.output.extend([output_class_indices])
    graph.output.extend([output_boxes])
    graph.output.extend([output_confidences])
    graph.node.extend([node])
    onnx.save(model,output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input onnx model")
    parser.add_argument("output", help="output onnx model")
    parser.add_argument('--skipverify', dest='skipverify', action='store_true',
                    help='skip verification of model. Useful if shapes are not known')
    args = parser.parse_args()
    onnx_update(args.input,args.output,not args.skipverify)