import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import selective_registration_header_lib
from tensorflow.python.tools import optimize_for_inference_lib


def WriteGraphFiles(graphs, dir):
    fnames = []
    for i, graph in enumerate(graphs):
        fname = os.path.join(dir, 'graph%s.pb' % i)
        with gfile.GFile(fname, 'wb') as f:
            f.write(graph.SerializeToString())
        fnames.append(fname)
    return fnames


def save_model(sess, saver, train_dir, global_step):
    checkpoint_train_path = os.path.join(train_dir, "saved_checkpoint")
    checkpoint = saver.save(
        sess,
        checkpoint_train_path,
        global_step=global_step,
        latest_filename="checkpoint_state")

    return checkpoint


def deploy_model(sess, output_graph_name, output_node_names, input_node_names, model_dir, checkpoint):
    checkpoint_meta_graph_file = str(checkpoint) + '.meta'
    input_graph_name = "graph.pb"
    graph_io.write_graph(sess.graph, model_dir, input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # Definitions
    input_graph_path = os.path.join(model_dir, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    restore_op_name = ""
    filename_tensor_name = ""
    output_graph_path = os.path.join(model_dir, output_graph_name)
    clear_devices = False
    input_meta_graph = checkpoint_meta_graph_file

    # Freeze graph
    freeze_graph.freeze_graph(
        input_graph_path, input_saver_def_path, input_binary, checkpoint,
        output_node_names, restore_op_name, filename_tensor_name,
        output_graph_path, clear_devices, "", "", input_meta_graph)

    # Optimize for inference
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        sess.graph_def, input_node_names, [output_node_names], dtypes.float32.as_datatype_enum)

    # Selective registration header
    graphs = []  # supports writing multiple graphs
    graphs.append(output_graph_def)
    fnames = WriteGraphFiles(graphs, model_dir)  # Optimized graph definition
    default_ops = 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'
    fnames2 = []
    fnames2.append(os.path.join(model_dir, output_graph_name))
    header = selective_registration_header_lib.get_header(fnames2, 'rawproto', default_ops)

    # Write header file
    with open(model_dir + "/ops_to_register.h", "w") as text_file:
        print(header, file=text_file)

    return output_graph_def

def load_graph(frozen_graph_filename):
    # First we need to load the protobuf file from the disk and parse it to retrieve the
    # Unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph
