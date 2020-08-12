import tensorflow as tf


def load_graph(frozen_graph_filepath):
    """Load a tensorflow graph from a frozen model (.pb).

    Args:
        frozen_graph_filepath (str): Path to a pretrained (.pb) model.

    Returns:
        graph (tf.Graph): The loaded graph

    """
    with tf.io.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="net",
            op_dict=None,
            producer_op_list=None
        )
        return graph
