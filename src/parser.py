import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Chameleons dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run Attention Walk.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/chameleon_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--embedding-path",
                        nargs = "?",
                        default = "./output/chameleon_AW_embedding.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--attention-path",
                        nargs = "?",
                        default = "./output/chameleon_AW_attention.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 128,
	                help = "Number of dimensions. Default is 128.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 200,
	                help = "Number of gradient descent iterations. Default is 200.")

    parser.add_argument("--window-size",
                        type = int,
                        default = 5,
	                help = "Skip-gram window size. Default is 5.")

    parser.add_argument("--beta",
                        type = float,
                        default = 0.1,
	                help = "Regularization parameter. Default is 0.1.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Gradient descent learning rate. Default is 0.01.")
    
    return parser.parse_args()
