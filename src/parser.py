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

    parser.add_argument("--output-path",
                        nargs = "?",
                        default = "./output/chameleon_attention_walk.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 32,
	                help = "Number of dimensions. Default is 128.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
	                help = "Number of gradient descent iterations. Default is 100.")

    parser.add_argument("--window-size",
                        type = int,
                        default = 5,
	                help = "Skip-gram window size. Default is 5.")

    parser.add_argument("--lamb",
                        type = float,
                        default = 10.0,
	                help = "Regularization parameter. Default is 0.1.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Gradient descent. Default is 0.001.")

    parser.add_argument("--geometric",
                        dest="geometric",
                        action="store_true")

    parser.set_defaults(geometric=False)
    
    return parser.parse_args()
