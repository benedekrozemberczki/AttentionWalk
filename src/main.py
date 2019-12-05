"""Run Attention Walk."""

from param_parser import parameter_parser
from utils import tab_printer
from attentionwalk import AttentionWalkTrainer

def main():
    """
    Parsing command lines, creating target matrix.
    Fitting an Attention Walker and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    model = AttentionWalkTrainer(args)
    model.fit()
    model.save_model()

if __name__ == "__main__":
    main()
