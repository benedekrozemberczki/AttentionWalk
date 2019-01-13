from attentionwalk import AttentionWalkTrainer
from parser import parameter_parser
from utils import read_graph, tab_printer

def main():
    """
    Parsing command lines, creating target matrix, fitting an Attention Walker and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    model = AttentionWalkTrainer(args)
    model.fit()
    model.save_embedding()
    model.save_attention()

if __name__ =="__main__":
    main()
