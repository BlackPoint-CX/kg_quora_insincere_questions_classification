from argparse import ArgumentParser


def build_argparser(argument_parser):
    ap.register('type', 'bool', lambda v: v.lower() == 'true')
    ap.add_argument('--vocab_size', type=int, default=3000, help='Size of vocabulary.')
    ap.add_argument('--embedding_dim', type=int, default=300, help='Di')
    ap.add_argument('--embedding_trainable', type='bool', nargs='?', const=True, help='')
    ap.add_argument('--embedding_file',type=str, default='')

def build_hparams():
    pass


def main():
    pass


if __name__ == '__main__':
    ap = ArgumentParser()
    main()
