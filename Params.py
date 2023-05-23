import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--latdim', default=16, type=int, help='embedding size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--kg_gnn_layer', default=1, type=int, help='number of kg_gnn layers')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')
    parser.add_argument('--kg_mult', default=1, type=float, help='multiplier for the result')
    parser.add_argument('--slot', default=5, type=int, help='length of time slots')
    parser.add_argument('--graphSampleN', default=25000, type=int, help='use 25000 for training and 200000 for testing, empirically')
    parser.add_argument('--test_graphSampleN', default=25000, type=int, help='test sampling')
    parser.add_argument('--divSize', default=50, type=int, help='div size for smallTestEpoch')
    parser.add_argument('--n_factors', default=2, type=int, help='the number of factors')
    parser.add_argument('--specific_factors', default=1, type=int, )
    parser.add_argument('--n_iterations', default=2, type=int, help='iteration of capsulenet')
    parser.add_argument('--encoder', default='lightgcn', type=str, help='type of encoder')
    parser.add_argument('--gpu', type=int, default=6 ,help='gpu id')
    parser.add_argument('--norm',type=str,default='left', help='norm type')
    parser.add_argument('--use_att',type=str,default='new_selfattention', help='use lightselfattention')
    parser.add_argument('--kg_loss_alpha',type=float,default=1, help='the alpha of kg loss')
    parser.add_argument('--mode',type=str,default='4', help='mode')
    parser.add_argument('--temp',type=float,default=1.0, help='the temperature of capsulenet')
    parser.add_argument('--wTime',type=str,default='yes', help='use or not time embs')   
    parser.add_argument('--loss_alphas',nargs='?', default='[0,0,0,1]', help='alpha of preloss, [1,0,0.5,1] for yelp, [1,1,1] for m10, [0,0,0,1] for retail.') 
    return parser.parse_args()


args = parse_args()
print(args)
# args.user = 147894
# args.item = 99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734       

if args.data == 'yelp':
    args.use_att = 'new_selfattention'
    args.gnn_layer = 3
    args.graphSampleN = 25000
    args.loss_alphas = '[1,0,0.2,1]'
    args.mult = 1
    args.encoder = 'lightgcn'
    args.kg_mult=100
elif args.data == 'retail':
    args.use_att = 'new_selfattention'
    args.gnn_layer = 4
    args.graphSampleN = 15000
    args.test_graphSampleN = 30000
    args.loss_alphas = '[1,0,0.4,1]'
    args.mult = 1
    args.encoder = 'gccf'
    args.temp = 20
    args.kg_mult=100
elif args.data == 'tmall':
    args.use_att = 'new_selfattention'
    args.gnn_layer = 4
    args.graphSampleN = 25000
    args.loss_alphas = '[1,0,0.4,1]'
    args.mult = 1
    args.encoder = 'gccf'
    args.temp = 10
    args.kg_mult=100
else:
    raise 'dataset is invalid!'

args.decay_step = args.trnNum // args.batch
