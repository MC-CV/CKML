import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam, lightSelfAttention, lightSelfAttention0, selfAttention
from DataHandler_time import LoadData, negSamp, transToLsts, transpose, prepareGlobalData, sampleLargeGraph, ObtainIIMats, negSamp_aux
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import time

class Recommender:
    def __init__(self, sess, datas):
        self.sess = sess
        self.trnMats, self.iiMats, self.tstInt, self.label, self.tstUsrs, args.intTypes, self.maxTime, self.predir = datas
        self.weights = self._init_weights()
        self.coefficient = eval(args.loss_alphas) 
        prepareGlobalData(self.trnMats, self.label, self.iiMats)
        args.user, args.item = self.trnMats[0].shape
        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'auxLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * 3
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Varaibles Inited')
        mx=-1.0
        for ep in range(stloc, args.epoch):
            test = (ep % 3 == 0)
            a=time.time()
            reses = self.trainEpoch()
            b=time.time()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch() 
                log(self.makePrint('Test', ep, reses, test))
            # if ep % 5 == 0:
            #     self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        # self.saveHistory()

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)

        self.weight_size_list = [args.latdim] + [args.latdim] * args.kg_gnn_layer

        for k in range(args.kg_gnn_layer):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        latdim = args.latdim // (args.n_factors//2)
        self.cap_weight_size_list = [latdim] + [latdim] * args.gnn_layer
        for k in range(args.gnn_layer):
            all_weights['W_gc_cap_%d' % k] = tf.Variable(
                initializer([self.cap_weight_size_list[k], self.cap_weight_size_list[k + 1]]), name='W_gc_cap_%d' % k)
            all_weights['b_gc_cap_%d' % k] = tf.Variable(
                initializer([1, self.cap_weight_size_list[k + 1]]), name='b_gc_cap_%d' % k)

            all_weights['W_bi_cap_%d' % k] = tf.Variable(
                initializer([self.cap_weight_size_list[k], self.cap_weight_size_list[k + 1]]), name='W_bi_cap_%d' % k)
            all_weights['b_bi_cap_%d' % k] = tf.Variable(
                initializer([1, self.cap_weight_size_list[k + 1]]), name='b_bi_cap_%d' % k)

        return all_weights

    def makeTimeEmbed(self):
        divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
        pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
        sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim * 2]) / 4.0
        return timeEmbed

    def capsulenet(self, embs, A, num_src, num_tgt, emb0, layer,num_user,num_item):
        # srcEmbeds, tgtEmbeds, srcNodes, tgtNodes, num_src, num_tgt
        latdim = args.latdim//(args.n_factors//2)
        srcNodes = A.indices[:,1]
        tgtNodes = A.indices[:,0]
        edgeVals = A.values
        n_iterations = args.n_iterations
        n_factors = args.n_factors
        timeEmbed = FC(self.timeEmbed, latdim, reg=True)
        embs1=tf.transpose(embs,[1,0,2]) # [n,n_factors,latdim]
        time=tf.nn.embedding_lookup(timeEmbed, edgeVals)
        time=tf.expand_dims(time,1)
        srcfactorEmbeds = tf.nn.embedding_lookup(embs1,srcNodes)
        tgtfactorEmbeds = tf.nn.embedding_lookup(embs1,tgtNodes)
        if(args.wTime == 'yes'):
            srcfactorEmbeds += time
            tgtfactorEmbeds += time
        srcfactorEmbeds=tf.transpose(srcfactorEmbeds,[1,0,2]) # [n_factors,n,latdim]
        tgtfactorEmbeds=tf.transpose(tgtfactorEmbeds,[1,0,2]) # [n_factors,n,latdim]
        A_values = tf.ones(shape=[n_factors, tf.shape(srcNodes)[0]])
        res = []
        for t in range(n_iterations):
            A_factors = tf.nn.softmax(A_values/args.temp, 0)
            A_iter_values = []
            for k in range(n_factors):
                
                norm2 = tf.math.unsorted_segment_sum(A_factors[k], tgtNodes, num_tgt)
                norm2 = tf.math.pow(norm2, -1)
                norm2 = tf.where(tf.math.is_inf(norm2),tf.zeros_like(norm2),norm2)
                norm2 = tf.nn.embedding_lookup(norm2, tgtNodes)

                factors_emb = tf.reshape(A_factors[k] * norm2, [-1, 1]) * srcfactorEmbeds[k]
                val = tf.math.unsorted_segment_sum(factors_emb, tgtNodes, num_tgt)
                if t == n_iterations - 1:
                    res.append(val)

                
                factors_emb = tf.nn.embedding_lookup(val, tgtNodes)
                src_emb = tf.math.l2_normalize(srcfactorEmbeds[k], axis=1)
                tgt_emb = tf.math.l2_normalize(factors_emb, axis=1)
                
                A_factor_values = tf.reduce_sum(tf.multiply(src_emb, tf.tanh(tgt_emb)), axis=1)
                A_iter_values.append(A_factor_values)

            A_iter_values = tf.stack(A_iter_values, 0)
            A_iter_values = tf.reshape(A_iter_values, [n_factors, -1])
            A_values = tf.add(A_values, A_iter_values)

        res_emb = tf.stack(res, axis=0)
        res_emb = tf.reshape(res_emb, shape=[n_factors, num_tgt, latdim])
        if args.encoder == 'lightgcn':
            lightgcn_emb = res_emb
            res_emb = lightgcn_emb
        elif args.encoder == 'gccf':
            gccf_emb = Activate(res_emb, self.actFunc)
            res_emb = gccf_emb
        elif args.encoder == 'gcn':
            gcn_emb = Activate(tf.matmul(res_emb, self.weights['W_gc_cap_%d' % layer]) + self.weights['b_gc_cap_%d' % layer], self.actFunc)
            res_emb = gcn_emb
        elif args.encoder == 'ngcf':
            gcn_emb = Activate(tf.matmul(res_emb, self.weights['W_gc_cap_%d' % layer]) + self.weights['b_gc_cap_%d' % layer], self.actFunc)
            bi_emb = tf.multiply(emb0, gcn_emb)
            bi_emb = Activate(
                tf.matmul(bi_emb, self.weights['W_bi_cap_%d' % layer]) + self.weights['b_bi_cap_%d' % layer], self.actFunc)
            res_emb = gcn_emb + bi_emb
        else:
            raise 'encoder is invalid!'
        return res_emb

    def gnns(self, embs, A, gnn_layer, encoder='lightgcn'):
        embs_list = [embs]
        ego_emb = embs
        for i in range(gnn_layer):
            cur_embs = embs_list[-1]
            symm_emb = tf.sparse_tensor_dense_matmul(A, cur_embs)
            if encoder == 'lightgcn':
                lightgcn_emb = symm_emb
                res_emb = lightgcn_emb
            elif encoder == 'gccf':
                gccf_emb = Activate(symm_emb, self.actFunc)
                res_emb = gccf_emb
            elif encoder == 'gcn':
                gcn_emb = Activate(tf.matmul(symm_emb, self.weights['W_gc_%d' % i]) + self.weights['b_gc_%d' % i], self.actFunc)
                res_emb = gcn_emb
            elif encoder == 'ngcf':
                gcn_emb = Activate(tf.matmul(symm_emb, self.weights['W_gc_%d' % i]) + self.weights['b_gc_%d' % i], self.actFunc)
                bi_emb = tf.multiply(ego_emb, gcn_emb)
                bi_emb = Activate(
                    tf.matmul(bi_emb, self.weights['W_bi_%d' % i]) + self.weights['b_bi_%d' % i], self.actFunc)
                res_emb = gcn_emb + bi_emb
            else:
                raise 'encoder is invalid!'

            embs_list.append((res_emb+embs_list[-1])/2)
        return tf.add_n(embs_list)/len(embs_list)
    
    def interest_net(self, src, emb): 
        # define the number of different interest layers
        specific_interest_num = args.specific_factors
        shared_interest_num = args.n_factors-specific_interest_num
            
        # build interest-specific layer   
        specific_interest_outputs = []             
        for i in range(specific_interest_num):
            interest_network = FC(emb, args.latdim//(args.n_factors//2), reg=True, useBias=True,
                                activation=self.actFunc, name='level_' + '_interest_specific_' + str(i) + str(src),
                                reuse=True)
            specific_interest_outputs.append(interest_network)
            
        # build interest-shared layer
        shared_interest_outputs = []
        for k in range(shared_interest_num):
            interest_network = FC(emb, args.latdim//(args.n_factors//2), reg=True, useBias=True,
                                activation=self.actFunc, name='level_' + 'interest_shared_' + str(k), reuse=True)
            shared_interest_outputs.append(interest_network)
        
        return tf.stack(specific_interest_outputs + shared_interest_outputs, axis=0)

    def ours(self):
# """---------------------------------------------------------------module of embedding layer---------------------------------------------------------------"""
        num_user = tf.shape(self.all_usrs)[0]
        num_item = tf.shape(self.all_itms)[0]

        all_uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True) 
        all_iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
        uEmbed0 = tf.nn.embedding_lookup(all_uEmbed0, self.all_usrs)
        iEmbed0 = tf.nn.embedding_lookup(all_iEmbed0, self.all_itms)
        self.timeEmbed = tf.Variable(initial_value=self.makeTimeEmbed(), shape=[self.maxTime, args.latdim*2], name='timeEmbed', trainable=True)
        NNs.addReg('timeEmbed', self.timeEmbed)
        
        iEmbed_KG = []
        for i in range(len(self.iiMats)):
            iEmbed_i = self.gnns(iEmbed0, self.iiAdjs[i], args.kg_gnn_layer, encoder=args.encoder)
            iEmbed_KG.append(iEmbed_i)
# """---------------------------------------------------------------module of embedding layer---------------------------------------------------------------"""

# """---------------------------------------------------------------module of kg aux_loss---------------------------------------------------------------"""        
        self.auxloss = 0    
        for gra in range(len(self.iiMats)):

            pckULat = tf.nn.embedding_lookup(iEmbed_KG[gra], self.uids_aux[gra])
            pckILat = tf.nn.embedding_lookup(iEmbed_KG[gra], self.iids_aux[gra])

            predLat = pckULat * pckILat * args.kg_mult

            self.pred_aux = tf.reduce_sum(predLat, axis=-1)

            sampNum = tf.shape(self.iids_aux[gra])[0] // 2
            posPred = tf.slice(self.pred_aux, [0], [sampNum])
            negPred = tf.slice(self.pred_aux, [sampNum], [-1])
            
            self.auxloss += tf.reduce_mean(tf.nn.softplus(-(posPred - negPred)))
# """---------------------------------------------------------------module of kg aux_loss---------------------------------------------------------------"""     
     
# """---------------------------------------------------------------module of coarse-grained classification of interests---------------------------------------------------------------"""             
        ulats = []
        ilats = []
        for beh in range(args.intTypes):  
            cnt=args.n_factors//2
            uEmbed0_beh=tf.stack(tf.split(uEmbed0,[args.latdim//cnt]*cnt,-1),0)
            ulats.append(tf.tile(uEmbed0_beh, [2, 1, 1]))
            ilats.append(self.interest_net(beh,tf.concat(iEmbed_KG,axis=-1)))        
        ulats = [tf.stack(ulats,axis=0)]
        ilats = [tf.stack(ilats,axis=0)]
# """---------------------------------------------------------------module of coarse-grained classification of interests---------------------------------------------------------------""" 

# """---------------------------------------------------------------module of fine-grained classification of interests---------------------------------------------------------------"""
        adjs_beh = []
        for beh in range(args.intTypes):
            R = self.adjs[beh]
            adjs_beh.append(R)
            
        ulats_final = []
        ilats_final = []
        for layer in range(args.gnn_layer):
            ulat_transfer = []
            ilat_transfer = []
            for beh in range(args.intTypes):
                ulat = ulats[-1][beh]  # [n_factors,num_users,latdim]
                ilat = ilats[-1][beh]  # [n_factors,num_items,latdim]                

                lats = tf.concat([ulat, ilat], axis=1)
                
                lats1 = self.capsulenet(lats, adjs_beh[beh], num_user + num_item, num_user + num_item, tf.concat([ulats[0][beh], ilats[0][beh]], axis=1), layer,num_user,num_item)  # [n_factors,num_users+num_items,latdim]
                ulat1, ilat1 = tf.split(lats1, [num_user, num_item], 1)
                
                ulat_transfer.append(ulat1) # [n_behs,n_factors,num_users,latdim]
                ilat_transfer.append(ilat1)

            ulat_transfer_tmp = tf.stack(ulat_transfer, 1)  # [n_factors,n_behs,num_users,latdim]
            ilat_transfer_tmp = tf.stack(ilat_transfer, 1)  # [n_factors,n_behs,num_items,latdim]         

            ulat_transfer_tmp1 = []
            ilat_transfer_tmp1 = []
            specific_interest_num = args.specific_factors
            shared_interest_num = args.n_factors - specific_interest_num
            for spe_num in range(specific_interest_num):
                ulat_transfer_tmp1.append(ulat_transfer_tmp[spe_num])
                ilat_transfer_tmp1.append(ilat_transfer_tmp[spe_num])
            latdim = args.latdim//(args.n_factors//2) 
                
            if args.use_att == 'no':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp1.append(ulat_transfer_tmp[sha_num])
                    ilat_transfer_tmp1.append(ilat_transfer_tmp[sha_num])
            elif args.use_att == 'mean':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp_mean = tf.tile(tf.expand_dims(tf.reduce_mean(ulat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ilat_transfer_tmp_mean = tf.tile(tf.expand_dims(tf.reduce_mean(ilat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ulat_transfer_tmp1.append(ulat_transfer_tmp_mean)
                    ilat_transfer_tmp1.append(ilat_transfer_tmp_mean)
            elif args.use_att == 'sum':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ulat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ilat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ilat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ulat_transfer_tmp1.append(ulat_transfer_tmp_sum)
                    ilat_transfer_tmp1.append(ilat_transfer_tmp_sum)
            elif args.use_att == 'new_selfattention':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp_att = selfAttention(ulat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ilat_transfer_tmp_att = selfAttention(ilat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ulat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ulat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ilat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ilat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ulat_transfer_tmp1.append(ulat_transfer_tmp_att+ulat_transfer_tmp_sum)
                    ilat_transfer_tmp1.append(ilat_transfer_tmp_att+ilat_transfer_tmp_sum)
            elif args.use_att == 'new_light':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp_att = lightSelfAttention(ulat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ilat_transfer_tmp_att = lightSelfAttention(ilat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ulat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ulat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ilat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ilat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ulat_transfer_tmp1.append(ulat_transfer_tmp_att+ulat_transfer_tmp_sum)
                    ilat_transfer_tmp1.append(ilat_transfer_tmp_att+ilat_transfer_tmp_sum)
            elif args.use_att == 'new_light0':
                for sha_num in range(specific_interest_num,specific_interest_num + shared_interest_num):
                    ulat_transfer_tmp_att = lightSelfAttention0(ulat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ilat_transfer_tmp_att = lightSelfAttention0(ilat_transfer_tmp[sha_num],number=args.intTypes,inpDim=latdim,numHeads=args.att_head)
                    ulat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ulat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ilat_transfer_tmp_sum = tf.tile(tf.expand_dims(tf.reduce_sum(ilat_transfer_tmp[sha_num], axis=0),0),[args.intTypes,1,1])
                    ulat_transfer_tmp1.append(ulat_transfer_tmp_att+ulat_transfer_tmp_sum)
                    ilat_transfer_tmp1.append(ilat_transfer_tmp_att+ilat_transfer_tmp_sum)  
            else:
                raise 'use_att is invalid!'

            # specific(generate one) || shared(transfer one)
            ulat_transfer_tmp1 = tf.transpose(ulat_transfer_tmp1, [1,0,2,3])# [n_behs,n_factors,num_user+num_item,latdim]
            ilat_transfer_tmp1 = tf.transpose(ilat_transfer_tmp1, [1,0,2,3])# [n_behs,n_factors,num_user+num_item,latdim]
            ulat_final, ilat_final = ulat_transfer_tmp1, ilat_transfer_tmp1
            ulat_transfer1, ilat_transfer1 = ulat_transfer_tmp1, ilat_transfer_tmp1

            ulats_final.append(ulat_final) # [gnn_layers,n_factors,num_users,latdim]
            ilats_final.append(ilat_final) # [gnn_layers,n_factors,num_items,latdim]
            
            ulats.append(ulats[-1] + ulat_transfer1)  # [gnn_layers,num_behs,n_factors,num_users,latdim]
            ilats.append(ilats[-1] + ilat_transfer1)  # [gnn_layers,num_behs,n_factors,num_items,latdim]

        u_final = tf.add_n(ulats_final)  # [n_factors,num_users,latdim]
        i_final = tf.add_n(ilats_final)  # [n_factors,num_items,latdim]
            
# """---------------------------------------------------------------module of fine-grained classification of interests---------------------------------------------------------------"""

# """---------------------------------------------------------------module of pred---------------------------------------------------------------"""

        self.target_pred_tmp = []
        self.preLoss = 0
        alpha = self.coefficient

        for beh in range(args.intTypes):
            u_pred = tf.transpose(u_final[beh],perm=[1,0,2])
            i_pred = tf.transpose(i_final[beh],perm=[1,0,2])
            pckULat = tf.nn.embedding_lookup(u_pred, self.uids[beh])
            pckILat = tf.nn.embedding_lookup(i_pred, self.iids[beh])
            predLat = pckULat * pckILat * args.mult
            self.pred = tf.reduce_max(tf.reduce_sum(predLat, axis=-1),axis=1)
            if beh == args.intTypes-1:
                self.target_pred = self.pred
            sampNum = tf.shape(self.iids[beh])[0] // 2
            posPred = tf.slice(self.pred, [0], [sampNum])
            negPred = tf.slice(self.pred, [sampNum], [-1])
            self.preLoss += alpha[beh]*tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
        self.regLoss = args.reg * Regularize()
        self.loss = self.preLoss + self.regLoss + self.auxloss*args.kg_loss_alpha

# """---------------------------------------------------------------module of pred---------------------------------------------------------------"""

    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = [] 
        self.iiAdjs = []
        for i in range(args.intTypes):
            self.adjs.append(tf.sparse_placeholder(dtype=tf.int32))

        self.uids_aux = []
        self.iids_aux = []   
        for i in range(len(self.iiMats)):
            self.iiAdjs.append(tf.sparse_placeholder(dtype=tf.float32))
            self.uids_aux.append(tf.placeholder(name='uids_aux_'+str(i), dtype=tf.int32, shape=[None])) 
            self.iids_aux.append(tf.placeholder(name='iids_aux_'+str(i), dtype=tf.int32, shape=[None]))
        self.all_usrs = tf.placeholder(name='all_usrs', dtype=tf.int32, shape=[None])
        self.all_itms = tf.placeholder(name='all_itms', dtype=tf.int32, shape=[None])
        self.usrNum = tf.placeholder(name='usrNum', dtype=tf.int64, shape=[])
        self.itmNum = tf.placeholder(name='itmNum', dtype=tf.int64, shape=[])
        self.uids = []
        self.iids = []
        for i in range(args.intTypes):
            self.uids.append(tf.placeholder(name='uids'+str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.placeholder(name='iids'+str(i), dtype=tf.int32, shape=[None]))

        self.ours()

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batchIds, itmnum, label):
        preSamp = list(np.random.permutation(itmnum))
        temLabel = label[batchIds].toarray()
        batch = len(batchIds)
        temlen = batch * 2 * args.sampNum
        uIntLoc = [None] * temlen
        iIntLoc = [None] * temlen
        cur = 0
        i = -1
        cnt = batch
        if temLabel.sum() == 0:
            return uIntLoc,iIntLoc,False
        while cnt>0:
            i=(i+1)%batch
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            if len(posset)==0:
                continue
            cnt-=1
            negset = negSamp(temLabel[i], preSamp)
            # print(len(posset))
            poslocs = np.random.choice(posset, args.sampNum)
            neglocs = np.random.choice(negset, args.sampNum)
            for j in range(args.sampNum):
                uIntLoc[cur] = uIntLoc[cur + temlen // 2] = batchIds[i]
                iIntLoc[cur] = poslocs[j]
                iIntLoc[cur + temlen // 2] = neglocs[j]
                cur += 1
        return uIntLoc, iIntLoc, True
    
    def sampleTrainBatch_Aux(self, batIds, labelMat, num_item):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(num_item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp_aux(temLabel[i], sampNum, num_item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        tot=0
        
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss, epochAuxLoss = [0] * 3
        num = len(sfIds)

        steps = int(np.ceil(num / args.batch))

        pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(sfIds)
        pckLabel = transpose(transpose(self.label[usrs])[itms])
        usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
        sfIds = list(map(lambda x: usrIdMap[x], sfIds))
        
        num_aux = len(itms)
        sfIds_aux = np.random.permutation(num_aux)[:args.trnNum]
        self.feed_dict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
        for i in range(args.intTypes):
            self.feed_dict[self.adjs[i]] = transToLsts(pckAdjs[i])
        self.iiAdjs_numpy = []
        for i in range(len(pckIiAdjs)):
            idx, data, shape = transToLsts(pckIiAdjs[i], ui=False)
            self.feed_dict[self.iiAdjs[i]] = idx, data, shape
            self.iiAdjs_numpy.append(sp.coo_matrix((data,(np.array(idx.T[0])[0],np.array(idx.T[1])[0])),shape).tocsr())

        for i in range(steps):
            flags = True
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]
            batIds_aux = sfIds_aux[st: ed]
            for beh in range(args.intTypes):
                uLocs, iLocs, flag = self.sampleTrainBatch(batIds, pckAdjs[0].shape[1], pckAdjs[beh])
                flags = flag and flags                      
                self.feed_dict[self.uids[beh]]=uLocs
                self.feed_dict[self.iids[beh]]=iLocs
            if not flags:
                continue             
            itmnum = len(itms)
            for beh in range(len(self.iiMats)):
                uLocs_aux, iLocs_aux = self.sampleTrainBatch_Aux(batIds_aux, self.iiAdjs_numpy[beh], itmnum)
                self.feed_dict[self.uids_aux[beh]] = uLocs_aux
                self.feed_dict[self.iids_aux[beh]] = iLocs_aux   
    
            target = [self.optimizer, self.preLoss, self.regLoss, self.auxloss, self.loss]
            a=time.time()
            res = self.sess.run(target, feed_dict=self.feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            b=time.time()
            tot+=b-a
            preLoss, regLoss, auxLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            epochAuxLoss += auxLoss

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        ret['auxLoss'] = epochAuxLoss / steps
        return ret

    def sampleTestBatch(self, batchIds, label, tstInt):
        batch = len(batchIds)
        temTst = tstInt[batchIds]
        temLabel = label[batchIds].toarray()
        temlen = batch * 100
        uIntLoc = [None] * temlen
        iIntLoc = [None] * temlen
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            rdnNegSet = np.random.permutation(negset)[:99]
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
            tstLocs[i] = locset
            for j in range(100):
                uIntLoc[cur] = batchIds[i]
                iIntLoc[cur] = locset[j]
                cur += 1
        return uIntLoc, iIntLoc, temTst, tstLocs

    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        ids = self.tstUsrs
        num = len(ids)
        tstBat = np.maximum(1, args.batch * args.sampNum // 100)
        steps = int(np.ceil(num / tstBat))

        posItms = self.tstInt[ids]
        pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(ids, pckItms=list(set(posItms)), sampDepth=2, sampNum=args.test_graphSampleN)
        pckLabel = transpose(transpose(self.label[usrs])[itms])
        usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
        itmIdMap = dict(map(lambda x: (itms[x], x), range(len(itms))))
        ids = list(map(lambda x: usrIdMap[x], ids))
        itmMapping = (lambda x: None if (x is None) else itmIdMap[x])
        pckTstInt = np.array(list(map(lambda x: itmMapping(self.tstInt[usrs[x]]), range(len(usrs)))))
        self.feed_dict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
        for i in range(args.intTypes):
            self.feed_dict[self.adjs[i]] = transToLsts(pckAdjs[i])
        for i in range(len(pckIiAdjs)):
            self.feed_dict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i], ui=False)

        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, pckLabel, pckTstInt)
            self.feed_dict[self.uids[-1]] = uLocs
            self.feed_dict[self.iids[-1]] = iLocs
            preds = self.sess.run(self.target_pred, feed_dict=self.feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 100]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
            
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    import os
    import random
    random.seed(42)  # 为python设置随机种子
    np.random.seed(42)  # 为numpy设置随机种子
    tf.set_random_seed(42)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    log_dir = 'log/' + os.path.basename(__file__)
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    import datetime    
    
    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')
    

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text
    
    
    ph_out = PrintHook()    
    ph_out.Start(my_hook_out)
    print('Use gpu id:', args.gpu)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))            
    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log('Start')
    datas = LoadData()
    log('Load Data')
    with tf.Session(config=config) as sess:
        # with tf.device("/gpu:1"):
        recom = Recommender(sess, datas)
        recom.run()
