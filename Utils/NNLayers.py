import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from Params import args

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	# else:
	# 	print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False,trainable=True):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse,trainable=trainable)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros', reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
    embs = tf.transpose(localReps, [1, 0, 2])
    Q = defineRandomNameParam([inpDim, inpDim], reg=True)
    K = defineRandomNameParam([inpDim, inpDim], reg=True)
    V = defineRandomNameParam([inpDim, inpDim], reg=True)
    rspReps = tf.reshape(embs, [-1, inpDim])
    q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim // numHeads])
    k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim // numHeads])
    v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim // numHeads])
    att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) /
                        tf.sqrt(inpDim / numHeads),
                        axis=2)
    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number
    paramId = 'dfltP%d' % getParamId()
    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        # tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
        if args.data == 'yelp':
            rets[i] = tem1 + localReps[i]
        else:
            rets[i] = tem1
    rets1 = tf.stack(rets, 0)
    return rets1


def lightSelfAttention(localReps, number, inpDim, numHeads):
    # localReps shape [number,n,inpDim]
    embs = tf.transpose(localReps, [1, 0, 2])
    Q = defineRandomNameParam(shape=[numHeads, inpDim // numHeads, inpDim // numHeads], reg=False)
    rspReps = tf.reshape(embs, [-1, numHeads, inpDim // numHeads])
    trans = []
    for i in range(numHeads):
        t = tf.slice(rspReps, [0, i, 0], [-1, 1, -1]) @ Q[i]
        trans.append(t)
    tem = tf.concat(trans, axis=-1)

    q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim // numHeads])
    k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim // numHeads])
    v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim // numHeads])
    att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim / numHeads), axis=2)
    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number
    paramId = 'dfltP%d' % getParamId()
    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        # tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
        if args.data == 'yelp':
            rets[i] = tem1 + localReps[i]
        else:
            rets[i] = tem1
    rets1 = tf.stack(rets, 0)
    return rets1

def lightSelfAttention0(localReps, number, inpDim, numHeads):
    # localReps shape [number,n,inpDim]
    embs = tf.transpose(localReps, [1, 0, 2])
    # Q = defineRandomNameParam(shape=[numHeads, inpDim // numHeads, inpDim // numHeads], reg=False)
    rspReps = tf.reshape(embs, [-1, numHeads, inpDim // numHeads])
    trans = []
    for i in range(numHeads):
        t = tf.slice(rspReps, [0, i, 0], [-1, 1, -1])
        trans.append(t)
    tem = tf.concat(trans, axis=-1)

    q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim // numHeads])
    k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim // numHeads])
    v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim // numHeads])
    att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim / numHeads), axis=2)
    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number
    paramId = 'dfltP%d' % getParamId()
    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        # tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
        if args.data == 'yelp':
            rets[i] = tem1 + localReps[i]
        else:
            rets[i] = tem1
    rets1 = tf.stack(rets, 0)
    return rets1

def create_distance_correlation(X1, X2):
    def _create_centered_distance(X):
        '''
            Used to calculate the distance matrix of N samples.
            (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
        '''
        # calculate the pairwise distance of X
        # .... A with the size of [batch_size, embed_size/n_factors]
        # .... D with the size of [batch_size, batch_size]
        # X = tf.math.l2_normalize(XX, axis=1)

        r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

        # # calculate the centered distance of X
        # # .... D with the size of [batch_size, batch_size]
        D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
            + tf.reduce_mean(D)
        return D

    def _create_distance_covariance(D1, D2):
        # calculate distance covariance between D1 and D2
        n_samples = tf.dtypes.cast(tf.shape(D1)[0], tf.float32)
        dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
        # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
        return dcov

    D1 = _create_centered_distance(X1)
    D2 = _create_centered_distance(X2)

    dcov_12 = _create_distance_covariance(D1, D2)
    dcov_11 = _create_distance_covariance(D1, D1)
    dcov_22 = _create_distance_covariance(D2, D2)

    # calculate the distance correlation
    dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
    # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
    return dcor


def create_cor_loss(cor_u_embeddings, cor_i_embeddings, n_factors):
    cor_loss = tf.constant(0.0, tf.float32)

    ui_embeddings = tf.concat([cor_u_embeddings, cor_i_embeddings], axis=0)
    ui_factor_embeddings = tf.split(ui_embeddings, n_factors, 1)

    for i in range(0, n_factors):
        for j in range(i + 1, n_factors):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[j]
            cor_loss += create_distance_correlation(x, y)

    cor_loss /= ((n_factors - 1) * n_factors / 2)

    return cor_loss
