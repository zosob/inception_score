#calculate inception score with Keras
from math import floor
from numpy import ones, expand_dims, log, mean, std, exp
from keras.applications.inception_v3 import InceptionV3, preprocess_input

#Images need to have shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    #load inception model
    model = InceptionV3()
    #convert from uint8 to float32
    processed = images.astype('float32')
    # preprocess rae images from inception v3 model
    processed = preprocess_input(processed)
    #predict class probability from Images
    yhat = model.predict(processed)
    #enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        #retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        #calculate p(Y)
        p_y = expand_dims(p_yx.mean(axis=0),0)
        #calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over Images
        avg_kl_d = mean(sum_kl_d)
        #undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    #average across scores
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std
