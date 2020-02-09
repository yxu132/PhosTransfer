import PhosphoNet
import tensorflow as tf
import numpy as np
import time
import matplotlib
import evaluation
import os
from sets import Set
from shutil import copyfile

matplotlib.use('Agg')
import input_data
import sys
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef

running_mode='test'   # options: 'cv' or 'test'
done_list = Set([])

is_weighted = False
is_negfull = True

WINDOW_SIZE = 9
MAX_EPOCHS = 800

LR_RATE = 1e-3
KEEP_PROBABILITY = 0.75
BATCH_SIZE = 32

hidden_layers = 1
pretrained_model_name = 'Y'

def print_results_diso(output, mean, std):
    output.write("\t Entropy \t Accuracy" + '\n')
    output.write("\t" + str(mean[0]) + '(' + str(std[0]) + ')' + '\t' + str(mean[1]) + '(' + str(std[1]) + ')\n')
    output.write('\n')


def evaluate_diso(y_label, y_conv, output, sess):
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
    y_conv_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
    result = sess.run([evaluation.loss_cross_entropy(y_placeholder, y_conv_placeholder),
                       evaluation.accuracy(y_placeholder, y_conv_placeholder)],
                      feed_dict={y_placeholder: y_label,
                                 y_conv_placeholder: y_conv})
    # print("\t Entropy=%g, Accuracy=%g" % (result[0], result[1]))
    results = [str(res) for res in result]
    if output != None:
        output.write('\t'.join(results) + '\n')
    return result, result[1]


def run_epoch(model, cv, train_output, valid_output, name_str, num_fold=10, graph=None):
    '''Run 10-fold cross-validation'''
    results_train = []
    results_valid = []
    time_train = []
    eval_function = evaluate_diso
    print_function = print_results_diso
    saver = tf.train.Saver(max_to_keep=10)
    best_index = -1
    best_score = -1
    best_entro = 100
    best_entropy = 20
    for i in xrange(10):
        print("Fold " + str(i) + " ................. ")
        fold = cv.nextFold()
        print 'Statistics: ', fold[0].count

        if num_fold < 10:
            if i != num_fold:
                continue
        sess = tf.InteractiveSession(graph=graph)
        sess.run(tf.global_variables_initializer())

        if hidden_layers > 1:
            for j in xrange(hidden_layers-1):
                saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden_layer_'+str(j+1)))
                saver1.restore(sess, '../FINALS/models/'+pretrained_model_name+'.ckpt')


        step = 0
        start_time = time.time()
        best_validation_accuracy = -1
        last_improvement_epoch = -1

        while fold[0].epochs_completed < MAX_EPOCHS:
            batch = fold[0].next_batch(BATCH_SIZE)
            weights = []
            for label in batch[1]:
                if label[1] == 1:
                    weights.append(0.1)
                else:
                    weights.append(0.9)

            _, loss = sess.run([model.train_step, model.loss], feed_dict={model.input: batch[0],
                                                                    model.label: batch[1],
                                                                    model.weights: np.array(weights),
                                                                    model.keep_prob: KEEP_PROBABILITY})
            if step % 200 == 0:
                valid = fold[1].all()
                y_conv = sess.run(model.output,
                              feed_dict={model.input: valid[0],
                                         model.keep_prob: 1.0})
                y = np.array(valid[1])
                result, sum_score = eval_function(y, y_conv, None, sess)
                if sum_score > best_validation_accuracy or \
                        (sum_score == best_validation_accuracy and
                         result[0] < best_entropy):
                # if result[0] < best_entropy:
                    best_validation_accuracy = sum_score
                    best_entropy = result[0]
                    last_improvement_epoch = fold[0].epochs_completed
                    improvement_str = '*'
                    saver.save(sess=sess, save_path='../OUTPUT/checkpoints/'+name_str+'_'+str(i)+'.ckpt')
                else:
                    improvement_str = ''
                print step+1
                print 'Validation: Entropy loss: ', result[0]
                print 'Validation: Accuracy: ', sum_score, ', Best Accuracy: ', best_validation_accuracy, ' ', improvement_str

                if fold[0].epochs_completed - last_improvement_epoch > 50:
                    break

            step = step + 1
        elapse = (time.time() - start_time)
        time_train.append(elapse)

        saver.restore(sess, '../OUTPUT/checkpoints/'+name_str+'_'+str(i)+'.ckpt')

        print 'Training finished. '

        all_y, all_y_conv = [], []
        for j in xrange(9):
            train = cv.get_i(j)
            data = train.all()
            y_conv = sess.run(model.output,
                          feed_dict={model.input: data[0],
                                     model.keep_prob: 1.0})
            all_y.append(np.array(data[1]))
            all_y_conv.append(y_conv)
        y = np.concatenate(all_y, axis=0)
        y_conv = np.concatenate(all_y_conv, axis=0)
        result, _ = eval_function(y, y_conv, train_output, sess)
        results_train.append(result)

        valid = fold[1].all()
        y_conv = sess.run(model.output,
                      feed_dict={model.input: valid[0],
                                 model.keep_prob: 1.0})
        y = np.array(valid[1])
        result, sum_score = eval_function(y, y_conv, valid_output, sess)
        results_valid.append(result)
        if (sum_score > best_score) or (sum_score == best_score and result[0]<best_entro):
            best_score = sum_score
            best_entro = result[0]
            best_index = i
        sess.close()

    # Calculate the mean and std of predicted results
    results_train = np.asarray(results_train)
    results_valid = np.asarray(results_valid)
    time_train = np.asarray(time_train)

    train_mean = np.mean(results_train, axis=0, dtype=np.float32)
    train_std = np.std(results_train, axis=0, dtype=np.float32)
    print_function(train_output, train_mean, train_std)

    valid_mean = np.mean(results_valid, axis=0, dtype=np.float32)
    valid_std = np.std(results_valid, axis=0, dtype=np.float32)
    print_function(valid_output, valid_mean, valid_std)

    time_mean = np.mean(time_train)
    time_std = np.std(time_train)
    print("Time for training: %g(%g)" % (time_mean, time_std))
    train_output.write('\nTime for Training: ' + str(time_mean) + '(' + str(time_std) + ').')
    return best_score, best_entropy, best_index

def run_valid(model, cv, name_str, output_file):
        '''Run 10-fold cross-validation'''
        saver = tf.train.Saver(max_to_keep=10)
        results_valid = []
        for i in xrange(10):
            sess = tf.InteractiveSession()
            print("Fold " + str(i) + " ................. ")
            fold = cv.nextFold()
            saver.restore(sess, '../OUTPUT/checkpoints/'+name_str+'_'+str(i)+'.ckpt')
            print 'Model restored. '
            #

            valid = fold[1].all()
            y_conv = sess.run(model.output,
                              feed_dict={model.input: valid[0],
                                         model.label:valid[1],
                                         model.keep_prob: 1.0})
            # print loss
            y = np.array(valid[1])
            output = y_conv[:, 1]
            predict_disordered(output, y, output_file)
            sess.close()


def run_test(sess, test_model, test):
    output = []
    all_y = []
    for j in xrange(10):
        print 'Part: ' + str(j + 1)
        fold = test.get_i(j)
        data = fold.all()
        print len(data), len(data[0]), len(data[1])

        y_conv = sess.run(test_model.output,
                      feed_dict={test_model.input: data[0],
                                 test_model.keep_prob: 1.0})
 #       print y_conv
        # result, sum_score = evaluate_diso(data[1], y_conv, None, sess)
        output.extend(y_conv[:, 1])
        all_y.extend(data[1])
    return output, all_y



def predict_disordered(outputs, labels, threshold=0.5):
    labels = list(np.array(labels)[:, 1])
    outputs_tmp = []
    labels_tmp = []
    for ind, o in enumerate(outputs):
        if labels[ind] == 1 or labels[ind] == 0:
            outputs_tmp.append(o)
            labels_tmp.append(labels[ind])
    outputs = outputs_tmp
    labels = labels_tmp

    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    output_binary = []
    for ind, l in enumerate(labels):
        o = outputs[ind]
        if o >= threshold:
            output_binary.append(1)
            if l == 1:
                correct += 1
                total += 1
                tp += 1
            elif l == 0:
                total += 1
                fp += 1
        else:
            output_binary.append(0)
            if l == 0:
                correct += 1
                total += 1
                tn += 1
            elif l == 1:
                total += 1
                fn += 1

    mcc = matthews_corrcoef(labels, output_binary)
    print 'MCC: ', mcc
    outputs = np.nan_to_num(outputs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, outputs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('AUC: ' + str(auc))
    positve = tp+fn
    if positve == 0:
        positve += sys.float_info.epsilon
    negative = tn+fp
    if negative == 0:
        negative += sys.float_info.epsilon
    print tp, fp, tn, fn, str(0.5 * (float(tp) / float(positve) + float(tn) / (negative))), mcc, auc
    return 0.5 * (float(tp) / float(positve) + float(tn) / (negative)), mcc, auc

def main(path, args, num_folds=10, model_path=None, save_predict=False):

    global BATCH_SIZE
    BATCH_SIZE = int(args[0])
    global KEEP_PROBABILITY
    KEEP_PROBABILITY = float(args[1])
    global LR_RATE
    LR_RATE = float(args[2])

    appendix = '_phospho'
    if is_weighted:
        appendix += '_wght'
    if not is_negfull:
        appendix += '_sgneg'
    name_str = 'btchs_' + str(BATCH_SIZE) \
               + '_wins_' + str(WINDOW_SIZE) \
               + '_kpprob_' + str(KEEP_PROBABILITY) \
               + '_lr_'+str(LR_RATE) \
               + appendix

    # path='ST'

    if running_mode == "cv":
        print 'Kinase:', path
        print 'Pretrain_model_name:', pretrained_model_name
        print 'Hidden layers:', hidden_layers
        print 'Reading cv data ................. '
        cv = input_data.get_dataset('../DATA/Combined_train/'+path+'/', WINDOW_SIZE, is_negful=is_negfull)
        print("Cross validation begin ................. ")
        g = tf.Graph()
        with g.as_default():
            model = PhosphoNet.PhosphoNet(26, WINDOW_SIZE, 2, hidden_layers=hidden_layers,
                                          is_train=True, LR_ALPHA=LR_RATE,
                                          is_weighted=is_weighted)
            name_str = path.replace('/', '_')+'_'+name_str
            train_output_file = open('../OUTPUT/train_' + name_str + '.txt', 'w')
            train_output_file.write('Entropy\tAccuracy\n')
            valid_output_file = open('../OUTPUT/valid_' + name_str + '.txt', 'w')
            valid_output_file.write('Entropy\tAccuracy\n')

            best_score, best_entropy, best_index = run_epoch(model,
                                               cv,
                                               train_output_file,
                                               valid_output_file,
                                               name_str,
                                               num_fold=num_folds,
                                               graph=g)
            return best_score, best_entropy, best_index


    elif running_mode == "test":
        print 'Kinase:', path
        print 'Pretrain_model_name:', pretrained_model_name
        print 'Hidden layers:', hidden_layers
        print 'Reading test data ................. '
        sites = ['S', 'T']
        if path.startswith('Y/TK') or path == 'Y':
            sites=['Y']
        ids, seqs, test, label = input_data.get_test_dateset(
            dir='../DATA/Combined_test/'+path+'/',
            site=sites,
            win_size=WINDOW_SIZE)
        print len(ids)

        g = tf.Graph()
        with g.as_default():
            print("Independent test begin ................. ")
            test_model = PhosphoNet.PhosphoNet(26, WINDOW_SIZE, 2, hidden_layers=hidden_layers,
                                               is_train=False, LR_ALPHA=LR_RATE,
                                               is_weighted=is_weighted)

            name_str = path.replace('/', '_')
            with tf.Session(graph=g) as sess:
                saver = tf.train.Saver()
                saver.restore(sess, '../FINALS/models/'+model_path+'.ckpt')
                print("Model restored.")
                outputs, labels = run_test(sess, test_model, test)
                if save_predict:
                    np.save('../FINALS/predicts/'+name_str+'.predicts', np.array(outputs))
                    np.save('../FINALS/predicts/'+name_str+'.labels', np.array(labels)[:, 1])

                bacc, mcc, auc = predict_disordered(outputs, labels)
        return bacc, mcc, auc

    elif running_mode== "valid":
        print 'Reading cv data ................. '
        cv = input_data.get_dataset('../DATA/Combined_train/'+path+'/', WINDOW_SIZE, is_negful=is_negfull)
        print("Valid test begin ................. ")
        test_model = PhosphoNet.PhosphoNet(26, WINDOW_SIZE, 2, hidden_layers=hidden_layers,
                                           is_train=False, LR_ALPHA=LR_RATE,
                                           is_weighted=is_weighted)
        name_str = path.replace('/', '_') + '_' + name_str
        test_output_file = open('../OUTPUT/' + 'valid_ind_' + name_str + '.txt', 'w')
        run_valid(test_model, cv, name_str, test_output_file)
        return None, None, None


def sub_procedure(path, output_log):

    orig_hidden_layers = hidden_layers

    comps = path.split('/')

    saved_path_set = []
    dataset_path_set= []
    pretrained_path_set = []
    h_layer_set = []
    for i in xrange(len(comps)):
        dataset_paths, pretrained_paths, saved_paths, h_layers = [], [], [], []
        for j in xrange(i+1):
            saved_paths.append('_'.join(comps[len(comps)-(i+1):len(comps)-(i+1)+(j+1)]))
            dataset_path = comps[0] + '/' + '/'.join(comps[1:len(comps)-(i-j)])
            dataset_paths.append(dataset_path.strip('/'))
            pretrained_paths.append('_'.join(dataset_path.split('/')[:-1]))
            h_layers.append(j+1)

        dataset_path_set.append(dataset_paths)
        pretrained_path_set.append(pretrained_paths)
        saved_path_set.append(saved_paths)
        h_layer_set.append(h_layers)

    for i in xrange(len(dataset_path_set)):
        for j in xrange(len(dataset_path_set[i])):
            if running_mode == 'cv':
                if not os.path.exists('../FINALS/models/'+saved_path_set[i][j]+'.ckpt.index'):
                    global hidden_layers
                    hidden_layers = h_layer_set[i][j]
                    if j > 0:
                        global pretrained_model_name
                        pretrained_model_name = saved_path_set[i][j-1]

                    drop_out_rates = [0.6, 0.75, 0.9]
                    lr_rates = [1e-4, 1e-5]
                    s = dataset_path_set[i][j]
                    if len(s.split('/')) <= 3:
                        lr_rates = [1e-3, 1e-4]
                    batch_sizes = [16, 8, 4]
                    if len(s.split('/')) < 3:
                        batch_sizes = [32, 16, 8]
                    if 'STE' in dataset_path_set[i][j] or 'TKL' in dataset_path_set[i][j]:
                        batch_sizes = [16, 8, 4]
                    if dataset_path_set[i][j] == 'ST':
                        batch_sizes = [128, 64, 32]
                    if 'PIM' in dataset_path_set[i][j] or 'JAK1' in dataset_path_set[i][j] \
                            or 'RSK2' in dataset_path_set[i][j] or 'HIPK4' in dataset_path_set[i][j] \
                            or 'Dyrk1' in dataset_path_set[i][j] or 'PKG1' in dataset_path_set[i][j]\
                            or 'CHK1' in dataset_path_set[i][j] or 'CHK2' in dataset_path_set[i][j] \
                            or 'RAD53' in dataset_path_set[i][j] or 'NDR' in dataset_path_set[i][j] \
                            or 'Met' in dataset_path_set[i][j]:
                        batch_sizes = [8, 4]

                    best_auc = 0.0
                    batch_size_best = 0
                    best_lr_rate = -1
                    drop_out_rate_best = 0
                    best_index = -1
                    best_entropy = 100
                    for lr_rate in lr_rates:
                        for batch_size in batch_sizes:
                            for drop_out_rate in drop_out_rates:
                                if best_index > -1:
                                    avg_auc, entropy, best_index = main(dataset_path_set[i][j], [batch_size, drop_out_rate, lr_rate], num_folds=best_index)
                                else:
                                    avg_auc, entropy, best_index = main(dataset_path_set[i][j], [batch_size, drop_out_rate, lr_rate])
                                if avg_auc > best_auc or (avg_auc == best_auc and entropy < best_entropy):
                                    best_auc = avg_auc
                                    best_entropy = entropy
                                    batch_size_best = batch_size
                                    drop_out_rate_best = drop_out_rate
                                    best_lr_rate = lr_rate

                    print 'RESULTS FOLD(' + str(
                        best_index) + '):',  dataset_path_set[i][j], best_auc, batch_size_best, drop_out_rate_best, best_lr_rate

                    appendix = '_phospho'
                    if is_weighted:
                        appendix += '_wght'
                    if not is_negfull:
                        appendix += '_sgneg'
                    name_str = 'btchs_' + str(batch_size_best) + '_wins_' + str(9) + '_kpprob_' \
                                   + str(drop_out_rate_best) + '_lr_' + str(best_lr_rate) + appendix
                    name_str = dataset_path_set[i][j].replace('/', '_') + '_' + name_str

                    copyfile('../OUTPUT/checkpoints/' + name_str + '_' + str(best_index) + '.ckpt.index',
                             '../FINALS/models/' + saved_path_set[i][j] + '.ckpt.index')
                    copyfile('../OUTPUT/checkpoints/' + name_str + '_' + str(best_index) + '.ckpt.meta',
                             '../FINALS/models/' + saved_path_set[i][j] + '.ckpt.meta')
                    copyfile('../OUTPUT/checkpoints/' + name_str + '_' + str(best_index) + '.ckpt.data-00000-of-00001',
                             '../FINALS/models/' + saved_path_set[i][j] + '.ckpt.data-00000-of-00001')

                    output_log.write('TRAINING: '+
                        saved_path_set[i][j] + '_f' + str(best_index) + ':' + '\t' + str(best_auc) + '\t' + str(
                        batch_size_best) + '\t' + str(drop_out_rate_best) + '\t' + str(best_lr_rate) + '\n')
                    output_log.flush()

            elif running_mode == 'test':
                global hidden_layers
                hidden_layers = h_layer_set[i][j]
                if j > 0:
                    global pretrained_model_name
                    pretrained_model_name = saved_path_set[i][j - 1]
                if not dataset_path_set[i][j]+'-'+ saved_path_set[i][j]+'-'+str(hidden_layers) in done_list:
                   bacc, mcc, auc = main(dataset_path_set[i][j], [0, 0, 0], model_path=saved_path_set[i][j])
                   output_log.write('TESTING: ' +dataset_path_set[i][j]+'\t'+ saved_path_set[i][j] + '\t'
                                      + str(bacc) + '\t' + str(mcc) + '\t'
                                      + str(auc) + '\n')
                   output_log.flush()
                   global done_list
                   done_list.add(dataset_path_set[i][j]+'-'+ saved_path_set[i][j]+'-'+str(hidden_layers))

    output_log.write('\n')
    output_log.flush()

    global hidden_layers
    hidden_layers = orig_hidden_layers

def dir_recursive(dir, output_log):
    for path in os.listdir('../DATA/Combined_train/'+dir):
        p = dir+'/'+path

        if 'TKL' in path or 'PDGFRB' in path or 'Met' in path or 'ST' in path:
            continue
        if os.path.exists('../DATA/Combined_train/'+dir+'/'+path+'/ID.txt'):
            if len(p.strip('/').split('/')) > 5: # \
                    # or os.path.exists('../FINALS/models/' + p.strip('/').replace('/', '_') + '.ckpt.index'):
                print 'skip:', p.strip('/')
            else:
                sub_procedure((dir+'/'+path).strip('/'), output_log)

        if os.path.isdir('../DATA/Combined_train/'+dir+'/'+path):
            global hidden_layers
            hidden_layers = hidden_layers+1
            global pretrained_model_name
            pretrained_model_name = p.strip('/').replace('/', '_')
            dir_recursive(dir+'/'+path, output_log)
            global hidden_layers
            hidden_layers = hidden_layers-1
            global pretrained_model_name
            d = dir
            if dir == '':
                d = 'Y'
            pretrained_model_name = d.strip('/').replace('/', '_')
        else:
            continue

if __name__ == '__main__':
    output_log = open('../FINALS/model_config3.txt', 'w')
    dir_recursive('', output_log)

