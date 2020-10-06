from pathlib import Path
from data_generator import TrainingGenerator, EvaluationGenerator
from model import model1d
from model import resnet1d
from model import resnet1d_v20201002
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
import datetime

# TODO 
# Set USE_DATE to 1 in order to append a unique date identifier to the saved
# networks and solution files in order to run different training runs at the
# same time. 
# SET USE_DATE to 0 for compliance with official ruleset
USE_DATE = 0
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def get_csv(args):
    if args.run_on_contest_data:
        csv = {
            1: 'contest_train_data_labels.csv',
            3: 'contest_test_data_labels_public.csv'
        }
    else:
        csv = {
            1: 'train_filenames_labels_patient{}_segment_length_10.csv'.format(args.pat),
            2: 'validation_filenames_patient{}_segment_length_10.csv'.format(args.pat),
            3: 'test_filenames_patient{}_segment_length_10.csv'.format(args.pat)
        }

    return Path(args.CSV) / csv[args.mode]


def load_csv(args):
    csv = pd.read_csv(get_csv(args))
    try:
        _ = csv['image']
    except KeyError:
        csv = csv.rename(columns={'image ': 'image'})  # handle inconsistent column naming in csv
        print('renamed column')
    # --- check, if data specified in csv are existent ---
    for i, f in enumerate(csv['image']):
        if not os.path.isfile(f):
            # try path relative to csv as in the sample files
            if os.path.isfile(os.path.join(args.CSV, f)):
                csv.loc[i, 'image'] = os.path.join(args.CSV, f)
            else:
                raise FileNotFoundError('File {} from csv does not exist.'.format(f))

    return csv

def training(args):
    if not os.path.isfile(get_csv(args)):
        raise FileNotFoundError(
            'CSV file not found: {}'.format(get_csv(args)))

    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    df_filenames = load_csv(args)

    if args.run_on_contest_data:
        train_runs = [(df_filenames.loc[df_filenames['image'].str.contains('Pat{}'.format(i))], i) for i in range(1, 4)]
        # train_runs = [(df_filenames, args.pat)]  # TODO uncomment for specifying only one patient
    else:
        train_runs = [(df_filenames, args.pat)]

    for df_filenames, patient in train_runs:
        print('Starting Training Patient {}...'.format(patient))

        dg = TrainingGenerator(df_filenames_csv=df_filenames,
                               segment_length_minutes=10,
                               buffer_length=16000,
                               batch_size=40,
                               n_workers=5,
                               standardize_mode=standardize_mode,
                               shuffle=True)


        # TODO change model for training HERE
        # model = model1d()
        model = resnet1d()
        # model = resnet1d_v20201002()

        model.fit(x=dg,
                  shuffle=False,  # do not change these settings!
                  use_multiprocessing=False,
                  verbose=2,
                  workers=1,
                  epochs=1)  # ORIGINAL
        print('training patient {} done'.format(patient))
        Path(args.model).mkdir(exist_ok=True)
        if USE_DATE:
            # append a unique date identifier to the data
            model_file = 'model_dataset{}_pat{}_subtract{}_{}.h5'.format(args.run_on_contest_data,
                                                                         patient,
                                                                         args.subtract_mean,
                                                                         now)
        else:
            # ORIGINAL
            model_file = 'model_dataset{}_pat{}_subtract{}.h5'.format(args.run_on_contest_data,
                                                                            patient,
                                                                            args.subtract_mean)
        model_archive = os.path.join(args.model, model_file)
        print('Archiving model weights to ' + model_archive)
        model.save_weights(model_archive)

def evaluate(args):
    print('Starting Evaluation...')
    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    df_filenames = load_csv(args)

    if args.run_on_contest_data:
        dfs = [(df_filenames.loc[df_filenames['image'].str.contains('Pat{}'.format(i))], i) for i in range(1, 4)]
        # dfs = [(df_filenames, args.pat)]  # TODO uncomment for specifying only one patient
    else:
        dfs = [(df_filenames, args.pat)]

    solutions = []
    for df_filenames, patient in dfs:
        print('Starting Evaluation for Patient {}...'.format(patient))

        # TODO Specify model that shall be evaluated here - original or via
        # manual mode for specific model files that were created beforehand
        model_file = 'model_dataset{}_pat{}_subtract{}.h5'.format(args.run_on_contest_data,
                                                                  patient,
                                                                  args.subtract_mean) # ORIGINAL
        # model_file = 'model_dataset1_pat1_subtract1_20201001-095007.h5'  # only tmp
        # model_file = 'model_dataset{}_pat{}_subtract{}_20201002-resnet1d_v20201002.h5'.format(args.run_on_contest_data,
        #                                                                                       patient,
        #                                                                                       args.subtract_mean)

        # TODO change model for evaluation HERE
        # model = model1d()
        model = resnet1d()
        # model = resnet1d_v20201002()

        model_path = os.path.join(args.model, model_file)
        print('Using model located at: {}'.format(model_path))

        try:
            model.load_weights(model_path)
        except FileNotFoundError:
            raise FileNotFoundError('Please train model for specified options and patient before evaluating.')

        dg = EvaluationGenerator(df_filenames_csv=df_filenames,
                                 segment_length_minutes=10,
                                 standardize_mode=standardize_mode,
                                 batch_size=40,
                                 class_weights=None)

        probs = model.predict(dg, verbose=0)

        probs = probs.reshape(len(df_filenames), -1).mean(axis=1)
        if args.mode == 1:
            print('Results on training set:')
            metrics = [roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score]
            names = ['roc_auc', 'average_precision', 'precision', 'recall', 'accuracy']

            for m, n in zip(metrics[:2], names[:2]):
                print('{}: {:.4f}'.format(n, m(df_filenames['class'], probs)))
            print('For Threshold = 0.5:')
            for m, n in zip(metrics[2:], names[2:]):
                print('{}: {:.4f}'.format(n, m(df_filenames['class'], probs > 0.5)))

        df_filenames['class'] = probs

        solutions.append(df_filenames)

    s = pd.concat(solutions)
    s['image'] = s['image'].str.split('/').str[-1]
    s['image'] = s['image'].str.split('.').str[0]
    s = s[['image', 'class']]

    Path(args.solutions).mkdir(exist_ok=True)
    if args.run_on_contest_data:
        if USE_DATE:
            # append a unique date identifier to the data
            fn = 'contest_data_solution_shwertt_mode{}_{}.csv'.format(args.mode,
                                                                      now)
        else:
            # ORIGINAL
            fn = 'contest_data_solution_shwertt_mode{}.csv'.format(args.mode)

    else:
        fn = 'solution_shwertt_pat{}_mode{}_subtract{}.csv'.format(args.pat,
                                                                   args.mode,
                                                                   args.subtract_mean)
    s.to_csv(os.path.join(args.solutions, fn), index=False)

    print('Saving solution file to : {}'.format(os.path.join(args.solutions, fn)))
    print('Evaluation done')
