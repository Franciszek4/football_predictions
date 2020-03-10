from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
import numpy as np
import sys
import glob
from sklearn.ensemble import RandomForestClassifier

float_formatter = "{:.2f}".format
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float_kind': float_formatter})


def get_data_from_dir(path, columns, teams=None):

    data, labels = [], []

    for file in glob.glob(path):

        print('file: %s' % file)
        dataset = np.genfromtxt(file, usecols=columns, skip_header=1, delimiter=',', dtype=str)

        output_scheme = {'H': 0, 'D': 1, 'A': 2}

        for idx, el in enumerate(dataset[:, 4]):
            dataset[idx, 4] = output_scheme[dataset[idx, 4]]

        if len(data) == 0:
            data = dataset.copy()
        else:
            data = np.concatenate((data, dataset), 0)

    team_names = data[:, 0]

    team_names = np.unique(team_names)

    if teams == None:
        team_labels = {el: i for i, el in enumerate(team_names)}
    else:
        keys = list(teams.keys())
        temp = np.isin(team_names, keys)
        if False in temp:
            args = np.argwhere(temp == False)
            for idx, el in enumerate(args):
                num_el = el[0]
                teams.update({team_names[num_el]: len(keys) + idx})
        team_labels = teams

    data[:, 0] = [team_labels[el] for el in data[:, 0]]
    data[:, 1] = [team_labels[el] for el in data[:, 1]]

    print(data.shape)

    data = data.astype(int)

    return data, team_labels


def extract_dataset_and_labels(data, teams):

    away_result_decoder = {0: 2, 1: 1, 2: 0}

    first_pass = True

    for idx in np.arange(0, data.shape[0]):

        home_team, away_team, home_goals, away_goals, result = data[idx, :5]
        home_shots, away_shots, home_shots_ot, away_shots_ot, home_corners, away_corners = data[idx, 5:]

        home_team_data = np.array([idx,
                                   home_team,
                                   away_team,
                                   home_goals,
                                   home_shots,
                                   home_shots_ot,
                                   home_corners,
                                   away_goals,
                                   away_shots,
                                   away_shots_ot,
                                   away_corners,
                                   result])

        away_team_data = np.array([idx,
                                   away_team,
                                   home_team,
                                   away_goals,
                                   away_shots,
                                   away_shots_ot,
                                   away_corners,
                                   home_goals,
                                   home_shots,
                                   home_shots_ot,
                                   home_corners,
                                   away_result_decoder[result]])

        if first_pass:
            all_matches_doubled = np.vstack((home_team_data, away_team_data))
            first_pass = False
        else:
            all_matches_doubled = np.vstack((all_matches_doubled, home_team_data, away_team_data))

    first_row = True

    print(teams)

    for idx in np.arange(0, all_matches_doubled.shape[0], 2):

        home_team = all_matches_doubled[idx, 1]
        args_home_team = np.where(all_matches_doubled[0:idx, 1] == home_team)[0]

        away_team = all_matches_doubled[idx + 1, 1]
        args_away_team = np.where(all_matches_doubled[0:idx + 1, 1] == away_team)[0]

        h2h_mat = np.zeros((len(teams), len(teams)), dtype=float)
        h2h_counter = np.ones(len(teams))
        avg_points = np.zeros(len(teams))

        for i in np.arange(0, idx, 2):

            h_team = all_matches_doubled[i, 1]
            a_team = all_matches_doubled[i, 2]

            if all_matches_doubled[i, 11] == 0:
                h2h_mat[h_team, a_team] += 3
                h2h_mat[a_team, h_team] -= 3
                avg_points[h_team] += 3
            elif all_matches_doubled[i, 11] == 2:
                h2h_mat[h_team, a_team] -= 3
                h2h_mat[a_team, h_team] += 3
                avg_points[a_team] += 3
            else:
                avg_points[h_team] += 1
                avg_points[a_team] += 1

            h2h_counter[h_team] += 1
            h2h_counter[a_team] += 1

        max_val = h2h_mat.max()

        if max_val == 0:
            max_val = 1

        h2h_mat = h2h_mat / max_val

        avg_points = np.true_divide(avg_points, h2h_counter)

        if args_home_team.shape[0] > 4 and args_away_team.shape[0] > 4:

            last_five_matches_ht = all_matches_doubled[args_home_team[-5:], :]
            wins_ht = np.where(last_five_matches_ht[:, -1] == 0)[0].shape[0]
            draws_ht = np.where(last_five_matches_ht[:, -1] == 1)[0].shape[0]
            defeats_ht = np.where(last_five_matches_ht[:, -1] == 2)[0].shape[0]

            goals_ht = np.sum(last_five_matches_ht[:, 3])
            shots_ht = np.sum(last_five_matches_ht[:, 4])
            shots_ot_ht = np.sum(last_five_matches_ht[:, 5])
            corners_ht = np.sum(last_five_matches_ht[:, 6])

            opponent_goals_ht = np.sum(last_five_matches_ht[:, 7])
            opponent_shots_ht = np.sum(last_five_matches_ht[:, 8])
            opponent_shots_ot_ht = np.sum(last_five_matches_ht[:, 9])
            opponent_corners_ht = np.sum(last_five_matches_ht[:, 10])

            last_five_matches_at = all_matches_doubled[args_away_team[-5:], :]
            wins_at = np.where(last_five_matches_at[:, -1] == 0)[0].shape[0]
            draws_at = np.where(last_five_matches_at[:, -1] == 1)[0].shape[0]
            defeats_at = np.where(last_five_matches_at[:, -1] == 2)[0].shape[0]

            goals_at = np.sum(last_five_matches_at[:, 3])
            shots_at = np.sum(last_five_matches_at[:, 4])
            shots_ot_at = np.sum(last_five_matches_at[:, 5])
            corners_at = np.sum(last_five_matches_at[:, 6])

            opponent_goals_at = np.sum(last_five_matches_at[:, 7])
            opponent_shots_at = np.sum(last_five_matches_at[:, 8])
            opponent_shots_ot_at = np.sum(last_five_matches_at[:, 9])
            opponent_corners_at = np.sum(last_five_matches_at[:, 10])

            row = np.array([
                            wins_ht,
                            draws_ht,
                            defeats_ht,
                            goals_ht,
                            # shots_ht,
                            shots_ot_ht,
                            opponent_goals_ht,
                            # opponent_shots_ht,
                            opponent_shots_ot_ht,
                            avg_points[home_team],
                            wins_at,
                            draws_at,
                            defeats_at,
                            goals_at,
                            # shots_at,
                            shots_ot_at,
                            opponent_goals_at,
                            # opponent_shots_at,
                            opponent_shots_ot_at,
                            # h2h_counter[home_team],
                            # h2h_counter[away_team],
                            avg_points[away_team],
                            h2h_mat[home_team, away_team],
                            ])

            label = all_matches_doubled[idx, -1]

            if first_row:
                dataset = row
                labels = label
                first_row = False
            else:
                dataset = np.vstack((dataset, row))
                labels = np.vstack((labels, label))

    return dataset, labels


raw_data, teams = get_data_from_dir('data\\premier_league\\*.csv', (2, 3, 4, 5, 6, 11, 12, 13, 14, 17, 18))
data, labels = extract_dataset_and_labels(raw_data, teams)

print(data)
data = data.astype(float)

# data = data / data.max(axis=0)

print('data.shape = %s' % str(data.shape))

BORDER0 = 0
BORDER1 = 3000
BORDER2 = 3300
train_set = data[BORDER0:BORDER1, :]
eval_set = data[BORDER1:BORDER2, :]
test_set = data[BORDER2:, :]

train_labels = labels[BORDER0:BORDER1]
eval_labels = labels[BORDER1:BORDER2]
test_labels = labels[BORDER2:]

# RF #############################################################################################################

rf = RandomForestClassifier(n_estimators=2000)
rf.fit(train_set, train_labels.ravel())

y = rf.predict(test_set)
print(y)

counter = 0
counter_1 = 0
counter_2 = 0
counter_3 = 0

true_counter_1 = 0
true_counter_2 = 0
true_counter_3 = 0

false_counter_1 = 0
false_counter_2 = 0
false_counter_3 = 0

for i in np.arange(test_labels.shape[0]):
    # print(y[i], test_labels[i][0])
    if test_labels[i] == 0:
        true_counter_1 += 1
    elif test_labels[i] == 1:
        true_counter_2 += 1
    elif test_labels[i] == 2:
        true_counter_3 += 1

    if y[i] == test_labels[i]:
        counter += 1
        if y[i] == 0:
            counter_1 += 1
        elif y[i] == 1:
            counter_2 += 1
        elif y[i] == 2:
            counter_3 += 1
    elif y[i] == 0:
        false_counter_1 += 1
    elif y[i] == 1:
        false_counter_2 += 1
    elif y[i] == 2:
        false_counter_3 += 1

print('matched \'1\': %d,     matched \'2\': %d,      matched \'3\': %d' % (counter_1, counter_2, counter_3))
print('all \'1\':     %d,     all \'2\':     %d,      all \'3\':     %d' % (true_counter_1, true_counter_2, true_counter_3))
print('false \'1\':   %d,     false \'2\':   %d,      false \'3\':   %d' % (false_counter_1, false_counter_2, false_counter_3))

print('samples = %d, matched = %d, acc = %f' % (test_labels.shape[0], counter, counter / test_labels.shape[0]))

# NN #############################################################################################################

# model = keras.models.Sequential([
#     # keras.layers.Flatten(input_shape=(2,)),
#     keras.layers.Dense(15, activation=tf.nn.relu, input_dim=17),
#     keras.layers.Dense(3, activation=tf.nn.softmax)
# ])
#
# model.compile(
# #               optimizer=keras.optimizers.SGD(lr=0.01, clipnorm=1.),
#               optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_set, train_labels, epochs=2000, batch_size=100)
#
# test_loss, test_acc = model.evaluate(eval_set, eval_labels)
# y = model.predict(test_set, verbose=0)
#
# counter = 0
# counter_1 = 0
# counter_2 = 0
# counter_3 = 0
#
# true_counter_1 = 0
# true_counter_2 = 0
# true_counter_3 = 0
#
# false_counter_1 = 0
# false_counter_2 = 0
# false_counter_3 = 0
#
# decode_output_scheme = {0: 'H', 1: 'D', 2: 'A'}
#
# # for i in np.arange(y.shape[0]):
# #     print(y[i], test_labels[i])
#
# print(y.shape)
# print(test_labels.shape)
#
# for i in np.arange(test_labels.shape[0]):
#
#     arg = np.argmax(y[i])
#
#     if test_labels[i] == 0:
#         true_counter_1 += 1
#     elif test_labels[i] == 1:
#         true_counter_2 += 1
#     elif test_labels[i] == 2:
#         true_counter_3 += 1
#
#     if test_labels[i] == arg:
#         counter += 1
#         if arg == 0:
#             counter_1 += 1
#         elif arg == 1:
#             counter_2 += 1
#         elif arg == 2:
#             counter_3 += 1
#     elif arg == 0:
#         false_counter_1 += 1
#     elif arg == 1:
#         false_counter_2 += 1
#     elif arg == 2:
#         false_counter_3 += 1
#
# acc = counter/test_labels.shape[0]
#
# print('Counter = %d, %f' % (counter, acc))
# print('eval loss = %f, acc = %f' % (test_loss, test_acc))
#
# print('matched \'1\': %d,     matched \'2\': %d,      matched \'3\': %d' % (counter_1, counter_2, counter_3))
# print('all \'1\':     %d,     all \'2\':     %d,      all \'3\':     %d' % (true_counter_1, true_counter_2, true_counter_3))
# print('false \'1\':   %d,     false \'2\':   %d,      false \'3\':   %d' % (false_counter_1, false_counter_2, false_counter_3))
