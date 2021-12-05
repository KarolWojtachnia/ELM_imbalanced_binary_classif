# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # X, y = make_classification(
    #     n_samples=700,
    #     n_features=8,
    #     n_informative=8,
    #     n_repeated=0,
    #     n_redundant=0,
    #     flip_y=.15,
    #     random_state=32,
    #     n_clusters_per_class=1,
    #     weights=[0.9, 0.1]
    # )

    # dataset = 'datasets\ecoli4.csv'
    # # dataset = 'datasets\glass4.csv'
    # dataset = np.genfromtxt(dataset, delimiter=",")
    # X = dataset[:, :-1]
    # y = dataset[:, -1].astype(int)
    # for i in range(y.shape[0]):
    #     if y[i] == 0:
    #         y[i] = -1
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y,
    #     test_size=.3,
    #     random_state=42
    # )
    # clf = ExtremeLearningMachine(pow(2, 2), 1000)
    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    #
    # print(accuracy_score(y_test, predictions))

    # import glob
    # import os
    #
    # counter = 0
    # os.chdir("datasets")
    # for file in glob.glob("*.csv"):
    #     print(file.replace('.csv', ''))
    #     counter += 1
    # print(counter)

    import numpy as np
    print(np.random.uniform(size = [3,4]))


