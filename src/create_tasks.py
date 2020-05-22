import itertools
import pickle

import numpy as np

def create_tasks(n_tasks, n_classes):
    """Create n tasks from a given dataset for a continual learning experiment."""

    task_settings = set(itertools.product(set(range(n_classes)), repeat=2))

    tasks = []
    for meta_task in range(int(n_tasks / (n_classes))):
        avail_classes = list(range(n_classes))

        for inner_task in range(int(n_classes / 2)):
            cur_classes = np.random.choice(avail_classes, size=2, replace=False)
            avail_classes.remove(cur_classes[0])
            avail_classes.remove(cur_classes[1])
            tasks.append(cur_classes)

    with open("tasks.pkl", "wb") as filewriter:
        pickle.dump(tasks, filewriter)


if __name__ == "__main__":
    create_tasks(500, 100)