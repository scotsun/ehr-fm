import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


# raw data -> fewshot dataset -> setfit dataset

# raw data object: data
# each item is a dict {'input_ids', 'attention_mask', .., 'label'}

# fewshot dataset: FewShotDataset
# each item is a dict {'input_ids', 'attention_mask', .., 'label'}

# setfit dataset: SetFitDataset
# each item is a dict {'input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2', 'pair_label'}


class FewShotDataset(Dataset):
    def __init__(self, data, num_classes, num_samples_per_class, label_key="label"):
        self.data = data
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.label_key = label_key

        self.outcomes = data.outcomes  # assumed ndarray

        by_label = defaultdict(list)
        for i, y in enumerate(self.outcomes):
            by_label[int(y)].append(i)

        labels = list(by_label.keys())
        assert len(labels) >= num_classes

        chosen_labels = np.random.choice(labels, num_classes, replace=False)

        self.indices = []
        for lbl in chosen_labels:
            idxs = by_label[lbl]
            assert len(idxs) >= num_samples_per_class

            chosen = np.random.choice(idxs, num_samples_per_class, replace=False)
            self.indices.extend(chosen.tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_idx = self.indices[idx]
        item = dict(self.data[raw_idx])
        return item


class SetFitDataset(Dataset):
    def __init__(self, fewshot_dataset):
        super().__init__()
        self.fewshot_dataset = fewshot_dataset
        self.label_key = fewshot_dataset.label_key
        self._create_setfit_data()

    def _create_setfit_data(self):
        examples = [self.fewshot_dataset[i] for i in range(len(self.fewshot_dataset))]

        feature_keys = [k for k in examples[0].keys() if k != self.label_key]

        by_label = defaultdict(list)
        for ex in examples:
            by_label[ex[self.label_key]].append(ex)

        labels = sorted(by_label.keys())

        # Create all positive pairs within each class.
        positive_pairs = []
        for label in labels:
            items = by_label[label]
            # Need at least 2 samples to form a positive pair.
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    ex1, ex2 = items[i], items[j]
                    pair = {"pair_label": 1}
                    for key in feature_keys:
                        pair[f"{key}_a"] = ex1[key]
                        pair[f"{key}_b"] = ex2[key]
                    positive_pairs.append(pair)

        # Create a matched number of negative pairs (different labels) deterministically.
        negative_pairs = []
        if len(labels) >= 2:
            num_negatives = len(positive_pairs)
            num_labels = len(labels)
            for n in range(num_negatives):
                label_a = labels[n % num_labels]
                # Choose a different label in a deterministic cycle.
                label_b = labels[(n + 1 + (n // num_labels)) % num_labels]
                if label_b == label_a:
                    label_b = labels[(n + 2) % num_labels]

                items_a = by_label[label_a]
                items_b = by_label[label_b]
                ex1 = items_a[n % len(items_a)]
                ex2 = items_b[(n // num_labels) % len(items_b)]

                pair = {"pair_label": 0}
                for key in feature_keys:
                    pair[f"{key}_a"] = ex1[key]
                    pair[f"{key}_b"] = ex2[key]
                negative_pairs.append(pair)

        setfit_data = positive_pairs + negative_pairs
        perm = np.random.permutation(len(setfit_data))
        setfit_data = [setfit_data[i] for i in perm]

        self.setfit_data = setfit_data

    def __len__(self):
        return len(self.setfit_data)

    def __getitem__(self, idx):
        return self.setfit_data[idx]
