"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import logging
import random


def add_noise_in_data(client_data, label_col, noise_rate):
    """Add noise in the data."""
    all_noise_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    assert (
        noise_rate in all_noise_rates
    ), f"Noise rate must be in {all_noise_rates} but got {noise_rate}"
    labels = client_data[label_col]

    unique_labels = list(set(labels))
    select_x_num = int(noise_rate * len(unique_labels))
    flip_labels = random.sample(unique_labels, select_x_num)

    def _lambda(example):
        label = example[label_col]
        if label in flip_labels:
            fli_label = random.choice(unique_labels)
            i = 0
            while fli_label == label:
                fli_label = random.choice(unique_labels)
                i += 1
                if i > 100:
                    logging.warn(
                        f"Could not find a different label for {label}. "
                        f"Results may be incorrect. "
                        f"It might be due ot random fix seed somewhere in the code."
                    )
                    break
            example[label_col] = fli_label
        return example

    client_data = client_data.map(_lambda)
    return client_data



