// Rust Machine Learning
// By JJ Reibel

use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::min;
use std::vec::Vec;

fn train_val_test_split<T: Clone>(
    X: &[T],
    y: &[T],
    val_size: f64,
    test_size: f64,
    epochs: usize,
    random_state: Option<u64>,
) -> (
    Vec<Vec<T>>,
    Vec<Vec<T>>,
    Vec<Vec<T>>,
    Vec<Vec<T>>,
    Vec<Vec<T>>,
    Vec<Vec<T>>,
) {
    let n_samples = X.len();
    let mut rng = match random_state {
        Some(seed) => rand::SeedableRng::seed_from_u64(seed),
        None => rand::rngs::ThreadRng::default(),
    };
    let mut idx: Vec<usize> = (0..n_samples).collect();
    idx.shuffle(&mut rng);
    let n_val = (n_samples as f64 * val_size).ceil() as usize;
    let n_test = (n_samples as f64 * test_size).ceil() as usize;

    let epoch_start_idx: Vec<usize> = (0..epochs)
        .map(|i| i * n_samples / epochs)
        .collect();
    let epoch_end_idx: Vec<usize> = epoch_start_idx[1..]
        .iter()
        .chain(std::iter::once(&n_samples))
        .copied()
        .collect();

    let mut train_idx_epoch = Vec::new();
    let mut val_idx_epoch = Vec::new();
    let mut test_idx_epoch = Vec::new();

    for i in 0..epochs {
        let epoch_indices = &idx[epoch_start_idx[i]..epoch_end_idx[i]];
        let val_idx = &epoch_indices[..n_val];
        let test_idx = &epoch_indices[n_val..n_val + n_test];
        let train_idx = &epoch_indices[n_val + n_test..];
        train_idx_epoch.push(train_idx.to_vec());
        val_idx_epoch.push(val_idx.to_vec());
        test_idx_epoch.push(test_idx.to_vec());
    }

    let mut X_train_epoch = Vec::new();
    let mut X_val_epoch = Vec::new();
    let mut X_test_epoch = Vec::new();
    let mut y_train_epoch = Vec::new();
    let mut y_val_epoch = Vec::new();
    let mut y_test_epoch = Vec::new();

    for i in 0..epochs {
        let train_idx = &train_idx_epoch[i];
        let val_idx = &val_idx_epoch[i];
        let test_idx = &test_idx_epoch[i];
        let X_train = train_idx
            .iter()
            .map(|&idx| X[idx].clone())
            .collect::<Vec<_>>();
        let X_val = val_idx
            .iter()
            .map(|&idx| X[idx].clone())
            .collect::<Vec<_>>();
        let X_test = test_idx
            .iter()
            .map(|&idx| X[idx].clone())
            .collect::<Vec<_>>();
        let y_train = train_idx
            .iter()
            .map(|&idx| y[idx].clone())
            .collect::<Vec<_>>();
        let y_val = val_idx
            .iter()
            .map(|&idx| y[idx].clone())
            .collect::<Vec<_>>();
        let y_test = test_idx
            .iter()
            .map(|&idx| y[idx].clone())
            .collect::<Vec<_>>();
        X_train_epoch.push(X_train);
        X_val_epoch.push(X_val);
        X_test_epoch.push(X_test);
        y_train_epoch.push(y_train);
        y_val_epoch.push(y_val);
        y_test_epoch.push(y_test);
    }

(
    X_train_epoch,
    X_val_epoch,
    X_test_epoch,
    y_train_epoch,
    y_val_epoch,
    y_test_epoch,
)

}


/* Example
let (X_train_epoch, X_val_epoch, X_test_epoch, y_train_epoch, y_val_epoch, y_test_epoch) =
    train_val_test_split(X, y, 0.1, 0.1, 5);

// Can loop through
X_train_epoch[0]; // training data for epoch 0
X_val_epoch[0]; // validation data for epoch 0
X_test_epoch[0]; // test data for epoch 0
y_train_epoch[0]; // training labels for epoch 0
y_val_epoch[0]; // validation labels for epoch 0
y_test_epoch[0]; // test labels for epoch 0

*/

/* Example 2
let mut X_train_epoch: Vec<Vec<f32>> = vec![];
let mut X_val_epoch: Vec<Vec<f32>> = vec![];
let mut X_test_epoch: Vec<Vec<f32>> = vec![];
let mut y_train_epoch: Vec<f32> = vec![];
let mut y_val_epoch: Vec<f32> = vec![];
let mut y_test_epoch: Vec<f32> = vec![];

let epochs = 5;
let (X_train, X_val, X_test, y_train, y_val, y_test) = train_val_test_split(X, y, 0.1, 0.1);

for epoch in 0..epochs {
    let start_train_idx = epoch * X_train.len() / epochs;
    let end_train_idx = (epoch + 1) * X_train.len() / epochs;
    let start_val_idx = epoch * X_val.len() / epochs;
    let end_val_idx = (epoch + 1) * X_val.len() / epochs;
    let start_test_idx = epoch * X_test.len() / epochs;
    let end_test_idx = (epoch + 1) * X_test.len() / epochs;

    let X_train_epoch_cur = X_train[start_train_idx..end_train_idx].to_vec();
    let X_val_epoch_cur = X_val[start_val_idx..end_val_idx].to_vec();
    let X_test_epoch_cur = X_test[start_test_idx..end_test_idx].to_vec();
    let y_train_epoch_cur = y_train[start_train_idx..end_train_idx].to_vec();
    let y_val_epoch_cur = y_val[start_val_idx..end_val_idx].to_vec();
    let y_test_epoch_cur = y_test[start_test_idx..end_test_idx].to_vec();

    X_train_epoch.push(X_train_epoch_cur);
    X_val_epoch.push(X_val_epoch_cur);
    X_test_epoch.push(X_test_epoch_cur);
    y_train_epoch.push(y_train_epoch_cur);
    y_val_epoch.push(y_val_epoch_cur);
    y_test_epoch.push(y_test_epoch_cur);
}

// Can loop through
// X_train_epoch[0] // training data for epoch 0
// X_val_epoch[0] // validation data for epoch 0
// X_test_epoch[0] // test data for epoch 0
// y_train_epoch[0] // training labels for epoch 0
// y_val_epoch[0] // validation labels for epoch 0
// y_test_epoch[0] // test labels for epoch 0


*/
   
