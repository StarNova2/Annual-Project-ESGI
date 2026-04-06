use rand::RngExt;

pub struct LinearModel {
    input_dim: usize,
    learning_rate: f32,
    weights: Vec<f32>,
    bias: f32,
}

impl LinearModel {
    fn new(input_dim: usize, learning_rate: f32) -> Option<Self> {
        if input_dim == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return None;
        }

        // initialization of the weights and bias
        let mut rng = rand::rng();
        let mut weights = vec![0.0; input_dim];
        for weight in &mut weights {
            *weight = rng.random();
        }

        Some(Self {
            input_dim,
            learning_rate,
            weights,
            bias: rng.random(),
        })
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn predict(&self, features: &[f32]) -> Option<f32> {
        if features.len() != self.input_dim {
            return None;
        }

        // computation of the linear score
        Some(
            self.weights
                .iter()
                .zip(features.iter())
                .fold(self.bias, |total, (&weight, &feature)| total + weight * feature),
        )
    }

    fn fit(&mut self, inputs: &[f32], targets: &[f32], sample_count: usize, n_loop: usize) -> bool {
        if sample_count == 0
            || inputs.len() != sample_count * self.input_dim
            || targets.len() != sample_count
        {
            return false;
        }

        // training on random samples
        let mut rng = rand::rng();
        for _ in 0..n_loop {
            let k = rng.random_range(0..sample_count);
            let start = k * self.input_dim;
            let features = &inputs[start..start + self.input_dim];
            let yk = if targets[k] >= 0.0 { 1.0 } else { -1.0 };
            let gxk = if self.predict(features).unwrap_or(-1.0) >= 0.0 {
                1.0
            } else {
                -1.0
            };

            self.bias += self.learning_rate * (yk - gxk);
            for (weight, &feature) in self.weights.iter_mut().zip(features.iter()) {
                *weight += self.learning_rate * feature * (yk - gxk);
            }
        }

        true
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_create(input_dim: usize, learning_rate: f32) -> *mut LinearModel {
    // creation of the linear model
    match LinearModel::new(input_dim, learning_rate) {
        Some(model) => Box::into_raw(Box::new(model)),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_train(
    model: *mut LinearModel,
    inputs: *const f32,
    targets: *const f32,
    sample_count: usize,
    epochs: usize,
) -> i32 {
    if model.is_null() || inputs.is_null() || targets.is_null() {
        return -1;
    }

    // creation of the input and target slices
    let model = unsafe { &mut *model };
    let inputs = unsafe { std::slice::from_raw_parts(inputs, sample_count * model.input_dim()) };
    let targets = unsafe { std::slice::from_raw_parts(targets, sample_count) };

    if model.fit(inputs, targets, sample_count, epochs) {
        0
    } else {
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_predict(
    model: *const LinearModel,
    features: *const f32,
    feature_len: usize,
) -> f32 {
    if model.is_null() || features.is_null() {
        return f32::NAN;
    }

    // creation of the feature slice
    let model = unsafe { &*model };
    let features = unsafe { std::slice::from_raw_parts(features, feature_len) };
    model.predict(features).unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_free(model: *mut LinearModel) {
    if model.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(model);
    }
}
