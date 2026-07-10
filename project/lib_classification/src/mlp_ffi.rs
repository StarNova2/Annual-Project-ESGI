use crate::mlp::Mlp;

#[unsafe(no_mangle)]
pub extern "C" fn mlp_create(layer_sizes: *const usize, layer_count: usize, seed: u64) -> *mut Mlp {
    if layer_sizes.is_null() || layer_count < 2 {
        return std::ptr::null_mut();
    }

    // creation of the layer sizes slice
    let layer_sizes = unsafe { std::slice::from_raw_parts(layer_sizes, layer_count) };
    if layer_sizes.iter().any(|&size| size == 0 || size > i32::MAX as usize) {
        return std::ptr::null_mut();
    }

    // creation of the mlp dimensions
    let input_size = layer_sizes[0] as usize;
    let output_size = layer_sizes[layer_count - 1] as usize;
    let hidden_layers: Vec<usize> = layer_sizes[1..layer_count - 1]
        .iter()
        .map(|&size| size)
        .collect();

    Box::into_raw(Box::new(Mlp::new(input_size, hidden_layers, output_size, seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_train(
    model: *mut Mlp,
    dataset_inputs: *const f32,
    dataset_expected_outputs: *const f32,
    sample_count: usize,
    training_steps: usize,
    learning_rate: f32,
    is_classification: u8,
) -> i32 {
    if model.is_null() || dataset_inputs.is_null() || dataset_expected_outputs.is_null() {
        return -1;
    }

    // creation of the flat input and output slices
    let model = unsafe { &mut *model };
    let input_dim = model.input_size();
    let output_dim = model.output_size();
    let input_len = sample_count * input_dim;
    let output_len = sample_count * output_dim;
    let inputs = unsafe { std::slice::from_raw_parts(dataset_inputs, input_len) };
    let outputs = unsafe { std::slice::from_raw_parts(dataset_expected_outputs, output_len) };

    // creation of the dataset matrices
    let dataset_inputs: Vec<Vec<f64>> = inputs
        .chunks_exact(input_dim)
        .map(|row| row.iter().map(|&value| value as f64).collect())
        .collect();
    let dataset_outputs: Vec<Vec<f64>> = outputs
        .chunks_exact(output_dim)
        .map(|row| row.iter().map(|&value| value as f64).collect())
        .collect();

    model.training(
        dataset_inputs,
        dataset_outputs,
        training_steps as i32,
        learning_rate as f64,
        is_classification != 0,
    );

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_predict(
    model: *mut Mlp,
    inputs: *const f32,
    output: *mut f32,
    is_classification: u8,
) -> i32 {
    if model.is_null() || inputs.is_null() || output.is_null() {
        return -1;
    }

    // creation of the input slice and output
    let model = unsafe { &mut *model };
    let input_size = model.input_size();
    let output_size = model.output_size();
    let inputs = unsafe { std::slice::from_raw_parts(inputs, input_size) };
    let input: Vec<f64> = inputs.iter().map(|&value| value as f64).collect();
    let prediction = model.prediction(&input, is_classification != 0);
    let output = unsafe { std::slice::from_raw_parts_mut(output, output_size) };

    for (target, value) in output.iter_mut().zip(prediction.iter()) {
        *target = *value as f32;
    }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_output_dim(model: *const Mlp) -> usize {
    if model.is_null() {
        return 0;
    }

    unsafe { (&*model).output_size() }
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_get_weights(
    model: *const Mlp,
    len: *mut usize,
) -> *mut f64 {
    if model.is_null() || len.is_null() {
        return std::ptr::null_mut();
    }

    let weights = unsafe { (&*model).flattened_weights() };

    unsafe {
        *len = weights.len();
    }

    let boxed = weights.into_boxed_slice();
    Box::into_raw(boxed) as *mut f64
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_free_weights(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_get_deltas(
    model: *const Mlp,
    len: *mut usize,
) -> *mut f64 {
    if model.is_null() || len.is_null() {
        return std::ptr::null_mut();
    }

    let deltas = unsafe { (&*model).flattened_deltas() };

    unsafe {
        *len = deltas.len();
    }

    let boxed = deltas.into_boxed_slice();
    Box::into_raw(boxed) as *mut f64
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_free_deltas(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mlp_free(model: *mut Mlp) {
    if model.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(model);
    }
}
