use crate::svm::Svm;

#[unsafe(no_mangle)]
pub extern "C" fn svm_create(layer_sizes: *const usize, layer_count: usize) -> *mut Svm {
    if layer_sizes.is_null() || layer_count < 2{
        return std::ptr::null_mut();
    }

    let layer_sizes = unsafe { std::slice::from_raw_parts(layer_sizes, layer_count)};
    if layer_sizes.iter().any(|&size| size == 0 || size > i32::MAX as usize) {
        return std::ptr::null_mut();
    }

    let input_size = layer_sizes[0] as i32;
    let output_size = layer_sizes[layer_count - 1] as i32;
    Box::into_raw(Box::new(Svm::new(input_size,output_size)))
}

#[unsafe(no_mangle)]
pub extern "C" fn svm_train(){
}

#[unsafe(no_mangle)]
pub extern "C" fn svm_predict(
    
){
}

#[unsafe(no_mangle)]
pub extern "C" fn svm_output_dim(model: *const Svm) -> usize {
    if model.is_null(){
        return 0;
    }

    unsafe { (&*model).output_size() }
}

#[unsafe(no_mangle)]
pub extern "C" fn svm_free(model: *mut Svm) {
    if model.is_null(){
        return;
    }

    unsafe {
        let _ = Box::from_raw(model);
    }
}