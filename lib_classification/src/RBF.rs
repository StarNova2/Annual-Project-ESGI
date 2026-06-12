use rand::RngExt;

#[unsafe(no_mangle)]
pub extern "C" fn lloyd(ensemble_point: *const f32, num_k: f32, n_loop: f32) -> *mut f32{

}

#[unsafe(no_mangle)]
pub extern "C" fn RBF_train(pas_apprentissage : f32, ensemble_point: *const f32, n_loop: f32, cluster_1: f32, cluster_2: f32){

}