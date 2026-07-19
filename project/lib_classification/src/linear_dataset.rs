use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use rand::RngExt;
use faer::{Mat, Side};
use faer::prelude::*;

pub struct LinearModel {
    entree_dim: usize,
    pas_apprentissage: f32,
    poids: Vec<f32>,
    biais: f32,
    rng: StdRng
}

impl LinearModel {
    fn new(entree_dim: usize, pas_apprentissage: f32, rand_seed: u64) -> Option<Self> {

        if entree_dim == 0 || !pas_apprentissage.is_finite() || pas_apprentissage <= 0.0 {
            return None;
        }

        //---Innitialisation des poids et biais---
        let mut rng = StdRng::seed_from_u64(rand_seed);
        let mut poids = vec![0.0; entree_dim];

        for petit_poids in &mut poids {
            *petit_poids = rng.random();
        }
        let biais = rng.random();

        Some(Self {entree_dim, pas_apprentissage, poids, biais, rng})
    }

    fn entree_dim(&self) -> usize {
        self.entree_dim
    }

    fn prediction(&self, liste_valeur_pixel: &[f32]) -> Option<f32> {
        if liste_valeur_pixel.len() != self.entree_dim {
            return None;
        }


        let mut score = self.biais;

        //--Calcul du score---
        for i in 0..self.poids.len() {
            score += self.poids[i] * liste_valeur_pixel[i];
        }


        Some(score)
    }

    fn regression(&mut self, donnee: &[f32], liste_label: &[f32], nb_samples: usize, nb_features: usize) -> i32 {
        if donnee.len() != nb_samples * nb_features || nb_features != self.entree_dim {
            return -1;
        }

        let mut x = Mat::<f32>::zeros(nb_samples, nb_features + 1);
        let mut y = Mat::<f32>::zeros(nb_samples, 1);

        for i in 0..nb_samples {
            x[(i, 0)] = 1.0;
            for j in 0..nb_features {
                x[(i, j + 1)] = donnee[i * nb_features + j];   // <- j+1, pas j
            }
            y[(i, 0)] = liste_label[i];
        }

        let xt = x.transpose();
        let xtx = &xt * &x;
        let xty = &xt * &y;

        let llt = match xtx.llt(Side::Lower) {
            Ok(decomp) => decomp,
            Err(_) => return -2,
        };
        let w = llt.solve(&xty);

        self.biais = w[(0, 0)];
        self.poids = (0..nb_features).map(|i| w[(i + 1, 0)]).collect();
        0
    }

    fn entrainement(&mut self, nb_donnee: &[f32], liste_label: &[f32], nb_pixel_image: usize, epoch: usize) -> bool {
        if nb_pixel_image == 0 || nb_donnee.len() != nb_pixel_image * self.entree_dim || liste_label.len() != nb_pixel_image {
            return false;
        }

        // training on random samples
        for _ in 0..epoch {
            let k = self.rng.random_range(0..nb_pixel_image);
            let start = k * self.entree_dim;
            let features = &nb_donnee[start..start + self.entree_dim];


            let yk = liste_label[k];
            let gxk = self.prediction(features).unwrap_or(0.0);
            //let yk = if liste_label[k] >= 0.0 { 1.0 } else { -1.0 };
            //let gxk = if self.prediction(features).unwrap_or(0.0) >= 0.0 {1.0} else {-1.0};

            self.biais += self.pas_apprentissage * (yk - gxk);
            /*for (petit_poid, &feature) in self.poids.iter_mut().zip(features.iter()) {
                *petit_poid += self.pas_apprentissage * feature * (yk - gxk);
            }*/

            for i in 0..self.poids.len() {

                self.poids[i] += self.pas_apprentissage * features[i] * (yk - gxk);

            }
        }

        true
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_create(entree_dim: usize, pas_apprentissage: f32, seed: u64) -> *mut LinearModel {
    // creation of the linear model
    match LinearModel::new(entree_dim, pas_apprentissage, seed) {
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
    epochs: usize
) -> i32 {
    if model.is_null() || inputs.is_null() || targets.is_null() {
        return -1;
    }

    // creation of the input and target slices
    let model = unsafe { &mut *model };
    let inputs = unsafe { std::slice::from_raw_parts(inputs, sample_count * model.entree_dim()) };
    let targets = unsafe { std::slice::from_raw_parts(targets, sample_count) };

    if model.entrainement(inputs, targets, sample_count, epochs) {
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
    model.prediction(features).unwrap_or(f32::NAN)
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

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_regression(model: *mut LinearModel, nb_donnee: *const f32, liste_label: *const f32, nb_samples: usize, nb_features: usize){
    let model = unsafe { &mut *model };

    let x = unsafe { std::slice::from_raw_parts(nb_donnee, nb_samples * nb_features) };
    let y = unsafe { std::slice::from_raw_parts(liste_label, nb_samples) };

    model.regression(x, y, nb_samples, nb_features);
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_get_poids(model: *const LinearModel, out: *mut f32) -> i32 {
    if model.is_null() || out.is_null() { return -1; }
    let model = unsafe { &*model };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, model.entree_dim) };
    out_slice.copy_from_slice(&model.poids);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_get_bias(model: *const LinearModel) -> f32 {
    if model.is_null() { return f32::NAN; }
    unsafe { &*model }.biais
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_set_state(model: *mut LinearModel, weights_in: *const f32, bias: f32) -> i32 {
    if model.is_null() || weights_in.is_null() { return -1; }
    let model = unsafe { &mut *model };
    let weights_slice = unsafe { std::slice::from_raw_parts(weights_in, model.entree_dim) };
    model.poids.copy_from_slice(weights_slice);
    model.biais = bias;
    0
}
