use rand::RngExt;

pub struct LinearModel {
    entree_dim: usize,
    pas_apprentissage: f32,
    poids: Vec<f32>,
    biais: f32,
}

impl LinearModel {
    fn new(entree_dim: usize, pas_apprentissage: f32) -> Option<Self> {

        if entree_dim == 0 || !pas_apprentissage.is_finite() || pas_apprentissage <= 0.0 {
            return None;
        }

        //---Innitialisation des poids et biais---
        let mut rng = rand::rng();
        let mut poids = vec![0.0; entree_dim];

        for petit_poids in &mut poids {
            *petit_poids = rng.random();
        }
        let biais = rng.random();

        Some(Self {entree_dim, pas_apprentissage, poids, biais})
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


        //Some(self.poids.iter().zip(features.iter()).fold(self.biais, |total, (&weight, &feature)| total + weight * feature))
        Some(score)
    }

    fn entrainement(&mut self, nb_donnee: &[f32], liste_label: &[f32], nb_pixel_image: usize, epoch: usize) -> bool {
        if nb_pixel_image == 0 || nb_donnee.len() != nb_pixel_image * self.entree_dim || liste_label.len() != nb_pixel_image {
            return false;
        }

        // training on random samples
        let mut rng = rand::rng();

        for _ in 0..epoch {
            let k = rng.random_range(0..nb_pixel_image);
            let start = k * self.entree_dim;
            let features = &nb_donnee[start..start + self.entree_dim];


            let yk = if liste_label[k] >= 0.0 { 1.0 } else { -1.0 };
            let gxk = if self.prediction(features).unwrap_or(0.0) >= 0.0 {1.0} else {-1.0};

            /*self.biais += self.pas_apprentissage * (yk - gxk);
            for (petit_poid, &feature) in self.poids.iter_mut().zip(features.iter()) {
                *petit_poid += self.pas_apprentissage * feature * (yk - gxk);
            }*/

            for i in 0..self.poids.len() {

                let correction = self.pas_apprentissage * features[i] * (yk - gxk);
                self.poids[i] += correction;
            }
        }

        true
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_model_create(entree_dim: usize, pas_apprentissage: f32) -> *mut LinearModel {
    // creation of the linear model
    match LinearModel::new(entree_dim, pas_apprentissage) {
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
