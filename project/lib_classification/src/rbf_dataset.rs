use core::f32;
use std::vec;
use rand::RngExt;
use faer::{Mat, Side};
use faer::prelude::*;


pub struct RBFModel {
    entree_dim: usize,
    nb_clusters: usize,

    liste_cluster: Vec<Vec<f32>>,
    liste_poids: Vec<f32>,

    gamma : f32,
}

impl RBFModel {
    fn new(entree_dim: usize, nb_clusters: usize) -> Option<Self> {
        if entree_dim == 0 || nb_clusters == 0 {
            return None;
        }

        let liste_cluster = vec![vec![0.0; entree_dim]; nb_clusters];
        let liste_poids = vec![0.0; nb_clusters];
        let gamma = 1.0;

        Some(Self { entree_dim, nb_clusters, liste_cluster, liste_poids, gamma })
    }


    fn entree_dim(&self) -> usize {
        self.entree_dim
    }


    fn distance(&self, ensemble_image: &[f32], cluster: &[f32]) -> f32 {
        let mut somme = 0.0;

        for i in 0..self.entree_dim {
            somme += (ensemble_image[i] - cluster[i]) * (ensemble_image[i] - cluster[i]);
        }
        somme.sqrt()
    }


    fn phi(&self, image: &[f32], cluster: &[f32]) -> f32{
        let mut somme = 0.0;
        for i in 0..self.entree_dim{
            somme += (image[i] - cluster[i]) * (image[i] - cluster[i]);
        }

        (-self.gamma * somme).exp()
    }


    fn calcul_phi_image(&self, image : &[f32]) -> Vec<f32>{
        let mut activation = vec![0.0; self.nb_clusters];

        for index_cluster in 0..self.nb_clusters{
            activation[index_cluster] = self.phi(image, &self.liste_cluster[index_cluster]);
        }
        activation
    }

    fn prediction (&self, image : &[f32]) -> Option<f32>{
        if image.len() != self.entree_dim {
            return None;
        }

        let activation = self.calcul_phi_image(image);
        let mut score = 0.0;

        for i in 0..self.nb_clusters{
            score += self.liste_poids[i] * activation[i];

        }
        Some(score)
    }



    fn lloyd(&mut self, ensemble_image: &[f32], nb_image: usize, mouvement_max: f32, max_loop: f32) {
        let mut rng = rand::rng();

        //---Mise en place des cluster aléatoirement---
        for cluster in 0..self.nb_clusters{
            let image_rng = rng.random_range(0..nb_image as usize);
            let debut = image_rng * self.entree_dim;

            let image = &ensemble_image[debut..debut + self.entree_dim];
            for pixel in 0..self.entree_dim{
                self.liste_cluster[cluster][pixel] = image[pixel];
            }


        }

        let mut sommes = vec![vec![0.0; self.entree_dim]; self.nb_clusters];
        let mut compteurs = vec![0 as usize; self.nb_clusters];

        let mut mouvement = true;
        let mut compteur = 0;

        while mouvement {
            mouvement = false;

            //------On vide les index des données dans les cluster------
            for cluster in sommes.iter_mut() {
                for pixel in cluster.iter_mut() {
                    *pixel = 0.0;
                }
            }

            for cluster in compteurs.iter_mut() {
                *cluster = 0;
            }

            //---Assignation des images aux cluster----
            for index_image in 0..nb_image {
                let debut = index_image as usize * self.entree_dim;
                let image = &ensemble_image[debut..debut + self.entree_dim];


                //---Les meilleurs cluster---
                let mut meilleur_distance: f32 = f32::MAX;
                let mut meilleur_index: usize = 0;

                for cluster in 0..self.nb_clusters {
                    let distance = self.distance(image, &self.liste_cluster[cluster]);

                    if distance < meilleur_distance {
                        meilleur_distance = distance;
                        meilleur_index = cluster;
                    }
                }

                for pixel in 0..self.entree_dim {
                    sommes[meilleur_index][pixel] += image[pixel];
                }
                compteurs[meilleur_index] += 1;
            }

            //---Mise à jour des Cluster---
            for cluster in 0..self.nb_clusters {
                if compteurs[cluster] == 0 {
                    continue;
                }

                let mut mouvement_actuel = 0.0;


                for pixel in 0..self.entree_dim {
                    let ancien = self.liste_cluster[cluster][pixel];
                    let nouveau = sommes[cluster][pixel] / compteurs[cluster] as f32;

                    self.liste_cluster[cluster][pixel] = nouveau;

                    mouvement_actuel += (nouveau - ancien).abs();
                }

                //---Regarde si on doit recommencer une itération---
                if mouvement_actuel > mouvement_max {
                    mouvement = true;
                }
            }

            //---Est ce que le nombre d'itération max est terminé---
            compteur += 1;
            if compteur > max_loop as usize {
                break;
            }
        }
    }


    fn entrainement(&mut self, ensemble_images : &[f32],nb_image: usize, mouvement_max: f32, max_loop: f32, label_y : &[f32]) -> bool{
        if nb_image == 0 || ensemble_images.len() != nb_image * self.entree_dim{
            return false;
        }

        self.lloyd(ensemble_images, nb_image, mouvement_max, max_loop);



        let mut phi = Mat::<f32>::zeros(nb_image as usize, self.nb_clusters as usize);

        for index_image in 0..nb_image{
            let debut = index_image as usize * self.entree_dim;
            let image = &ensemble_images[debut..debut + self.entree_dim];

            for index_cluster in 0..self.nb_clusters{

                let valeur_phi = self.phi(image, &self.liste_cluster[index_cluster]);
                phi[(index_image as usize, index_cluster)] = valeur_phi;
            }

        }


        let mut label = Mat::<f32>::zeros(nb_image as usize, 1);

        for i in 0..nb_image as usize {
            label[(i,0)] = label_y[i];
        }

        //--calcul des poids--
        let phi_t = phi.transpose();

        let phi_t_y = &phi_t * &label;

        let phi_t_phi = &phi_t * &phi;

        //let llt = phi_t_phi.llt(Side::Lower).unwrap();

        let llt = match phi_t_phi.llt(Side::Lower) {
            Ok(decomp) => decomp,
            Err(_) => return false,  // retourne false au lieu de crasher
        };

        let w = llt.solve(&phi_t_y);

        self.liste_poids.clear();

        for i in 0..self.nb_clusters{
            self.liste_poids.push(w[(i, 0)]);
        }

        true
    }


    fn prediction_label(&self, image : &[f32]) -> Option<f32>{

        if self.prediction(image).unwrap_or(f32::NAN) >= 0.0{
            Some(1.0)
        } else {
            Some(-1.0)
        }

    }

}

#[unsafe(no_mangle)]
pub extern "C" fn creation_RBF_model(entree_dim: usize, nb_clusters: usize) -> *mut RBFModel {
    //--Creation du model RBF
    match RBFModel::new(entree_dim, nb_clusters) {
        Some(model) => Box::into_raw(Box::new(model)),
        None => std::ptr::null_mut(),
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn entrainement_RBF_model(model: *mut RBFModel,entree: *const f32, label: *const f32, sample_count: usize, mouvement_max: f32, max_loop: f32, gamma: f32) -> i32 {
    if model.is_null() || entree.is_null() || label.is_null() {
        return -1;
    }

    //--convertion des pointeurs--
    let model = unsafe { &mut *model };
    model.gamma = gamma;
    let entree_ptr = unsafe { std::slice::from_raw_parts(entree, sample_count * model.entree_dim() )};
    let label_ptr = unsafe { std::slice::from_raw_parts(label, sample_count) };



    if model.entrainement(entree_ptr, sample_count, mouvement_max, max_loop, label_ptr){
        0
    } else {
        -1
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn RBF_model_predict(model: *const RBFModel, features: *const f32, feature_len: usize) -> f32 {
    if model.is_null() || features.is_null() {
        return f32::NAN;
    }

    // creation of the feature slice
    let model = unsafe { &*model };
    let features = unsafe { std::slice::from_raw_parts(features, feature_len) };
    model.prediction_label(features).unwrap_or(f32::NAN)
}


#[unsafe(no_mangle)]
pub extern "C" fn RBF_model_free(model: *mut RBFModel) {
    if model.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(model);
    }
}


