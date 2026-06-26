use core::f32;
use std::vec;
use rand::RngExt;
use faer::{Mat, Side};
use faer::prelude::*;
use faer::traits::math_utils::max;

pub struct RBFModel {
    entree_dim: usize,
    nb_clusters: usize,

    liste_cluster: Vec<Vec<f32>>,
    liste_poids: Vec<f32>,

    gamma : f32,
}

impl RBFModel {
    fn new(entree_dim: usize, nb_clusters: usize) -> Option<Self> {
        if entree_dim == 0 ||  nb_clusters == 0 {
            return None;
        }

        let liste_cluster = vec![vec![0.0; entree_dim]; nb_clusters];
        let liste_poids = vec![0.0; nb_clusters];
        let gamma = 1.0;

        Some(Self {entree_dim, nb_clusters,liste_cluster, liste_poids, gamma})
    }

    fn distance(&self, ensemble_image: &[f32], cluster: &[f32]) -> f32{
        let mut somme = 0.0;

        for i in 0..self.entree_dim {
            somme += (ensemble_image[i] - cluster[i]) * (ensemble_image[i] - cluster[i]);
        }
        somme.sqrt()
    }

    fn lloyd(&mut self, ensemble_image: &[f32], nb_image: u32, mouvement_max:f32, max_loop: usize)  {
        let mut rng = rand::rng();

        let mut sommes = vec![vec![0.0; self.entree_dim]; self.nb_clusters];
        let mut compteurs = vec![0 as usize; self.nb_clusters];

        let mut mouvement = true;
        let mut compteur = 0;

        while mouvement {
            mouvement = false;

            //------On vide les index des données dans les cluster------
            for cluster in sommes.iter_mut() {
                for pixel in &cluster.iter_mut() {
                    pixel = 0.0;
                }
            }

            for cluster in compteurs.iter_mut() {
                
            }
        }

        }


    }

}

