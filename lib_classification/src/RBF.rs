use std::vec;
use rand::RngExt;
use faer::{Mat, Side};
use faer::prelude::*;


fn distance_euclidienne(x1: f32, y1:f32, x2:f32, y2:f32) -> f32{
    let dx = x1 - x2;
    let dy = y1 - y2;

    (dx * dx  +  dy * dy).sqrt()
}


#[unsafe(no_mangle)]
pub extern "C" fn lloyd(num_k: usize, env_x_min_max: *const f32, env_y_min_max: *const f32, ensemble_point: *const f32, len_point: usize,mouvement_max:f32, max_loop: f32, output: *mut f32)  {
    let mut rng = rand::rng();
    //list_cluster = coordonée du cluster (x, y) + somme des coorodnnée (x, y) + nombre de cooronée pour faire la moyenne (len)
    //--> Vec<(x_cluster, y_cluster, x_sum, y_sum, count)>
    let mut list_cluster: Vec<(f32, f32, f32, f32, usize)> = Vec::new();


    //Convertion des pointeurs en liste [x,y]
    let env_x = unsafe { std::slice::from_raw_parts(env_x_min_max, 2) };
    let env_y = unsafe { std::slice::from_raw_parts(env_y_min_max, 2) };
    let data: &[f32] = unsafe { std::slice::from_raw_parts(ensemble_point, (len_point * 2) as usize) };

    let mut x: f32;
    let mut y: f32;

    //Mise en place des cluster aléatoirement
    for _ in 0..num_k as i32 {
        x = rng.random_range(env_x[0]..=env_x[1]);
        y = rng.random_range(env_y[0]..=env_y[1]);

        list_cluster.push((x, y, 0.0, 0.0, 0));
    }



    let mut mouvement = true;
    let mut count_boucle_max = 0;

    //----Boucle qui se termine si le mouvement des cluster sont inférieur au mouvemnet max proposé ou si l'itération max est dépassé----
    while mouvement{

        mouvement = false;
        //------On vide les index des données dans les cluster------
        for k in list_cluster.iter_mut() {
            k.2 = 0.0;
            k.3 = 0.0;
            k.4 = 0;
        }


        //---------Assignation des points aux clusters------------
        for i in 0..len_point {
            x = data[(i * 2) as usize];
            y = data[(i * 2 + 1) as usize];

            let mut meilleur_distance: f32 = f32::MAX;
            let mut meilleur_index: usize = 0;

            for (index, k) in list_cluster.iter().enumerate() {
                let distance = distance_euclidienne(x, y, k.0, k.1);
                if distance < meilleur_distance {
                    meilleur_distance = distance;
                    meilleur_index = index;
                }
            }
            list_cluster[meilleur_index].2 += x;
            list_cluster[meilleur_index].3 += y;
            list_cluster[meilleur_index].4 +=1;
        }


        //----------Mise à jour des Cluster-------------
        for k in list_cluster.iter_mut() {
            //Si il y pas de points dans le cluster, on le passe
            if k.4 ==0 {
                continue;
            }

            //Regarde si on doit recommencer une itération en regardant si la différence des coordonnées sontr trop grande
            let moy_x = k.2 / k.4 as f32;
            let moy_y = k.3 / k.4 as f32;

            let old_x = k.0;
            let old_y = k.1;

            k.0 = moy_x;
            k.1 = moy_y;

            if (k.0 - old_x).abs() > mouvement_max || (k.1 - old_y).abs() > mouvement_max {
                mouvement = true;
            }

        }

        count_boucle_max += 1;
        if count_boucle_max > max_loop as i32 {
            mouvement = false;
        }
    }

    //Rendu des cluster
    let output_cluster = unsafe {
        std::slice::from_raw_parts_mut(output, (num_k * 2) as usize)};

    for (i, k) in list_cluster.iter().enumerate() {
        output_cluster[i * 2] = k.0;
        output_cluster[i * 2 + 1] = k.1;
    }

}


#[unsafe(no_mangle)]
pub extern "C" fn RBF_train(gamma : f32, ensemble_point: *const f32, len_point: usize, list_cluster: *const f32, len_list_cluster: usize, ensemble_label: *const f32, output: *mut f32) {
    //convertion des pointeurs
    let data: &[f32] = unsafe { std::slice::from_raw_parts(ensemble_point, (len_point * 2) as usize) };
    let list_cluster: &[f32] = unsafe { std::slice::from_raw_parts(list_cluster, (len_list_cluster * 2) as usize) };
    let label: &[f32] = unsafe { std::slice::from_raw_parts(ensemble_label, len_point as usize)};

    //creation d'une liste contenant des listes
    let mut liste_c_apres_calcul =vec![vec![0.0; len_list_cluster as usize]; len_point as usize];


    //----- création d'une "pré-matrice" pour calculer les poids-------
    for j in 0..len_point {

        let x = data[(j * 2) as usize];
        let y = data[(j * 2+1) as usize];

        for i in 0..len_list_cluster {

            let cx = list_cluster[(i * 2) as usize];
            let cy = list_cluster[(i * 2+1) as usize];

            let dx = x-cx;
            let dy = y-cy;

            let d = dx*dx + dy*dy;

            liste_c_apres_calcul[j as usize][i as usize] = (-gamma * d).exp();
        }
    }


    //------innitialisation de la matrice + label pour calculer les poids-------
    let mut label_y = Mat::<f32>::zeros(len_point as usize, 1);
    let mut phi = Mat::<f32>::zeros(len_point as usize, len_list_cluster as usize);


    for i in 0..len_point as usize{
        for j in 0..len_list_cluster as usize{
            phi[(i, j)] = liste_c_apres_calcul[i][j];
        }
        label_y[(i, 0)] = label[i];
    }

    //calcul des poids
    let phi_t = phi.transpose();
    
    let phi_t_y = &phi_t * &label_y;

    let phi_t_phi = &phi_t * &phi;

    let llt = phi_t_phi.llt(Side::Lower).unwrap();

    let w = llt.solve(&phi_t_y);


    let output_w = unsafe {
        std::slice::from_raw_parts_mut(
            output,
            len_list_cluster as usize
        )
    };


    for i in 0..w.nrows() {
        output_w[i] = w[(i,0)];
    }

}


#[unsafe(no_mangle)]
pub extern "C" fn RBF_predict (gamma:f32, num_k:usize, cluster:*const f32, point_entre: *const f32, poids: *const f32) -> f32{
    //Convertion des pointeurs
    let liste_cluster = unsafe{std::slice::from_raw_parts(cluster, (num_k*2) as usize)};
    let data = unsafe{std::slice::from_raw_parts(point_entre, 2)};
    let liste_poids = unsafe {std::slice::from_raw_parts(poids, num_k as usize)};

    let mut score = 0.0;

    //mise en place des score
    for i in 0..num_k{
        let cx = liste_cluster[(i*2) as usize];
        let cy = liste_cluster[(i*2+1) as usize];

        let dx = data[0] - cx;
        let dy = data[1] - cy;

        let d = dx * dx + dy * dy;

        score += (-gamma*d).exp() * liste_poids[i as usize];
    }


    if score >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

