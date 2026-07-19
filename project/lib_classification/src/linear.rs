use std::ascii::escape_default;
use rand::distr::weighted::Weight;
use rand::RngExt;

#[repr(C)]
pub struct MyDroite{
    a:f32,
    b:f32,
    c:f32
}

impl MyDroite {
    fn new(a:f32, b:f32, c:f32) -> Self{
        MyDroite{
            a,
            b,
            c
        }
    }

}


#[unsafe(no_mangle)]
pub extern "C" fn print_hi(){
    println!("Hello !");
}

#[unsafe(no_mangle)]
pub extern "C" fn initialisation_droite() -> *mut MyDroite{
    //--Creation de 3 nombre random float (entre 0 et 1)---
    let mut rng = rand::rng();
    let r1 : f32 = rng.random::<f32>() * 2.0 - 1.0;
    let r2 : f32 = rng.random();
    let r3 : f32 = rng.random();

    //---Creation de MyDroite avec 3 random f32 tuples---
    let droite = MyDroite::new(r1, r2, r3);

    //Box::into_raw() --> Return address (pour que le renvoie de pointeur fonctionne) depuis l'heap
    //Box::new(x)      --> envoie x dans l'heap
    Box::into_raw(Box::new(droite))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_classification_prediction(weights1: f32, weights2: f32, weights3: f32, xinput1 : f32, xinput2: f32, xinput3: f32 ) ->f32{
    let scal = weights1*xinput1 + weights2*xinput2 + weights3*xinput3;
    if scal>=0.0 {
        1.0
    } else{
        -1.0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn training_linear(pas_apprentissage: f32, n_loop : u32, ln_point : u32, mut w  : MyDroite, ensemble_point: *const f32, label: *const f32 ) -> *mut MyDroite{
    let data: &[f32] = unsafe {std::slice::from_raw_parts(ensemble_point, (ln_point* 2 ) as usize)};

    let labels: &[f32] = unsafe{std::slice::from_raw_parts(label, ln_point as usize)};

    let mut rng = rand::rng();
    let mut k : u32;
    let mut xk :MyDroite;
    let mut yk : f32;
    let mut gxk : f32;
    let mut x : f32;
    let mut y : f32;

    for _ in 0..n_loop{
        k = rng.random_range(0..ln_point);
        x = data[(k*2) as usize];
        y = data[(k*2+1) as usize];
        xk = MyDroite::new(1.0,x,y );
        //yk = labels[k as usize];
        yk = labels[(k) as usize];
        gxk = linear_classification_prediction(w.a, w.b, w.c, xk.a, xk.b, xk.c);

        let error = (yk - gxk) as f32;
        if error!=0 as f32 {
            w.a = w.a + pas_apprentissage* xk.a*(yk as f32 - gxk as f32);
            w.b = w.b + pas_apprentissage* xk.b*(yk as f32 - gxk as f32);
            w.c = w.c + pas_apprentissage* xk.c*(yk as f32 - gxk as f32);
        }
    }


    Box::into_raw(Box::new(w))
}