use std::ascii::escape_default;
use rand::distr::weighted::Weight;
use rand::RngExt;

struct MyDroite{
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
    //gÃ©nÃ©ration de 3 nombres random
    let mut rng = rand::rng();
    let r1 : f32 = rng.random();
    let r2 : f32 = rng.random();
    let r3 : f32 = rng.random();

    //CrÃ©ation de MyDroite contenant un tuples de 3 random f32
    let droite = MyDroite::new(r1, r2, r3);

    //Box::into_raw() --> renvoie l'adresse (pour que le renvoie de pointeur fonctionne)depuis l'heap
    //Box::new(x)      --> envoie x dans l'heap
    Box::into_raw(Box::new(droite))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_classification_prediction(weights1: f32, weights2: f32, weights3: f32, xinput1 : f32, xinput2: f32, xinput3: f32 ) ->i8{
    let scal = weights1*xinput1 + weights2*xinput2 + weights3*xinput3;
    if(scal>=0.0){
        1
    } else{
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn training(n_loop : u32, ensemble : u32, mut w  : MyDroite, ensemble_point: *const f32, label: *const i8 ) -> *mut MyDroite{
    let data: &[f32] = unsafe {
        std::slice::from_raw_parts(ensemble_point, (ensemble* 2 ) as usize)
    };

    let labels: &[i8] = unsafe{
        std::slice::from_raw_parts(label, ensemble as usize)
    };

    let mut rng = rand::rng();
    let mut k : u32;
    let mut xk :MyDroite;
    let mut yk : i8;
    let mut gxk : i8;
    let mut x : f32;
    let mut y : f32;

    for _ in 0..n_loop{
        k = rng.random_range(0..ensemble);
        x = data[(k*2) as usize];
        y = data[(k*2+1) as usize];
        xk = MyDroite::new(1.0,x,y );
        yk = labels[k as usize];
        gxk = linear_classification_prediction(w.a, w.b, w.c, xk.a, xk.b, xk.c);
        w.a = w.a + 0.001* xk.a*(yk-gxk) as f32;
        w.b = w.b + 0.001* xk.b*(yk-gxk) as f32;
        w.c = w.c + 0.001* xk.c*(yk-gxk) as f32;
    }
    Box::into_raw(Box::new(w))
}
