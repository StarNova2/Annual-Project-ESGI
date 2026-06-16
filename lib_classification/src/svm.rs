/*
1) Linear SVM (primal, via stochastic subgradient / Pegasos)

Purpose: find w that minimizes 0.5 * ||w||^2 + (λ) * hinge_loss over training set.
Complexity: simple, fast for large-scale linear problems.
Pseudocode (Pegasos — stochastic subgradient)

Input: training set {(x_i, y_i)}, y_i ∈ {+1, −1}; regularization λ > 0; iterations T; mini-batch size k.
Initialize w ← 0.
for t = 1 to T:
a. Pick a random mini-batch A of k examples.
b. Let η_t = 1 / (λ * t).
c. Let A+ = { (x_i,y_i) ∈ A : y_i * (w · x_i) < 1 } // examples violating margin
d. Compute subgradient step: w ← (1 − η_t * λ) * w + (η_t / k) * sum_{(x_i,y_i) ∈ A+} (y_i * x_i)
e. (Optional) Project: if ||w|| > 1/√λ then w ← w * (1 / (√λ * ||w||))
Output: w
Notes:

Hinge loss uses max(0, 1 − y * (w·x)).
Pegasos is simple and effective for linear SVMs; mini-batch k=1 is classic stochastic, larger k reduces variance. */


pub struct Svm{
    set: Vec<f64>, /* Dataset */
    delta: f64, /* Regularization */
    t: f64, /* Iterations */
    k: f64 /* mini-batch size */
}

impl Svm{
    fn initialize() {
        let w = 0;
    }

    pub fn new(_inputs_size :i32, _output_size:i32) -> Self{
        Self{}
    }
    
}

