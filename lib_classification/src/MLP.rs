use libm::{self, pow};
use rand::{prelude::*, random_range};

pub struct Mlp{
    d: Vec<i32>,
    length: usize,
    weights: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>
}

impl Mlp{
    
    // MLP INITIALIZATION

    pub fn new(_inputs_size: i32,hidden_layers: Vec<i32>,_output_size: i32,)->Self{
        // creation of the list of inputs
        let mut d: Vec<i32> = Vec::new();
        d.push(_inputs_size);
        for (_size,layer) in hidden_layers.iter().enumerate() {
            let layer_int : i32 = *layer;
            d.push(layer_int);
        }
        d.push(_output_size);

        // creation of the length of the list of inputs
        let length: usize = d.len() - 1;

        // initialization of the weights matrix
        let mut weights: Vec<Vec<Vec<f64>>>  = Vec::new();
        for l in 0..length+1{
            let empty_list : Vec<Vec<f64>> = Vec::new();
            weights.push(empty_list);
            if l == 0{
                continue;
            }
            for i in 0..d[l - 1] + 1{
                let index:usize = i.try_into().unwrap();
                let new_empty_list : Vec<f64> = Vec::new();
                weights[l].push(new_empty_list);
                for j in 0..d[l] + 1{
                    if j == 0{
                        weights[l][index].push(0.0);
                    }
                    else{
                        let mut rng = rand::rng();
                        let random_number:f64 = rng.random();
                        weights[l][index].push(random_number * 2.0 - 1.0);
                    }
                }
            }
        }

        // initialization of the values and deltas matrix
        let mut values: Vec<Vec<f64>> = Vec::new();
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        for l in 0..length + 1{
            let v_empty_list: Vec<f64> = Vec::new();
            let d_empty_list: Vec<f64> = Vec::new();
            values.push(v_empty_list);
            deltas.push(d_empty_list);
            for j in 0..d[l] + 1{
                deltas[l].push(0.0);
                if j == 0{
                    values[l].push(1.0);
                }
                else {
                    values[l].push(0.0);
                }
            }
        }
        
        Self { d, length, weights, values, deltas }
    }

    // GRADIENT PROPAGATION

    pub fn propagation(&mut self,
        input: Vec<f64>,
        classification: bool){
        
        for j in 1..self.d[0] + 1{
            let j_index:usize = j.try_into().unwrap();
            self.values[0][j_index] = input[j_index - 1];
        }

        for l in 1..self.length + 1{
            for j in 1..self.d[l] + 1{
                let j_index:usize =j.try_into().unwrap();
                let mut total:f64 = 0.0;
                for i in 0..self.d[l-1] +1{
                    let i_index:usize =i.try_into().unwrap();
                    total += self.weights[l][i_index][j_index] * self.values[l-1][i_index];
                }
                if classification || l < self.length{
                    total = libm::tanh(total);
                }
                self.values[l][j_index] = total;
            }
        }
    }

    // VALUE PREDICTION (using propagation on a single element)

    pub fn prediction(&mut self, input: Vec<f64>, classification: bool) -> &[f64]{
        self.propagation(input, classification);
        let total = &self.values[self.length][1..];
        return total;
    }

    // MODEL TRAINING

    pub fn training(&mut self,
        dataset_inputs: Vec<Vec<f64>>,
        dataset_expected_output: Vec<Vec<f64>>,
        training_step: i32,
        learning_rate: f64,
        classification: bool){
        
        for _ in 0..training_step{
            let k = random_range(0..dataset_inputs.len());
            let inputs_k = &dataset_inputs[k];
            let y_k = &dataset_expected_output[k];

            self.propagation(inputs_k.to_vec(), classification);
            for j in 1..self.d[self.length] + 1{
                let j_index:usize = j.try_into().unwrap();
                self.deltas[self.length][j_index] = self.values[self.length][j_index] - y_k[j_index - 1];
                if classification{
                    self.deltas[self.length][j_index] *= 1.0 - pow(self.values[self.length][j_index], 2.0);
                }
            }

            for l in (2..self.length + 1).rev(){
                for i in 1..self.d[l-1] + 1{
                    let i_index: usize = i.try_into().unwrap();
                    let mut total = 0.0;
                    for j in 1..self.d[l] + 1{
                        let j_index: usize = j.try_into().unwrap();
                        total += self.weights[l][i_index][j_index] * self.deltas[l][j_index];
                    }
                    total *= 1.0 - pow(self.values[l-1][i_index],2.0);
                    self.deltas[l-1][i_index] = total;
                }
            }

            for l in 1..self.length + 1{
                for i in 0..self.d[l-1] +1{
                    let i_index: usize = i.try_into().unwrap();
                    for j in 1..self.d[l] + 1{
                        let j_index: usize = j.try_into().unwrap();
                        self.weights[l][i_index][j_index] -= learning_rate * self.values[l-1][i_index] * self.deltas[l][j_index];                        
                    }
                }
            }
        }
    }

    pub fn input_size(&self) -> usize {
        self.d[0].try_into().unwrap()
    }

    pub fn output_size(&self) -> usize {
        self.d[self.length].try_into().unwrap()
    }
}
