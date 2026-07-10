use libm::{self, pow};
use rand::{prelude::*, random_range};
use rand::{SeedableRng};
use rand::rngs::StdRng;

pub struct Mlp{
    d: Vec<usize>,
    length: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
    pub loss_history: Vec<f64>,
    pub accuracy_history: Vec<f64>,
    epoch: i32
}

impl Mlp{
    
    // MLP INITIALIZATION

    pub fn new(_inputs_size: usize,hidden_layers: Vec<usize>,_output_size: usize,seed : u64)->Self{
        // creation of the list of inputs
        let mut d= Vec::new();
        d.push(_inputs_size);
        for layer in hidden_layers {
            d.push(layer);
        }
        d.push(_output_size);

        // Using a seed to make this experiment repeatable
        let mut rng = StdRng::seed_from_u64(seed);

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
                let new_empty_list : Vec<f64> = Vec::new();
                weights[l].push(new_empty_list);
                for j in 0..d[l] + 1{
                    if j == 0{
                        weights[l][i].push(0.0);
                    }
                    else{
                        let random_number:f64 = rng.random();
                        weights[l][i].push(random_number * 2.0 - 1.0);
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

        let epoch = 1;
        
        Self { d, length, weights, values, deltas, loss_history: Vec::new(), accuracy_history: Vec::new(), epoch}
    }

    // GRADIENT PROPAGATION

    pub fn propagation(&mut self,
        input: &[f64],
        classification: bool){
        
        for j in 1..self.d[0] + 1{
            self.values[0][j] = input[j - 1];
        }

        for l in 1..self.length + 1{
            for j in 1..self.d[l] + 1{
                let mut total:f64 = 0.0;
                for i in 0..self.d[l-1] +1{
                    total += self.weights[l][i][j] * self.values[l-1][i];
                }
                if classification || l < self.length{
                    total = libm::tanh(total);
                }
                self.values[l][j] = total;
            }
        }
    }

    // VALUE PREDICTION (using propagation on a single element)

    pub fn prediction(&mut self, input: &[f64], classification: bool) -> &[f64]{
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

        let dataset_inputs_size: i32 = i32::try_from(dataset_inputs.len()).unwrap();
        
        for train_index in 0..training_step{
            let k = random_range(0..dataset_inputs.len());
            let inputs_k = &dataset_inputs[k];
            let y_k = &dataset_expected_output[k];

            self.propagation(inputs_k, classification);
            for j in 1..self.d[self.length] + 1{
                self.deltas[self.length][j] = self.values[self.length][j] - y_k[j - 1];
                if classification{
                    self.deltas[self.length][j] *= 1.0 - pow(self.values[self.length][j], 2.0);
                }
            }

            for l in (2..self.length + 1).rev(){
                for i in 1..self.d[l-1] + 1{
                    let i_index: usize = i;
                    let mut total = 0.0;
                    for j in 1..self.d[l] + 1{
                        total += self.weights[l][i_index][j] * self.deltas[l][j];
                    }
                    total *= 1.0 - pow(self.values[l-1][i_index],2.0);
                    self.deltas[l-1][i_index] = total;
                }
            }

            for l in 1..self.length + 1{
                for i in 0..self.d[l-1] +1{
                    for j in 1..self.d[l] + 1{
                        self.weights[l][i][j] -= learning_rate * self.values[l-1][i] * self.deltas[l][j];                        
                    }
                }
            }

            if train_index % dataset_inputs_size == 0{
                // calculer loss / accuracy
                let loss = self.compute_loss(&dataset_inputs, &dataset_expected_output, classification,);
                self.loss_history.push(loss);

                if classification {
                    let acc = self.compute_accuracy(&dataset_inputs, &dataset_expected_output, classification);

                    self.accuracy_history.push(acc);
                        
                    self.epoch +=1;
                }
            }
        }
        }

    // CALCUL DU LOSS SUR UN NEURONE
    pub fn compute_loss(&mut self,
        inputs: &[Vec<f64>],
        targets:&[Vec<f64>],
        classification: bool) -> f64{
        let mut total_loss = 0.0;

        for (input, expected_output) in inputs.iter().zip(targets.iter()) {

            let prediction = self.prediction(input, classification);

            for (predicted_value, expected_value) in
                prediction.iter().zip(expected_output.iter())
            {
                let error = predicted_value - expected_value;
                total_loss += error * error;
            }
        }

        total_loss / inputs.len() as f64
        }

    // CALCUL DE L'ACCURACY SUR UN NEURONE
    fn compute_accuracy(&mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        classification: bool,
        ) -> f64 {

        let mut correct = 0;

        for (x, y) in inputs.iter().zip(targets.iter()) {

            let prediction = self.prediction(x, classification);

            let predicted_class = argmax(prediction);
            let true_class = argmax(y);

            if predicted_class == true_class {
                correct += 1;
            }
        }

        correct as f64 / inputs.len() as f64
        }

    pub fn flattened_weights(&self) -> Vec<f64> {
        self.weights
        .iter()
        .flat_map(|layer| {
            layer.iter().flat_map(|neuron| neuron.iter().copied())
        })
        .collect()
    }

    pub fn flattened_deltas(&self) ->Vec<f64> {
        self.deltas
        .iter()
        .flat_map(|neuron| neuron.iter().copied())
        .collect()
    }

    pub fn input_size(&self) -> usize {
        self.d[0]
    }

    pub fn output_size(&self) -> usize {
        self.d[self.length]
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best_index = 0;
    let mut best_value = values[0];

    for (i, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = i;
        }
    }

    best_index
}
