use bullet::{
    default::formats::montyformat::chess::{Castling, Position}, nn::{optimiser::{AdamWOptimiser, Optimiser}, Graph}, ExecutionContext, NetworkTrainer
};

use crate::preparer::PreparedData;

pub struct Trainer {
    pub optimiser: Optimiser<ExecutionContext, AdamWOptimiser>,
}

/*
    Bullet's internal stuff turns the fen into a datapoint and then evals that so uhhhhhh
    do with that information what you will
*/
impl Trainer {
    pub fn print_policy(&mut self, fen: &str, graph: &Graph) {
        println!("FEN: {fen}");
        let mut castling = Castling::default();
        let board = Position::parse_fen(fen, &mut castling);
        let mut moves = vec![];
        board.map_legal_moves(&castling, |mov| moves.push(mov));

        // guess i gotta figure out how to inference

        // softmax

        // sort

        // output
    }
}

impl NetworkTrainer for Trainer {
    type PreparedData = PreparedData;
    type OptimiserState = AdamWOptimiser;

    fn optimiser(&self) -> &Optimiser<ExecutionContext, Self::OptimiserState> {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Optimiser<ExecutionContext, Self::OptimiserState> {
        &mut self.optimiser
    }

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = &mut self.optimiser.graph;

        let inputs = &prepared.stm;
        unsafe {
            graph
                .get_input_mut("stm")
                .load_sparse_from_slice(inputs.max_active, Some(batch_size), &inputs.value)
                .unwrap();
        }

        let inputs = &prepared.stm;
        unsafe {
            graph
                .get_input_mut("ntm")
                .load_sparse_from_slice(inputs.max_active, Some(batch_size), &inputs.value)
                .unwrap();
        }

        let mask = &prepared.mask;
        unsafe {
            graph
                .get_input_mut("mask")
                .load_sparse_from_slice(mask.max_active, Some(batch_size), &mask.value)
                .unwrap();
        }

        let dist = &prepared.dist;
        graph
            .get_input_mut("dist")
            .load_dense_from_slice(Some(batch_size), &dist.value)
            .unwrap();

        batch_size
    }
}
