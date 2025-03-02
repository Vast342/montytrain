mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    logger, lr, operations, optimiser::{AdamWOptimiser, AdamWParams, Optimiser}, wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Shape, TrainingSchedule, TrainingSteps
};
use moves::NUM_MOVES;
use std::env;
use trainer::Trainer;

const ID: &str = "apn_009";

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let data_preparer = preparer::DataPreparer::new("./interleaved.bin", 4000);

    let size = 512;

    let mut graph = network(size);

    graph
        .get_weights_mut("l0w")
        .seed_random(0.0, 1.0 / (inputs::INPUT_SIZE as f32).sqrt(), true);
    graph
        .get_weights_mut("l1w")
        .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: AdamWOptimiser::new(graph, optimiser_params),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 35,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.01,
            final_lr: 0.0001,
            final_superbatch: 35,
        },
        save_rate: 5,
    };

    let settings = LocalSettings {
        threads: 12,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();
    println!("Arch                   : {}->{}->{}x{}", inputs::INPUT_SIZE, size, 1, NUM_MOVES);

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                trainer
                    .save_weights_portion(
                        &format!("checkpoints/{ID}.pn-{sb}"),
                        &["l0w", "l0b", "l1w", "l1b"],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize) -> Graph {
    let mut builder = GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let mask = builder.create_input("mask", Shape::new(moves::NUM_MOVES, 1));
    let dist = builder.create_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let l0w = builder.create_weights("l0w", Shape::new(size, inputs::INPUT_SIZE));
    let l0b = builder.create_weights("l0b", Shape::new(size, 1));

    let l1w = builder.create_weights("l1w", Shape::new(moves::NUM_MOVES, size));
    let l1b = builder.create_weights("l1b", Shape::new(moves::NUM_MOVES, 1));

    let l1 = operations::affine(&mut builder, l0w, inputs, l0b);
    let l1a = operations::activate(&mut builder, l1, Activation::SCReLU);
    let l2 = operations::affine(&mut builder, l1w, l1a, l1b);

    operations::sparse_softmax_crossentropy_loss_masked(&mut builder, mask, l2, dist);

    let ctx = ExecutionContext::default();
    builder.build(ctx)
}
