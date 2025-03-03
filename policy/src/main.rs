mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    nn::{
        optimiser::{AdamWParams, Optimiser},
        Activation, ExecutionContext, Graph, NetworkBuilder, Shape,
    },
    trainer::{
        logger,
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
};

use trainer::Trainer;

const ID: &str = "cpn_001";

fn main() {
    let data_preparer = preparer::DataPreparer::new("montydata/montydata.binpack", 4096);

    let size = 128;

    let graph = network(size);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.95,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: Optimiser::new(graph, optimiser_params).unwrap(),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 100,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.00001,
            final_superbatch: 100,
        },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 6,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                trainer
                    .save_weights_portion(
                        &format!("checkpoints/{ID}.pn"),
                        &[
                            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new(
                                "l1w",
                                QuantTarget::Float,
                                Layout::Transposed(Shape::new(moves::NUM_MOVES, size)),
                            ),
                            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize) -> Graph {
    let builder = NetworkBuilder::default();

    let stm =
        builder.new_sparse_input("stm", Shape::new(inputs::INPUT_SIZE, 1), inputs::MAX_ACTIVE);
    let ntm =
        builder.new_sparse_input("ntm", Shape::new(inputs::INPUT_SIZE, 1), inputs::MAX_ACTIVE);
    let mask = builder.new_sparse_input("mask", Shape::new(moves::NUM_MOVES, 1), moves::MAX_MOVES);
    let dist = builder.new_dense_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", size, moves::NUM_MOVES);

    let mut out = l0.forward_sparse_dual_with_activation(stm, ntm, Activation::CReLU);
    out = out.pairwise_mul_post_affine_dual();
    out = l1.forward(out);
    out.masked_softmax_crossentropy_loss(dist, mask);

    builder.build(ExecutionContext::default())
}
