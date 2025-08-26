/*
    Anura's value net configuration, as retyped from memory by me right now
*/
use bullet::{
    nn::optimiser,
    trainer::{
        default::{inputs, loader, outputs},
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};
//use viriformat::dataformat::Filter;

/*
- first ob dataset DONE
- more superbatches (400 -> 500) DONE
- finetuning step lost
- threat inputs(may be hard with this config)
- warmup maybe? (this'll be later since i have doubts about it)
- different wdl (used constant 0.5 this whole time so far, 2 storys say 1.0 might be better)
- different lr schedule (heard good things about linear decay instead of cosine decay) DONE
- filtering (maybe need to train with viriformat for a bit, resplatting with different filters would take a while
might not be filtering mate scores rn though which i probably should do)
- whatever else I want to experiment with

then:
- more layers or bigger HL
 */

fn main() {
    const HL_SIZE: usize = 1024;
    const OUTPUT_BUCKET_COUNT: usize = 16;

    const TOTAL_SUPERBATCHES: usize = 600;

    const QA: i16 = 256;
    const QB: i16 = 64;

    #[rustfmt::skip]
    let mut trainer = ValueTrainerBuilder::default()
        .optimiser(optimiser::AdamW)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .single_perspective()
        .inputs(inputs::ChessBucketsMirrored::new([
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 
        ]))
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKET_COUNT>)
        .build(|builder, stm_inputs, output_buckets| {
            // weights
            let l0 = builder.new_affine("l0", 768, HL_SIZE);
            let l1 = builder.new_affine("l1", HL_SIZE, OUTPUT_BUCKET_COUNT);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            l1.forward(stm_hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: "avn_014_dot3".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: TOTAL_SUPERBATCHES,
        },

        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3 * 0.3, final_superbatch: TOTAL_SUPERBATCHES },
        save_rate: 50,
    };

    let optimiser_params =
        optimiser::AdamWParams { decay: 0.01, beta1: 0.95, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };
    trainer.optimiser.set_params(optimiser_params);

    // when using viriformat, set threads to 2, and dataloader threads to 4
    let settings =
        LocalSettings { threads: 6, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    //let data_loader = loader::DirectSequentialDataLoader::new(&["C:/Users/Joseph/Downloads/anura-data-shuffled.bin"]);
    // ./target/release/bullet-utils.exe viribinpack splat D:\bullet\anura-datagen.vf C:\Users\Joseph\Downloads\anura-datagen.bf .
    let data_loader = loader::DirectSequentialDataLoader::new(&["C:/Users/Joseph/Downloads/anura-datagen-notr-s.bf"]);

    // currently unrestricted filter
    /*let data_loader = loader::ViriBinpackLoader::new("D:/bullet/anura-datagen-notr.vf", 4096, 4   , Filter {
        min_ply: 0,
        min_pieces: 0,
        max_eval: u32::MAX,
        filter_tactical: false,
        filter_check: false,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,   
        random_fen_skipping: false,
        random_fen_skip_probability: 0.0,
        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.0,
    });*/

    //trainer.load_from_checkpoint("checkpoints/avn_011-wdl-180");
    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
