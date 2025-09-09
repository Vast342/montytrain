use montyformat::chess::{Move, Position};

pub const MAX_MOVES: usize = 64;
pub const NUM_MOVES_INDICES: usize = OFFSET_COUNT;

pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let from = mov.src() as usize;
    let to = mov.to() as usize;
    let from_bb = 1u64 << from as u64;
    let piece = pos.get_pc(from_bb) as usize - 2;
    let to_bb = 1u64 << to as u64;
    let mask = DESTINATIONS[from][piece];
    
    let relative_pos = (mask & (to_bb - 1)).count_ones() as usize;
    
    OFFSET_TABLE[from][piece] + relative_pos
}

const OFFSET_RAW: (usize, [[usize; 6]; 64]) = calc_offsets();

const OFFSET_COUNT: usize = OFFSET_RAW.0;
const OFFSET_TABLE: [[usize; 6]; 64] = OFFSET_RAW.1;

//                        index with piece and square
// maybe make it a different type if i need later
const fn calc_offsets() -> (usize, [[usize; 6]; 64]) {
    let mut sum = 0;
    let mut res = [[0; 6]; 64];
    
    let mut sq = 0;
    while sq < 64 {
        // no pawns on front or back rank
        let mut piece = if sq <= 7 || sq >= 56 {
            1
        } else {
            0
        };
        
        while piece < 6 {
            res[sq][piece] = sum;
            // get attack bitboard
            let bb: u64 = get_attacks(sq, piece);
            // add possible moves to sum
            sum += bb.count_ones() as usize;
            
            piece += 1;
        }
        sq += 1;
    }
    
    (sum, res)
}

const fn get_attacks(sq: usize, piece: usize) -> u64 {
    DESTINATIONS[sq][piece]
}


macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

const DESTINATIONS: [[u64; 6]; 64] = init!(|sq, 64| [PAWN[sq], KNIGHT[sq], bishop(sq), rook(sq), queen(sq), KING[sq]]);

const A: u64 = 0x0101_0101_0101_0101;
const H: u64 = A << 7;

const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

const PAWN: [u64; 64] = init!(|sq, 64| {
    let bit = 1 << sq;
    ((bit & !A) << 7) | (bit << 8) | ((bit & !H) << 9)
});

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const fn bishop(sq: usize) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;

    DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
}

const fn rook(sq: usize) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;

    (0xFF << (rank * 8)) ^ (A << file)
}

const fn queen(sq: usize) -> u64 {
    bishop(sq) | rook(sq)
}

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !A) >> 1) | ((k & !H) << 1);
    k ^ (1 << sq)
});
