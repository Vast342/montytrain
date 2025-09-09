use bullet::game::formats::montyformat::chess::Attacks;
use bullet::game::formats::montyformat::chess::Piece;
use bullet::game::formats::montyformat::chess::Side;
use bulletformat::ChessBoard;

use bullet::game::inputs::SparseInputType;

#[derive(Clone, Copy, Debug)]
pub struct ChessMirroredThreats;

impl Default for ChessMirroredThreats {
    fn default() -> Self {
        Self
    }
}

impl ChessMirroredThreats {
    pub fn new() -> Self {
        Self
    }
}

impl SparseInputType for ChessMirroredThreats {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768 * 4
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get_flip = |ksq| if ksq % 8 > 3 { 7 } else { 0 };
        let stm_flip = get_flip(pos.our_ksq());
        let ntm_flip = get_flip(pos.opp_ksq());

        self.map_features_internal(pos, |stm, ntm| f(stm ^ stm_flip, ntm ^ ntm_flip));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        format!("3072")
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Horizontally mirrored psqt chess inputs with simple threats".to_string()
    }
}

impl ChessMirroredThreats {
    fn map_features_internal<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }
        let pos2 = InternalPosition::from_bbs(bbs);

        let threats = pos2.threats_by(0);
        let defences = pos2.threats_by(1);
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);

            let stm = [0, 384][c] + pc + sq;
            let ntm = [384, 0][c] + pc + (sq ^ 56);
            let mut threat_bonus = 0;
            if threats & (1 << sq) > 0 {
                threat_bonus += 768;
            }
            if defences & (1 << sq) > 0 {
                threat_bonus += 768 * 2;
            }
            f(stm + threat_bonus, ntm + threat_bonus)
        }
    }
}

// jerry-rigged montyformat position for threats
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct InternalPosition {
    bb: [u64; 8],
}

macro_rules! bitloop {
    (| $bb:expr, $sq:ident | $func:expr) => {{
        let mut bb = $bb;

        while bb > 0 {
            let $sq = bb.trailing_zeros() as u16;
            bb &= bb - 1;

            $func;
        }
    };};
}

impl InternalPosition {
    fn from_bbs(bb: [u64; 8]) -> Self {
        Self { bb }
    }

     #[must_use]
    pub fn piece(&self, piece: usize) -> u64 {
        self.bb[piece]
    }

    #[must_use]
    pub fn occ(&self) -> u64 {
        self.bb[Side::WHITE] | self.bb[Side::BLACK]
    }

    #[must_use]
    pub fn threats_by(&self, side: usize) -> u64 {
        let mut threats = 0;

        let king = self.piece(Piece::KING) & self.bb[side ^ 1];
        let occ = self.occ() ^ king;

        let opps = self.bb[side];

        let queens = self.bb[Piece::QUEEN];

        bitloop!(|opps & (self.bb[Piece::ROOK] | queens), sq| threats |= Attacks::rook(sq as usize, occ));
        bitloop!(|opps & (self.bb[Piece::BISHOP] | queens), sq| threats |= Attacks::bishop(sq as usize, occ));
        bitloop!(|opps & self.bb[Piece::KNIGHT], sq| threats |= Attacks::knight(sq as usize));
        bitloop!(|opps & self.bb[Piece::KING], sq| threats |= Attacks::king(sq as usize));

        let pawns = opps & self.bb[Piece::PAWN];
        threats |= if side == Side::WHITE {
            Attacks::white_pawn_setwise(pawns)
        } else {
            Attacks::black_pawn_setwise(pawns)
        };

        threats
    }
}