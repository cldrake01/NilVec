use ordered_float::OrderedFloat;

/// A candidate for nearest‐neighbor search.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    pub distance: OrderedFloat<f64>,
    pub id: usize,
}
