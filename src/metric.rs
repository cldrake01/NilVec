/// Available distance metrics.
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    L2,
    Cosine,
    InnerProduct,
}
