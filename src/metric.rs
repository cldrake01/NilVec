pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

impl Metric {
    pub fn distance(&self, left: &[f32], right: &[f32]) -> f32 {
        match self {
            Metric::Euclidean => euclidean(left, right),
            Metric::Cosine => cosine(left, right),
            Metric::DotProduct => dot_product(left, right),
        }
    }
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(l, r)| (l - r).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b.iter()).map(|(l, r)| l * r).sum::<f32>();
    let left_magnitude = a.iter().map(|l| l.powi(2)).sum::<f32>().sqrt();
    let right_magnitude = b.iter().map(|r| r.powi(2)).sum::<f32>().sqrt();
    dot_product / (left_magnitude * right_magnitude)
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(l, r)| l * r).sum()
}