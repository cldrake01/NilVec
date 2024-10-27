use crate::metric::Metric;
use std::cmp::Ord;
use std::mem::size_of;

/// The index represents an HNSW vector index.
/// The user must provide the dimension of the
/// vectors to be indexed, and a table containing
/// strings and types.
/// Ideally, the table should be any iterable
/// collection of tuples, where the first element
/// is a string and the second element is a type.
/// For example, a table could be a `Vec<(String, T)>`,
/// a `HashMap<String, T>`, or a `BTreeMap<String, T>`.
pub struct Index<Table> {
    dim: u64,
    table: Table,
    metric: Option<Metric>,
}

impl<Table, Element> Index<Table>
where
    Table: IntoIterator<Item = (String, Element)> + Clone,
    Element: std::fmt::Debug + Iterator + std::fmt::Display,
{
    /// Constructs a new index with the given dimension and table.
    pub fn new(dim: u64, meta: Table) -> Result<Self, String> {
        assert_ne!(dim, 0, "Dimension cannot be zero");

        let calc_dim = dim + meta.clone().into_iter().count() as u64;

        Ok(Index::<Table> {
            dim: calc_dim,
            table: meta,
            metric: None, // Add default metric if needed
        })
    }

    /// The user may supply us a table with all
    /// manner of types. Our job is to order the
    /// table in a way that is most efficient in
    /// terms of memory footprint and speed.
    /// Moreover, these attributes will be
    /// stored directly within each vector;
    /// they need to be ordered in the most
    /// compact way possible. Remember, accesses
    /// are done 8 bytes (64 bits) at a time.
    /// So, if we can order the table in a way
    /// that minimizes cache misses, we can
    /// speed up our search.
    fn pack(meta: Table) {
        // Load the elements into a stack
        let mut packed = meta
            .into_iter()
            .map(|(k, _)| (k, size_of::<Element>() - 8))
            .collect::<Vec<_>>();

        // Sort by size of element in ascending order
        packed.sort_by(|a, b| Ord::cmp(&a.1, &b.1));
     }
}
