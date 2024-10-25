mod index;
mod metric;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn index_from() {
        use crate::index::Index;
        use std::collections::{BTreeMap, HashMap};
        
        const DIM: u64 = 128;
        
        // Example using Vec<(String, u64)>
        let vec_table = vec![("attribute1".to_string(), 3), ("attribute2".to_string(), 5)];
        let _index_from_vec = Index::new(DIM, vec_table);

        // Example using HashMap<String, u64>
        let mut hash_table = HashMap::new();
        hash_table.insert("attribute1".to_string(), 3);
        hash_table.insert("attribute2".to_string(), 5);
        let _index_from_hash = Index::new(dim, hash_table);

        // Example using BTreeMap<String, u64>
        let mut btree_table = BTreeMap::new();
        btree_table.insert("attribute1".to_string(), 3);
        btree_table.insert("attribute2".to_string(), 5);
        let _index_from_btree = Index::new(dim, btree_table);

        println!("Indexes created successfully!");
    }
}
