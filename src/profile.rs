use nilvec::flat::Flat;
use rand::Rng;
use std::time::{Duration, Instant};

fn generate_random_vector(dim: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..dim)
        .map(|_| rng.random_range(-1.0..1.0) as f64)
        .collect()
}

fn main() {
    let dim = 128; // Vector dimensionality
    let num_vectors = 1_000;
    let query_interval = Duration::from_secs(1);

    println!("Initializing vector database...");
    // let mut index = HNSW::new(dim, None, None, None, None, None, None);
    let mut index = Flat::new(dim, None, None);

    let start_time = Instant::now();

    // Insert vectors
    for i in 0..num_vectors {
        let vector = generate_random_vector(dim);

        // HNSW
        // index.insert(&vector, None, None, &mut rng).unwrap();
        // Flat
        index.insert(&vector, None).unwrap();

        if i % 100 == 0 {
            println!("Inserted {} vectors...", i);
        }
    }

    let insert_duration = start_time.elapsed();
    println!(
        "Inserted {} vectors in {:.2} seconds.",
        num_vectors,
        insert_duration.as_secs_f32()
    );

    // Querying
    println!("Starting query profiling...");
    let query_vector = generate_random_vector(dim);
    let query_start = Instant::now();

    loop {
        let results = index.search(&query_vector, Some(10), None).unwrap();
        println!(
            "Top result: ID={} | Distance={:.4}",
            results[0].id, results[0].distance
        );

        std::thread::sleep(query_interval);

        if query_start.elapsed().as_secs() > 10 {
            println!("Profiling completed.");
            break;
        }
    }
}
