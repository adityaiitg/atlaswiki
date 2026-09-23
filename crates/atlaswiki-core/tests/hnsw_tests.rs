use atlaswiki_core::hnsw::{
    BruteForceScan, HnswConfig, HnswIndex, Sq8Vector, VectorEvalSuite,
};

#[test]
fn test_sq8_quantization_and_reconstruction_error() {
    // Generate 1,000 synthetic 384-dimensional unit vectors
    let vectors = VectorEvalSuite::generate_synthetic_unit_vectors(1000, 384, 42);

    // Verify vector norms are ~1.0
    for v in &vectors {
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-4);
    }

    // Evaluate average cosine reconstruction error across 1,000 vector pairs
    let (avg_error, max_error) =
        VectorEvalSuite::evaluate_sq8_reconstruction_error(&vectors, 1000, 12345);

    println!(
        "SQ8 reconstruction error: avg = {:.6}, max = {:.6}",
        avg_error, max_error
    );

    // Assert average cosine reconstruction error < 0.02
    assert!(
        avg_error < 0.02,
        "Expected avg_error < 0.02, got {}",
        avg_error
    );

    // In fact, SQ8 symmetric reconstruction error should be < 0.005
    assert!(avg_error < 0.005);
}

#[test]
fn test_sq8_asymmetric_reconstruction() {
    let raw = vec![0.5f32, -0.5, 0.5, -0.5];
    let sq8 = Sq8Vector::encode(&raw);
    let decoded = sq8.decode();

    // Cosine similarity between raw and decoded should be > 0.999
    let cos: f32 = raw.iter().zip(decoded.iter()).map(|(&a, &b)| a * b).sum();
    assert!(cos > 0.999);

    // Asymmetric similarity query
    let asym_cos = sq8.cosine_similarity_f32(&raw);
    assert!((asym_cos - 1.0).abs() < 0.01);
}

#[test]
fn test_hnsw_edge_cases() {
    let mut index = HnswIndex::with_default_config();

    // 1. Search empty index
    let dummy_query = vec![0.0f32; 384];
    assert!(index.search(&dummy_query, 10, 16).is_empty());

    // 2. Single vector
    let v0 = vec![1.0f32; 384];
    let norm = (384.0f32).sqrt();
    let v0_norm: Vec<f32> = v0.into_iter().map(|x| x / norm).collect();
    index.insert(0, &v0_norm);

    let res = index.search(&v0_norm, 10, 16);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].0, 0);
    assert!((res[0].1 - 1.0).abs() < 0.01);

    // 3. Duplicate identical vectors
    index.insert(1, &v0_norm);
    let res = index.search(&v0_norm, 10, 16);
    assert_eq!(res.len(), 2);
}

#[test]
fn test_hnsw_recall_against_brute_force() {
    let dim = 384;
    let n = 1000;
    let num_queries = 20;

    let dataset = VectorEvalSuite::generate_synthetic_unit_vectors(n, dim, 100);
    let queries = VectorEvalSuite::generate_synthetic_unit_vectors(num_queries, dim, 200);

    let mut config = HnswConfig::default();
    config.ef_construction = 64;
    config.ef_search = 64;
    config.m = 16;
    config.m0 = 32;

    let mut index = HnswIndex::new(config);
    for (i, v) in dataset.iter().enumerate() {
        index.insert(i, v);
    }

    let mut recall_10_sum = 0.0f32;
    let mut recall_50_sum = 0.0f32;

    for q in &queries {
        let exact = BruteForceScan::search_fp32(&dataset, q, 50);
        let ann = index.search(q, 50, 64);

        recall_10_sum += VectorEvalSuite::compute_recall(&ann, &exact, 10);
        recall_50_sum += VectorEvalSuite::compute_recall(&ann, &exact, 50);
    }

    let avg_r10 = recall_10_sum / (num_queries as f32);
    let avg_r50 = recall_50_sum / (num_queries as f32);

    println!(
        "HNSW test: Recall@10 = {:.4}, Recall@50 = {:.4}",
        avg_r10, avg_r50
    );

    // High recall expected for ef_search=64
    assert!(
        avg_r10 >= 0.85,
        "Expected Recall@10 >= 0.85, got {}",
        avg_r10
    );
    assert!(
        avg_r50 >= 0.85,
        "Expected Recall@50 >= 0.85, got {}",
        avg_r50
    );
}
