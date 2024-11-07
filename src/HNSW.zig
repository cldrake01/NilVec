const std = @import("std");

pub fn HNSW() type {
    return struct {
        
        var assign_probas = std.ArrayList(f32).init(
        var cum_nneighbor_per_level = []u32{};
        
        fn init() HNSW {}
    
        fn defaultProbabilities(m: u32, level_mult: f32) void {
            // (int M, float levelMult) {
            //     int nn = 0;
            //     cum_nneighbor_per_level.push_back(0);
            //     for (int level = 0;; level++) {
            //         float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            //         if (proba < 1e-9)
            //             break;
            //         assign_probas.push_back(proba);
            //         nn += level == 0 ? M * 2 : M;
            //         cum_nneighbor_per_level.push_back(nn);
            //     }
            // }
    
            // var probabilities = []f32{};
        }
    };
}
