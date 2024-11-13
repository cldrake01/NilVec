const std = @import("std");

pub fn HNSW() type {
    return struct {
        var assign_probas = std.ArrayList(f64).init(std.heap.ArenaAllocator);
        var cum_nneighbor_per_level = std.ArrayList(i32).init(std.heap.ArenaAllocator);

        fn init() HNSW {}

        // https://github.com/facebookresearch/faiss/blob/adb188411a98c3af5b7295c7016e5f46fee9eb07/faiss/impl/HNSW.cpp#L76
        // ```cpp
        // void HNSW::set_default_probas(int M, float levelMult) {
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
        // ```
        fn defaultProbabilities(m: u32, level_mult: f32) void {
            
            var nn: i32 = 0;
            cum_nneighbor_per_level.append(0);

            for (0..) |level| {
                const prob = @exp(-level / level_mult) * (1 - @exp(-1 / level_mult));
                if (prob < 1e-9)
                    break;
                assign_probas.append(prob);
                nn += if (level == 0) m * 2 else m;
            }
        }
    };
}
