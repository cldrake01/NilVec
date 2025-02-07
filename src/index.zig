const HNSW = @import("hnsw.zig").HNSW;
const Flat = @import("flat.zig").Flat;

pub fn Index(I: *anyopaque, comptime T: type, comptime N: usize) type {
    return struct {
        const Self = @This();

        index: I(T, N),

        pub fn init() Self {
            return Index(I, T, N){
                .index = I.init(T, N),
            };
        }
    };
}
